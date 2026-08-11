"""Attention-aware MIL replay for the continual WSI benchmark.

AMIL keeps a class-balanced pool of compact pseudo-bags.  The pool stores
attention and classifier outputs produced by the best model at the end of the
previous session, so replay does not require a second, frozen model.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.utils.amil_memory import (
    PseudoBag,
    PseudoBagMemoryPool,
    maxminrand_select,
)
from models.utils.continual_model import ContinualModel
from models.utils.wsi_replay import unpack_prepared_batch
from utils.args import add_experiment_args, add_management_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Attention-aware MIL continual learning")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--minibatch_size", type=int, default=1)
    parser.add_argument("--bags_per_update", type=int, default=1)
    parser.add_argument("--pmp_k", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    return parser


def _validate_args_without_stream(args) -> None:
    """Validate everything available before a dataset/backbone is built."""

    backbone = str(getattr(args, "backbone", "generic_mil")).lower()
    if backbone not in Amil.SUPPORTED_BACKBONES:
        raise ValueError(
            f"AMIL supports only {Amil.SUPPORTED_BACKBONES}, got backbone={backbone!r}"
        )
    feature_dim = int(getattr(args, "feature_dim", 768))
    if feature_dim <= 0:
        raise ValueError("AMIL feature_dim must be positive")
    if backbone == "feather" and feature_dim != 768:
        raise ValueError("AMIL with FEATHER requires 768-D patch features")
    if bool(getattr(args, "backbone_freeze", False)):
        raise ValueError("AMIL requires a trainable slide backbone")
    if int(getattr(args, "backbone_max_patches", 0) or 0) != 0:
        raise ValueError(
            "AMIL requires backbone_max_patches=0 so PMP receives each full WSI"
        )

    positive_integer_args = {
        "buffer_size": 30,
        "minibatch_size": 1,
        "bags_per_update": 1,
        "pmp_k": 400,
    }
    for name, default in positive_integer_args.items():
        if int(getattr(args, name, default)) <= 0:
            raise ValueError(f"AMIL {name} must be positive")
    positive_float_args = {
        "alpha": 1.0,
        "beta": 1.0,
        "kd_temperature": 1.0,
    }
    for name, default in positive_float_args.items():
        if float(getattr(args, name, default)) <= 0:
            raise ValueError(f"AMIL {name} must be positive")


def validate_args(args) -> None:
    """Fail before pretrained backbone resolution for invalid AMIL settings."""

    _validate_args_without_stream(args)


class Amil(ContinualModel):
    """PMP replay with attention and logit knowledge distillation."""

    NAME = "amil"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("generic_mil", "feather")
    REQUIRES_TRAINABLE_BACKBONE = True
    CHECKPOINT_VERSION = 1

    def __init__(self, backbone, loss, args, transform):
        self._validate_configuration(backbone, args)
        super().__init__(backbone, loss, args, transform)

        self.num_classes = int(args.num_classes)
        self.feature_dim = int(getattr(args, "feature_dim", 768))
        self.minibatch_size = int(getattr(args, "minibatch_size", 1))
        self.bags_per_update = int(getattr(args, "bags_per_update", 1))
        self.pmp_k = int(getattr(args, "pmp_k", 400))
        self.alpha = float(getattr(args, "alpha", 1.0))
        self.beta = float(getattr(args, "beta", 1.0))
        self.kd_temperature = float(getattr(args, "kd_temperature", 1.0))

        self.task_num_classes = tuple(
            int(value) for value in getattr(args, "task_num_classes", ())
        )
        self.class_offsets = tuple(
            int(value) for value in getattr(args, "class_offsets", ())
        )
        self._validate_task_metadata()

        seed = getattr(args, "seed", 0)
        self.buffer = PseudoBagMemoryPool(
            int(getattr(args, "buffer_size", 30)),
            pmp_k=self.pmp_k,
            seed=0 if seed is None else int(seed),
            num_classes=self.num_classes,
        )
        self.current_task = 0
        self.old_class_count = 0
        self.current_seen_class_count = self.task_num_classes[0]
        # This phase is deliberately not checkpointable.  Framework
        # checkpoints are written either before save_buffer or after end_task.
        self._buffer_update_started = False

    @classmethod
    def _validate_configuration(cls, backbone, args) -> None:
        _validate_args_without_stream(args)
        if int(getattr(args, "num_classes", 0)) <= 0:
            raise ValueError("AMIL requires a positive global class count")
        if not bool(getattr(backbone, "has_genuine_patch_attention", False)):
            raise ValueError(
                "AMIL requires genuine patch attention; uniform/synthetic attention "
                "is not supported"
            )

    def _validate_task_metadata(self) -> None:
        if not self.task_num_classes or len(self.task_num_classes) != len(
            self.class_offsets
        ):
            raise ValueError(
                "AMIL requires matching task_num_classes and class_offsets metadata"
            )
        expected_offset = 0
        for task, (offset, count) in enumerate(
            zip(self.class_offsets, self.task_num_classes)
        ):
            if count <= 0 or offset != expected_offset:
                raise ValueError(
                    "AMIL requires positive task sizes and contiguous prefix class "
                    f"offsets; task {task} has offset={offset}, count={count}, "
                    f"expected_offset={expected_offset}"
                )
            expected_offset += count
        if expected_offset != self.num_classes:
            raise ValueError(
                "AMIL task metadata does not cover the global classifier: "
                f"task_total={expected_offset}, num_classes={self.num_classes}"
            )

    def _task_boundaries(self, task: int) -> Tuple[int, int]:
        task = int(task)
        if task < 0 or task >= len(self.task_num_classes):
            raise ValueError(f"AMIL task index {task} is outside the stream metadata")
        old_count = self.class_offsets[task]
        seen_count = old_count + self.task_num_classes[task]
        return old_count, seen_count

    def _assert_previous_snapshot(self, task: int) -> None:
        if self.buffer.refresh_required:
            raise RuntimeError("AMIL memory has an unfinished cached-target refresh")
        snapshot_task = self.buffer.target_snapshot_task
        if task == 0:
            if len(self.buffer) != 0 or snapshot_task not in (None, -1):
                raise RuntimeError("AMIL task 0 requires an empty pseudo-bag memory")
            return
        if len(self.buffer) == 0:
            raise RuntimeError(
                f"AMIL task {task} requires a memory snapshot from task {task - 1}"
            )
        if snapshot_task != task - 1:
            raise RuntimeError(
                "AMIL memory snapshot is stale: "
                f"saved_task={snapshot_task!r}, expected={task - 1}"
            )
        # This also validates that no retained entry is missing cached targets.
        self.buffer.all(device="cpu", require_targets=True)

    def _configure_task(self, task: int, *, require_previous_snapshot: bool) -> None:
        old_count, seen_count = self._task_boundaries(task)
        if require_previous_snapshot:
            self._assert_previous_snapshot(int(task))
        self.current_task = int(task)
        self.old_class_count = old_count
        self.current_seen_class_count = seen_count

    def begin_task(self, dataset) -> None:
        if self._buffer_update_started:
            raise RuntimeError("AMIL cannot begin a task during a memory update")
        task = max(0, int(getattr(dataset, "current_task", 1)) - 1)
        self._configure_task(task, require_previous_snapshot=True)

    def _unpack_batch(
        self, batch: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return unpack_prepared_batch(batch, feature_dim=self.feature_dim)

    def _forward_attention(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.net([features, coords, patch_size])
        if not isinstance(output, (tuple, list)) or len(output) != 5:
            raise TypeError(
                "AMIL backbone must return the five-item MIL output contract"
            )
        logits, attention = output[0], output[3]
        if not torch.is_tensor(logits) or logits.shape != (1, self.num_classes):
            shape = tuple(logits.shape) if torch.is_tensor(logits) else None
            raise ValueError(
                f"AMIL expects logits [1,{self.num_classes}], got {shape}"
            )
        patch_count = int(features.shape[0])
        if not torch.is_tensor(attention) or attention.shape != (1, patch_count):
            shape = tuple(attention.shape) if torch.is_tensor(attention) else None
            raise ValueError(
                f"AMIL expects genuine attention [1,{patch_count}], got {shape}"
            )
        logits = logits.float()
        attention = attention.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("AMIL backbone returned non-finite logits")
        if not torch.isfinite(attention).all():
            raise FloatingPointError("AMIL backbone returned non-finite attention")
        if torch.any(attention < 0):
            raise ValueError("AMIL attention weights must be non-negative")
        total = attention.sum(dim=1, keepdim=True)
        if torch.any(total <= 0):
            raise ValueError("AMIL attention must have positive mass")
        return logits, attention / total

    @staticmethod
    def _stable_distribution(probabilities: torch.Tensor) -> torch.Tensor:
        probabilities = probabilities.float()
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError(
                "AMIL cached/student distributions must have shape [1,N]"
            )
        if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
            raise ValueError("AMIL distributions must be finite and non-negative")
        epsilon = torch.finfo(probabilities.dtype).eps
        probabilities = probabilities.clamp_min(epsilon)
        return probabilities / probabilities.sum(dim=1, keepdim=True)

    def attention_distillation_loss(
        self, student_attention: torch.Tensor, target_attention: torch.Tensor
    ) -> torch.Tensor:
        if student_attention.shape != target_attention.shape:
            raise ValueError(
                "AMIL student/cached attention shapes differ: "
                f"{tuple(student_attention.shape)} vs {tuple(target_attention.shape)}"
            )
        student = self._stable_distribution(student_attention)
        target = self._stable_distribution(
            target_attention.to(student.device, dtype=torch.float32)
        )
        return F.kl_div(student.log(), target, reduction="batchmean")

    def logit_distillation_loss(
        self,
        student_logits: torch.Tensor,
        target_logits: torch.Tensor,
        target_seen_class_count: int,
    ) -> torch.Tensor:
        count = int(target_seen_class_count)
        if count <= 0 or count > self.current_seen_class_count:
            raise ValueError(
                "AMIL cached target class count must be within the current seen prefix"
            )
        if student_logits.shape != (1, self.num_classes):
            raise ValueError("AMIL student logits have an invalid shape")
        if not torch.is_tensor(target_logits) or target_logits.shape != (
            1,
            self.num_classes,
        ):
            raise ValueError("AMIL cached logits have an invalid shape")
        target_logits = target_logits.to(student_logits.device, dtype=torch.float32)
        if not torch.isfinite(target_logits).all():
            raise ValueError("AMIL cached logits contain NaN or Inf")
        temperature = self.kd_temperature
        student_log_probability = F.log_softmax(
            student_logits[:, :count].float() / temperature, dim=1
        )
        target_probability = F.softmax(
            target_logits[:, :count] / temperature, dim=1
        )
        # Deliberately no T^2 scaling: T=1 is the benchmark setting and this
        # follows the direct KL convention used by the method specification.
        return F.kl_div(
            student_log_probability, target_probability, reduction="batchmean"
        )

    def _classification_loss(
        self, logits: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        label = label.long().reshape(-1)
        if label.numel() != 1:
            raise ValueError("AMIL expects one label per WSI bag")
        value = int(label.detach().cpu().item())
        if value < 0 or value >= self.current_seen_class_count:
            raise ValueError(
                f"AMIL label {value} is outside the current seen-class prefix"
            )
        return F.cross_entropy(
            logits[:, : self.current_seen_class_count].float(), label
        )

    def _replay_losses(
        self, entry: PseudoBag
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if entry.target_attention is None or entry.target_logits is None:
            raise RuntimeError("AMIL cannot replay a pseudo-bag without cached targets")
        logits, attention = self._forward_attention(
            entry.features, entry.coords, entry.patch_size
        )
        loss_ce = self._classification_loss(logits, entry.label)
        loss_attention = self.attention_distillation_loss(
            attention, entry.target_attention
        )
        loss_logits = self.logit_distillation_loss(
            logits, entry.target_logits, entry.target_seen_class_count
        )
        return loss_ce, loss_attention, loss_logits

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("AMIL does not define a separate SSL phase")
        if not batches:
            raise ValueError("AMIL observe_many requires at least one current WSI bag")
        if self._buffer_update_started:
            raise RuntimeError("AMIL cannot train while memory targets need refresh")
        if task is not None and int(task) != self.current_task:
            self._configure_task(int(task), require_previous_snapshot=True)

        self.net.train()
        self.opt.zero_grad(set_to_none=True)

        ce_losses: List[torch.Tensor] = []
        attention_losses: List[torch.Tensor] = []
        logit_losses: List[torch.Tensor] = []
        per_bag_losses: List[torch.Tensor] = []

        for raw_batch in batches:
            features, coords, patch_size, label = self._unpack_batch(raw_batch)
            logits, _ = self._forward_attention(features, coords, patch_size)
            loss_ce = self._classification_loss(logits, label)
            ce_losses.append(loss_ce)
            per_bag_losses.append(loss_ce)

        replay: List[PseudoBag] = []
        if self.old_class_count > 0:
            replay = self.buffer.sample(self.minibatch_size, self.device)
        for entry in replay:
            loss_ce, loss_attention, loss_logits = self._replay_losses(entry)
            ce_losses.append(loss_ce)
            attention_losses.append(loss_attention)
            logit_losses.append(loss_logits)
            per_bag_losses.append(
                loss_ce + self.alpha * loss_attention + self.beta * loss_logits
            )

        loss = torch.stack(per_bag_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("AMIL produced a non-finite loss")
        loss.backward()
        self.opt.step()

        zero = loss.detach().new_zeros(())
        mean_ce = torch.stack(ce_losses).mean()
        # Report each KD component with the same all-bag denominator used by
        # the optimizer objective, rather than a replay-only denominator.
        total_bags = len(per_bag_losses)
        mean_attention = (
            torch.stack(attention_losses).sum() / total_bags
            if attention_losses
            else zero
        )
        mean_logits = (
            torch.stack(logit_losses).sum() / total_bags if logit_losses else zero
        )
        return {
            "loss": float(loss.detach().cpu()),
            "loss_ce": float(mean_ce.detach().cpu()),
            "loss_attn": float(mean_attention.detach().cpu()),
            "loss_logits": float(mean_logits.detach().cpu()),
            "replay_bags": float(len(replay)),
            "buffer_size": float(len(self.buffer)),
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def _ensure_buffer_update(self) -> None:
        if self._buffer_update_started:
            return
        self._assert_previous_snapshot(self.current_task)
        self.buffer.start_update(
            self.current_seen_class_count, task_id=self.current_task
        )
        self._buffer_update_started = True

    def save_buffer(self, features, coords, patch_size, labels, task=None) -> int:
        if task is not None and int(task) != self.current_task:
            self._configure_task(int(task), require_previous_snapshot=True)
        features, coords, patch_size, label = self._unpack_batch(
            (features, coords, patch_size, labels)
        )
        value = int(label.detach().cpu().item())
        if value < self.old_class_count or value >= self.current_seen_class_count:
            raise ValueError(
                "AMIL save_buffer accepts only WSIs from the completed current task"
            )

        self._ensure_buffer_update()
        decision = self.buffer.consider(value)
        if decision is None:
            return -1

        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                _, attention = self._forward_attention(
                    features, coords, patch_size
                )
                indices = maxminrand_select(
                    attention,
                    self.pmp_k,
                    generator=self.buffer.selection_generator,
                ).to(features.device)
                pseudo_bag = PseudoBag(
                    features=features.index_select(0, indices),
                    coords=coords.index_select(0, indices),
                    patch_size=patch_size,
                    label=label,
                    origin_task_id=self.current_task,
                    target_seen_class_count=0,
                    target_attention=None,
                    target_logits=None,
                    target_snapshot_task=None,
                )
            self.buffer.commit(decision, pseudo_bag)
        finally:
            self.net.train(was_training)
        return 1

    def end_task(self, dataset=None) -> None:
        del dataset
        if not self._buffer_update_started:
            raise RuntimeError(
                "AMIL end_task requires save_buffer to start the memory update"
            )

        was_training = self.net.training
        self.net.eval()
        try:
            targets = []
            with torch.no_grad():
                entries = self.buffer.all(
                    device=self.device, require_targets=False
                )
                for entry in entries:
                    logits, attention = self._forward_attention(
                        entry.features, entry.coords, entry.patch_size
                    )
                    targets.append(
                        (
                            attention.detach().cpu().clone(),
                            logits.detach().cpu().clone(),
                        )
                    )
            # Validation and replacement are atomic inside the memory pool.
            self.buffer.refresh_targets(
                targets,
                target_seen_class_count=self.current_seen_class_count,
                target_snapshot_task=self.current_task,
            )
        finally:
            self.net.train(was_training)
        self._buffer_update_started = False

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "buffer_size": self.buffer.capacity,
            "minibatch_size": self.minibatch_size,
            "bags_per_update": self.bags_per_update,
            "pmp_k": self.pmp_k,
            "alpha": self.alpha,
            "beta": self.beta,
            "kd_temperature": self.kd_temperature,
            "backbone_max_patches": 0,
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        if self._buffer_update_started or self.buffer.refresh_required:
            raise RuntimeError(
                "AMIL cannot checkpoint memory before cached-target refresh completes"
            )
        # Trigger the pool's target completeness checks before serialization.
        self.buffer.all(device="cpu", require_targets=True)
        return {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "backbone": str(getattr(self.args, "backbone", "")).lower(),
            "config": self._checkpoint_config(),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "current_task": self.current_task,
            "old_class_count": self.old_class_count,
            "current_seen_class_count": self.current_seen_class_count,
            "buffer": self.buffer.state_dict(),
        }

    def load_checkpoint_state(self, state: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("AMIL checkpoint state must be a dictionary")
        expected = {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "backbone": str(getattr(self.args, "backbone", "")).lower(),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(
                    f"AMIL checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )
        if strict and state.get("config") != self._checkpoint_config():
            raise ValueError(
                "AMIL checkpoint hyperparameters do not match this run: "
                f"saved={state.get('config')!r}, "
                f"expected={self._checkpoint_config()!r}"
            )

        task = int(state.get("current_task", -1))
        old_count, seen_count = self._task_boundaries(task)
        if strict and (
            int(state.get("old_class_count", -1)) != old_count
            or int(state.get("current_seen_class_count", -1)) != seen_count
        ):
            raise ValueError("AMIL checkpoint class boundaries do not match metadata")
        self.buffer.load_state_dict(state.get("buffer"), strict=strict)
        if self.buffer.refresh_required:
            raise ValueError("AMIL checkpoints cannot contain an unfinished refresh")
        self.buffer.all(device="cpu", require_targets=True)

        snapshot = self.buffer.target_snapshot_task
        valid_snapshots = {task, task - 1}
        if task == 0 and len(self.buffer) == 0:
            valid_snapshots.update({None, -1})
        if snapshot not in valid_snapshots:
            raise ValueError(
                "AMIL checkpoint memory snapshot is incompatible with its task state"
            )
        self.current_task = task
        self.old_class_count = old_count
        self.current_seen_class_count = seen_count
        self._buffer_update_started = False


# Preserve the paper-method spelling for direct imports.
AMIL = Amil
