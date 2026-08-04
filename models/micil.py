"""Multiple Instance Class-Incremental Learning for ConSlide.

The immutable MICIL source is preserved under
``third_party/upstream/micil``.  The upstream project exposes its core through
``MICIL_train.py`` rather than a model class; this active adapter keeps its
class-balanced CE, old-class KD, embedding matching and classifier weight
normalization while removing CUDA, six-class and iterator-index assumptions.
"""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from models.utils.wsi_replay import ReplayBag, VariableBagReservoir, unpack_prepared_batch
from utils.args import add_experiment_args, add_management_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Multiple Instance Class-Incremental Learning")
    add_management_args(parser)
    add_experiment_args(parser)
    # BooleanOptionalAction accepts both positive and ``--no-*`` forms and,
    # unlike a mutually-exclusive group, lets a later CLI token override the
    # value injected from YAML.
    parser.add_argument(
        "--micil_replay", action=BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--micil_weight_norm", action=BooleanOptionalAction, default=True
    )
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--minibatch_size", type=int, default=4)
    parser.add_argument("--bags_per_update", type=int, default=1)
    parser.add_argument("--buffer_max_patches", type=int, default=400)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--kd_loss_weight", type=float, default=10.0)
    parser.add_argument("--embedding_loss_weight", type=float, default=1.0)
    parser.add_argument("--distillation_temperature", type=float, default=2.0)
    return parser


class Micil(ContinualModel):
    """MICIL with an optional, explicitly configured replay extension."""

    NAME = "micil"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("titan", "feather")
    REQUIRES_TRAINABLE_BACKBONE = True
    REQUIRED_FEATURE_DIM = 768
    CHECKPOINT_VERSION = 1

    def __init__(self, backbone, loss, args, transform):
        self._validate_configuration(backbone, args)
        super().__init__(backbone, loss, args, transform)

        seed = getattr(args, "seed", 0)
        self.num_classes = int(args.num_classes)
        self.replay_enabled = bool(getattr(args, "micil_replay", False))
        self.minibatch_size = max(1, int(getattr(args, "minibatch_size", 4) or 4))
        self.bags_per_update = max(1, int(getattr(args, "bags_per_update", 1) or 1))
        self.ce_loss_weight = float(getattr(args, "ce_loss_weight", 1.0))
        self.kd_loss_weight = float(getattr(args, "kd_loss_weight", 10.0))
        self.embedding_loss_weight = float(
            getattr(args, "embedding_loss_weight", 1.0)
        )
        self.temperature = float(getattr(args, "distillation_temperature", 2.0))
        self.weight_norm = bool(getattr(args, "micil_weight_norm", True))
        if min(
            self.ce_loss_weight,
            self.kd_loss_weight,
            self.embedding_loss_weight,
        ) < 0:
            raise ValueError("MICIL loss weights must be non-negative")
        if self.temperature <= 0:
            raise ValueError("MICIL distillation temperature must be positive")

        if self.replay_enabled:
            self.buffer = VariableBagReservoir(
                int(getattr(args, "buffer_size", 30)),
                max_patches=int(getattr(args, "buffer_max_patches", 400)),
                seed=0 if seed is None else int(seed),
                feature_dim=768,
            )
        # Avoid registering the frozen teacher as a child module.  It has its
        # own explicit checkpoint payload and must never enter the optimizer.
        object.__setattr__(self, "_teacher", None)

        self.current_task = 0
        self.old_class_count = 0
        task_counts = list(getattr(args, "task_num_classes", []))
        self.seen_class_count = int(task_counts[0]) if task_counts else self.num_classes
        self.class_weights = torch.ones(self.num_classes, device=self.device)

    @classmethod
    def _validate_configuration(cls, backbone, args) -> None:
        name = str(getattr(args, "backbone", "")).lower()
        if name not in cls.SUPPORTED_BACKBONES:
            raise ValueError(
                f"MICIL supports only {cls.SUPPORTED_BACKBONES}, got backbone={name!r}"
            )
        if int(getattr(args, "feature_dim", 768)) != cls.REQUIRED_FEATURE_DIM:
            raise ValueError("MICIL requires 768-D TITAN/FEATHER patch features")
        if bool(getattr(args, "backbone_freeze", False)):
            raise ValueError("MICIL requires a trainable slide backbone")
        if int(getattr(args, "num_classes", 0)) <= 0:
            raise ValueError("MICIL requires a positive global class count")
        if bool(getattr(args, "micil_replay", False)) and int(
            getattr(args, "buffer_size", 30)
        ) < 0:
            raise ValueError("MICIL replay buffer_size must be non-negative")
        if not callable(getattr(backbone, "forward_with_embedding", None)):
            raise TypeError("MICIL backbone must implement forward_with_embedding")
        if bool(getattr(args, "micil_weight_norm", True)) and not callable(
            getattr(backbone, "get_classifier", None)
        ):
            raise TypeError("MICIL weight normalization requires backbone.get_classifier()")

    @property
    def teacher(self) -> Optional[nn.Module]:
        return self.__dict__.get("_teacher")

    def _set_teacher(self, teacher: Optional[nn.Module]) -> None:
        if teacher is not None:
            teacher.to(self.device)
            teacher.eval()
            teacher.requires_grad_(False)
        object.__setattr__(self, "_teacher", teacher)

    def _task_boundaries(self, task: int) -> Tuple[int, int]:
        offsets = [int(value) for value in getattr(self.args, "class_offsets", [])]
        counts = [int(value) for value in getattr(self.args, "task_num_classes", [])]
        if task < 0 or task >= len(offsets) or len(offsets) != len(counts):
            raise ValueError(f"MICIL task {task} is incompatible with class-offset metadata")
        old_count = offsets[task]
        seen_count = old_count + counts[task]
        if old_count < 0 or seen_count <= old_count or seen_count > self.num_classes:
            raise ValueError("MICIL received invalid variable-class task boundaries")
        return old_count, seen_count

    def _configure_task(self, task: int) -> None:
        old_count, seen_count = self._task_boundaries(int(task))
        self.current_task = int(task)
        self.old_class_count = old_count
        self.seen_class_count = seen_count
        if old_count and self.teacher is None:
            raise RuntimeError("MICIL requires the previous best student as a frozen teacher")
        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.requires_grad_(False)

    @staticmethod
    def _dataset_labels(dataset) -> List[int]:
        loader = getattr(dataset, "train_loader", None)
        current = getattr(loader, "dataset", None)
        if current is None:
            return []
        slide_data = getattr(current, "slide_data", None)
        if slide_data is not None and hasattr(slide_data, "columns") and "label" in slide_data:
            return [int(value) for value in slide_data["label"].tolist()]
        targets = getattr(current, "targets", None)
        if targets is not None:
            if torch.is_tensor(targets):
                targets = targets.detach().cpu().reshape(-1).tolist()
            return [int(value) for value in targets]
        return []

    def _refresh_class_weights(self, dataset) -> None:
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for label in self._dataset_labels(dataset):
            if label < 0 or label >= self.seen_class_count:
                raise ValueError(
                    f"MICIL current train label {label} is outside the seen-class range"
                )
            counts[label] += 1
        if self.replay_enabled:
            counts += self.buffer.label_counts(self.num_classes)

        active = counts[: self.seen_class_count] > 0
        weights = torch.zeros(self.num_classes, dtype=torch.float32)
        if active.any():
            active_counts = counts[: self.seen_class_count][active].float()
            weights[: self.seen_class_count][active] = (
                active_counts.sum() / (active_counts.numel() * active_counts)
            )
        else:
            # Unit-level callers may not provide a dataset object; uniform seen
            # weights preserve the MICIL objective in that case.
            weights[: self.seen_class_count] = 1.0
        self.class_weights = weights.to(self.device)

    def begin_task(self, dataset) -> None:
        task = max(0, int(getattr(dataset, "current_task", 1)) - 1)
        self._configure_task(task)
        self._refresh_class_weights(dataset)

    def _forward_embedding(
        self,
        model: nn.Module,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = model.forward_with_embedding(features, coords, patch_size)
        if not isinstance(output, dict):
            raise TypeError("forward_with_embedding must return a dictionary")
        logits = output.get("logits")
        embedding = output.get("embedding")
        if not torch.is_tensor(logits) or not torch.is_tensor(embedding):
            raise ValueError("forward_with_embedding must return tensor logits and embedding")
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if logits.shape != (1, self.num_classes):
            raise ValueError(
                f"MICIL expects logits [1,{self.num_classes}], got {tuple(logits.shape)}"
            )
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(f"MICIL expects one slide embedding, got {tuple(embedding.shape)}")
        if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
            raise FloatingPointError("MICIL backbone returned non-finite values")
        return logits, embedding

    def _forward_batches(
        self,
        batches: Sequence[Sequence[torch.Tensor]],
        *,
        with_teacher: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        student_logits: List[torch.Tensor] = []
        student_embeddings: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        teacher_logits: List[torch.Tensor] = []
        teacher_embeddings: List[torch.Tensor] = []
        for raw_batch in batches:
            features, coords, patch_size, label = unpack_prepared_batch(raw_batch)
            logits, embedding = self._forward_embedding(
                self.net, features, coords, patch_size
            )
            student_logits.append(logits)
            student_embeddings.append(embedding)
            labels.append(label)
            if with_teacher:
                if self.teacher is None:
                    raise RuntimeError("MICIL teacher is not initialized")
                with torch.no_grad():
                    old_logits, old_embedding = self._forward_embedding(
                        self.teacher, features, coords, patch_size
                    )
                teacher_logits.append(old_logits)
                teacher_embeddings.append(old_embedding)
        if not student_logits:
            raise ValueError("MICIL observe_many received no WSI bags")
        return (
            torch.cat(student_logits),
            torch.cat(student_embeddings),
            torch.cat(labels),
            torch.cat(teacher_logits) if teacher_logits else None,
            torch.cat(teacher_embeddings) if teacher_embeddings else None,
        )

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.old_class_count == 0:
            return student_logits.sum() * 0.0
        if teacher_logits is None:
            raise RuntimeError("MICIL KD requires teacher logits")
        old = slice(0, self.old_class_count)
        teacher_probabilities = F.softmax(
            teacher_logits[:, old].float() / self.temperature, dim=1
        )
        student_log_probabilities = F.log_softmax(
            student_logits[:, old].float() / self.temperature, dim=1
        )
        return F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )

    def _embedding_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.old_class_count == 0:
            return student_embeddings.sum() * 0.0
        if teacher_embeddings is None:
            raise RuntimeError("MICIL embedding matching requires teacher embeddings")
        if student_embeddings.shape != teacher_embeddings.shape:
            raise ValueError(
                "MICIL student and teacher embedding shapes differ: "
                f"{tuple(student_embeddings.shape)} vs {tuple(teacher_embeddings.shape)}"
            )
        return F.mse_loss(student_embeddings.float(), teacher_embeddings.float())

    def _classification_loss(
        self,
        seen_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Class-balanced CE without PyTorch's weight-sum cancellation.

        Upstream MICIL uses ``reduction='sum'`` with physical batch size one,
        so even one WSI is scaled by its inverse-frequency class weight.  The
        default weighted-mean reduction would divide by that same weight and
        silently turn the objective back into unweighted CE.  Averaging the
        individually weighted losses preserves upstream behavior for one bag
        while keeping logical current/replay groups independent of group size.
        """

        weights = self.class_weights[: self.seen_class_count].to(seen_logits.device)
        if torch.any(weights.index_select(0, labels) <= 0):
            # Direct callers that did not invoke begin_task still receive a
            # well-defined CE instead of a zero-weight batch.
            weights = torch.ones_like(weights)
        per_bag = F.cross_entropy(
            seen_logits.float(),
            labels.long(),
            weight=weights,
            reduction="none",
        )
        return per_bag.mean()

    @staticmethod
    def _replay_tuple(item: ReplayBag) -> Tuple[torch.Tensor, ...]:
        return item.features, item.coords, item.patch_size, item.label

    def _normalize_classifier(self) -> None:
        if not self.weight_norm or self.current_task == 0:
            return
        classifier = self.net.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError("MICIL get_classifier() must return nn.Linear")
        with torch.no_grad():
            classifier.weight.copy_(F.normalize(classifier.weight, p=2, dim=1))

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("MICIL does not define a separate SSL phase")
        if task is not None and int(task) != self.current_task:
            self._configure_task(int(task))
        elif task is not None:
            self.current_task = int(task)
        with_teacher = self.old_class_count > 0

        self.net.train()
        if self.teacher is not None:
            self.teacher.eval()
        self.opt.zero_grad(set_to_none=True)

        replay: List[ReplayBag] = []
        if self.replay_enabled:
            replay = self.buffer.sample(self.minibatch_size, self.device)
        combined = list(batches) + [self._replay_tuple(item) for item in replay]
        logits, embeddings, labels, teacher_logits, teacher_embeddings = (
            self._forward_batches(combined, with_teacher=with_teacher)
        )
        if labels.min().item() < 0 or labels.max().item() >= self.seen_class_count:
            raise ValueError("MICIL label is outside the current seen-class slice")

        seen_logits = logits[:, : self.seen_class_count]
        loss_ce = self._classification_loss(seen_logits, labels)
        loss_kd = self._distillation_loss(logits, teacher_logits)
        loss_embedding = self._embedding_loss(embeddings, teacher_embeddings)
        loss = (
            self.ce_loss_weight * loss_ce
            + self.kd_loss_weight * loss_kd
            + self.embedding_loss_weight * loss_embedding
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("MICIL produced a non-finite loss")
        loss.backward()
        self.opt.step()
        self._normalize_classifier()

        return {
            "loss": float(loss.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_kd": float(loss_kd.detach().cpu()),
            "loss_embedding": float(loss_embedding.detach().cpu()),
            "replay_bags": float(len(replay)),
            "buffer_size": float(len(self.buffer)) if self.replay_enabled else 0.0,
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def save_buffer(self, features, coords, patch_size, labels, task=None) -> int:
        if not self.replay_enabled:
            return -1
        if task is not None:
            self.current_task = int(task)
        return self.buffer.add(features, coords, patch_size, labels)

    def end_task(self, dataset=None) -> None:
        # Training reloads the best student before this hook.  Rebuilding the
        # teacher here removes the upstream cache's DataLoader-order coupling.
        self._set_teacher(deepcopy(self.net))

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "replay_enabled": self.replay_enabled,
            "buffer_size": self.buffer.capacity if self.replay_enabled else None,
            "minibatch_size": self.minibatch_size,
            "bags_per_update": self.bags_per_update,
            "buffer_max_patches": (
                self.buffer.max_patches if self.replay_enabled else None
            ),
            "ce_loss_weight": self.ce_loss_weight,
            "kd_loss_weight": self.kd_loss_weight,
            "embedding_loss_weight": self.embedding_loss_weight,
            "distillation_temperature": self.temperature,
            "micil_weight_norm": self.weight_norm,
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        teacher_state = None
        if self.teacher is not None:
            teacher_state = {
                key: value.detach().cpu().clone()
                for key, value in self.teacher.state_dict().items()
            }
        state: Dict[str, Any] = {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "backbone": str(getattr(self.args, "backbone", "")).lower(),
            "feature_dim": 768,
            "replay_enabled": self.replay_enabled,
            "config": self._checkpoint_config(),
            "current_task": self.current_task,
            "old_class_count": self.old_class_count,
            "seen_class_count": self.seen_class_count,
            "class_weights": self.class_weights.detach().cpu().clone(),
            "teacher_state": teacher_state,
        }
        if self.replay_enabled:
            state["buffer"] = self.buffer.state_dict()
        return state

    def load_checkpoint_state(self, state: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("MICIL checkpoint state must be a dictionary")
        if bool(state.get("replay_enabled")) != self.replay_enabled:
            raise ValueError(
                "MICIL checkpoint replay mode does not match --micil_replay/"
                "--no-micil_replay"
            )
        expected = {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "backbone": str(getattr(self.args, "backbone", "")).lower(),
            "feature_dim": 768,
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(
                    f"MICIL checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )
        expected_config = self._checkpoint_config()
        if strict and state.get("config") != expected_config:
            raise ValueError(
                "MICIL checkpoint hyperparameters do not match this run: "
                f"saved={state.get('config')!r}, expected={expected_config!r}"
            )
        if self.replay_enabled:
            self.buffer.load_state_dict(state.get("buffer"), strict=True)
        elif "buffer" in state:
            raise ValueError("Non-replay MICIL checkpoint unexpectedly contains a buffer")

        current_task = int(state.get("current_task", 0))
        old_count, seen_count = self._task_boundaries(current_task)
        if strict and (
            int(state.get("old_class_count", -1)) != old_count
            or int(state.get("seen_class_count", -1)) != seen_count
        ):
            raise ValueError("MICIL checkpoint class boundaries do not match task metadata")
        weights = state.get("class_weights")
        if not torch.is_tensor(weights) or weights.shape != (self.num_classes,):
            raise ValueError("MICIL checkpoint has invalid class weights")
        self.current_task = current_task
        self.old_class_count = old_count
        self.seen_class_count = seen_count
        self.class_weights = weights.detach().to(self.device, dtype=torch.float32).clone()

        teacher_state = state.get("teacher_state")
        if teacher_state is None:
            self._set_teacher(None)
        else:
            if not isinstance(teacher_state, dict):
                raise ValueError("MICIL checkpoint teacher_state must be a dictionary")
            teacher = deepcopy(self.net)
            teacher.load_state_dict(teacher_state, strict=strict)
            self._set_teacher(teacher)


# Preserve the paper-method spelling for direct imports.
MICIL = Micil
