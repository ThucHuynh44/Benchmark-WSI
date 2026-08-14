"""Dynamic OWLoRA adaptation for the native TITAN and FEATHER backbones.

The immutable CoMEL sources used for algorithm provenance live under
``third_party/upstream/comel_owlora``.  This active implementation keeps the
benchmark's variable-length WSI and global-classifier contracts and does not
import the upstream training stack.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from models.utils.owlora import (
    OWLoRALinear,
    attach_owlora,
    expand_owlora,
    initialize_references,
    orthogonality_penalty,
    project_current_gradients,
    reconstruct_adapter_layout_from_state_dict,
)
from utils.args import add_experiment_args, add_management_args
from utils.optim import build_optimizer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Dynamic OWLoRA for TITAN and FEATHER")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--owlora_rank", type=int, default=8)
    parser.add_argument("--owlora_svd_energy", type=float, default=0.99)
    parser.add_argument("--owlora_orthogonal_weight", type=float, default=1.0)
    parser.add_argument("--bags_per_update", type=int, default=1)
    return parser


def validate_args(args) -> None:
    rank = int(getattr(args, "owlora_rank", 8))
    energy = float(getattr(args, "owlora_svd_energy", 0.99))
    weight = float(getattr(args, "owlora_orthogonal_weight", 1.0))
    bags = int(getattr(args, "bags_per_update", 1))
    if rank <= 0:
        raise ValueError("owlora_rank must be positive")
    if not 0.0 < energy < 1.0:
        raise ValueError("owlora_svd_energy must be in (0, 1)")
    if weight < 0.0:
        raise ValueError("owlora_orthogonal_weight must be non-negative")
    if bags <= 0:
        raise ValueError("bags_per_update must be positive")


class Owlora(ContinualModel):
    """CoMEL OWLoRA with dynamic adapters and seen-class cross entropy."""

    NAME = "owlora"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("titan", "feather")
    REQUIRED_FEATURE_DIM = 768
    REQUIRES_TRAINABLE_BACKBONE = True
    CHECKPOINT_INCLUDE_OPTIMIZER = False
    CHECKPOINT_VERSION = 1

    def __init__(self, backbone, loss, args, transform):
        validate_args(args)
        self._validate_backbone(backbone, args)

        classifier = backbone.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError(
                "OWLoRA requires the TITAN/FEATHER classifier to be nn.Linear"
            )
        num_classes = int(getattr(args, "num_classes", 0))
        if int(classifier.out_features) != num_classes:
            raise ValueError(
                "OWLoRA classifier output mismatch: "
                f"expected {num_classes}, got {classifier.out_features}"
            )

        backbone_name = str(args.backbone).lower()
        if backbone_name == "titan":
            root = getattr(backbone, "vision_encoder", None)
            root_name = "vision_encoder"
        else:
            root = getattr(backbone, "model", None)
            root_name = "model"
        if not isinstance(root, nn.Module):
            raise TypeError(
                f"OWLoRA could not resolve the {backbone_name} encoder root"
            )

        rank = int(getattr(args, "owlora_rank", 8))
        wrapped = attach_owlora(
            root,
            root_name=root_name,
            classifier=classifier,
            rank=rank,
        )
        super().__init__(backbone, loss, args, transform)

        # These are aliases into self.net.  Keep them out of nn.Module's child
        # registry so state_dict contains one canonical model hierarchy.
        object.__setattr__(self, "_classifier", classifier)
        object.__setattr__(self, "_owlora_modules", wrapped)

        self.rank = rank
        self.svd_energy = float(getattr(args, "owlora_svd_energy", 0.99))
        self.orthogonal_weight = float(
            getattr(args, "owlora_orthogonal_weight", 1.0)
        )
        self.task_num_classes = tuple(
            int(value) for value in getattr(args, "task_num_classes", ())
        )
        self.class_offsets = tuple(
            int(value) for value in getattr(args, "class_offsets", ())
        )
        self.task_order = tuple(str(value) for value in getattr(args, "task_order", ()))
        self.n_tasks = int(getattr(args, "n_tasks", len(self.task_num_classes)))
        self.num_classes = num_classes
        self._validate_task_layout()

        self.current_task = 0
        self.completed_tasks = 0
        self.reference_initialized = False
        self._configure_task(0)
        print(
            "[owlora] adapted "
            f"{len(self._owlora_modules)} Linear modules for {backbone_name}: "
            + ", ".join(self._owlora_modules)
        )

    @classmethod
    def _validate_backbone(cls, backbone, args) -> None:
        name = str(getattr(args, "backbone", "")).lower()
        if name not in cls.SUPPORTED_BACKBONES:
            raise ValueError(
                f"OWLoRA supports only {cls.SUPPORTED_BACKBONES}, got {name!r}"
            )
        if int(getattr(args, "feature_dim", 768)) != cls.REQUIRED_FEATURE_DIM:
            raise ValueError("OWLoRA requires 768-D TITAN/FEATHER patch features")
        if bool(getattr(args, "backbone_freeze", False)):
            raise ValueError("OWLoRA requires a trainable slide backbone")
        if not callable(getattr(backbone, "get_classifier", None)):
            raise TypeError("OWLoRA backbone must expose get_classifier()")

    def _validate_task_layout(self) -> None:
        if self.n_tasks <= 0:
            raise ValueError("OWLoRA requires at least one continual task")
        if len(self.task_num_classes) != self.n_tasks:
            raise ValueError("OWLoRA task_num_classes length must equal n_tasks")
        if len(self.class_offsets) != self.n_tasks:
            raise ValueError("OWLoRA class_offsets length must equal n_tasks")
        if any(count <= 0 for count in self.task_num_classes):
            raise ValueError("OWLoRA task_num_classes must be positive")
        expected_offset = 0
        for offset, count in zip(self.class_offsets, self.task_num_classes):
            if offset != expected_offset:
                raise ValueError(
                    "OWLoRA requires contiguous global class offsets, got "
                    f"{self.class_offsets}"
                )
            expected_offset += count
        if expected_offset != self.num_classes:
            raise ValueError(
                "OWLoRA task class counts do not sum to the global classifier size"
            )
        if self.task_order and len(self.task_order) != self.n_tasks:
            raise ValueError("OWLoRA task_order length must equal n_tasks")

    @property
    def classifier(self) -> nn.Linear:
        return self.__dict__["_classifier"]

    @property
    def owlora_modules(self) -> Mapping[str, OWLoRALinear]:
        return self.__dict__["_owlora_modules"]

    def _adapter_counts(self) -> Dict[str, int]:
        return {
            path: len(module.lora_layers)
            for path, module in self.owlora_modules.items()
        }

    def _task_adapter_count(self) -> int:
        counts = {module.task_adapter_count for module in self.owlora_modules.values()}
        if len(counts) != 1:
            raise RuntimeError(f"Inconsistent OWLoRA task adapter counts: {counts}")
        return counts.pop()

    def _configure_task(self, task: int) -> None:
        task = int(task)
        if task < 0 or task >= self.n_tasks:
            raise IndexError(f"OWLoRA task must be in [0,{self.n_tasks - 1}], got {task}")
        expected_adapters = task
        actual_adapters = self._task_adapter_count()
        if actual_adapters != expected_adapters:
            raise RuntimeError(
                f"OWLoRA task {task} requires {expected_adapters} task adapters, "
                f"found {actual_adapters}"
            )
        if task > 0 and not all(
            module.has_reference for module in self.owlora_modules.values()
        ):
            raise RuntimeError("OWLoRA task adapters are missing their references")

        if task == 0:
            for parameter in self.net.parameters():
                parameter.requires_grad_(True)
        else:
            for parameter in self.net.parameters():
                parameter.requires_grad_(False)
            for module in self.owlora_modules.values():
                module.lora_layers[-1].requires_grad_(True)
            self.classifier.requires_grad_(True)

        trainable = [
            parameter for parameter in self.net.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("OWLoRA task has no trainable parameters")
        self.opt = build_optimizer(trainable, self.args)
        self.current_task = task

    def begin_task(self, dataset) -> None:
        task = max(0, int(getattr(dataset, "current_task", 1)) - 1)
        if self.completed_tasks != task:
            raise RuntimeError(
                "OWLoRA task sequence mismatch: "
                f"completed={self.completed_tasks}, next={task}"
            )
        self._configure_task(task)

    def _task_bounds(self, task: int) -> tuple[int, int]:
        start = self.class_offsets[task]
        return start, start + self.task_num_classes[task]

    def _forward_batches(self, batches: Sequence[Sequence[torch.Tensor]]):
        logits, labels = [], []
        for features, coords, patch_size, label in batches:
            output = self.net([features, coords, patch_size])
            batch_logits = output[0] if isinstance(output, (tuple, list)) else output
            if batch_logits.ndim == 1:
                batch_logits = batch_logits.unsqueeze(0)
            if batch_logits.ndim != 2 or batch_logits.shape[1] != self.num_classes:
                raise ValueError(
                    "OWLoRA expects global logits "
                    f"[B,{self.num_classes}], got {tuple(batch_logits.shape)}"
                )
            logits.append(batch_logits)
            labels.append(label.long().reshape(-1))
        if not logits:
            raise ValueError("OWLoRA observe_many requires at least one WSI bag")
        return torch.cat(logits, dim=0), torch.cat(labels, dim=0)

    def _protect_classifier_rows(self, start: int, stop: int):
        weight_before = self.classifier.weight.detach().clone()
        bias_before = (
            self.classifier.bias.detach().clone()
            if self.classifier.bias is not None
            else None
        )
        weight_grad = self.classifier.weight.grad
        if weight_grad is not None:
            weight_grad[:start].zero_()
            weight_grad[stop:].zero_()
        if self.classifier.bias is not None and self.classifier.bias.grad is not None:
            self.classifier.bias.grad[:start].zero_()
            self.classifier.bias.grad[stop:].zero_()
        return weight_before, bias_before

    def _restore_classifier_rows(
        self,
        start: int,
        stop: int,
        weight_before: torch.Tensor,
        bias_before: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            self.classifier.weight[:start].copy_(weight_before[:start])
            self.classifier.weight[stop:].copy_(weight_before[stop:])
            if self.classifier.bias is not None and bias_before is not None:
                self.classifier.bias[:start].copy_(bias_before[:start])
                self.classifier.bias[stop:].copy_(bias_before[stop:])

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("OWLoRA does not define a separate SSL phase")
        if task is not None and int(task) != self.current_task:
            raise RuntimeError(
                f"OWLoRA received task={task}, active task={self.current_task}"
            )
        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        logits, labels = self._forward_batches(batches)
        start, seen_stop = self._task_bounds(self.current_task)
        if labels.numel() != logits.shape[0]:
            raise ValueError("OWLoRA logits and labels have different batch sizes")
        if labels.min().item() < start or labels.max().item() >= seen_stop:
            raise ValueError(
                f"OWLoRA task {self.current_task} labels must be in "
                f"[{start},{seen_stop}), got [{labels.min().item()},{labels.max().item()}]"
            )

        loss_ce = F.cross_entropy(logits[:, :seen_stop].float(), labels)
        if self.current_task > 0:
            loss_orthogonal = orthogonality_penalty(self.owlora_modules)
        else:
            loss_orthogonal = loss_ce.detach() * 0.0
        loss = loss_ce + self.orthogonal_weight * loss_orthogonal
        if not torch.isfinite(loss):
            raise FloatingPointError("OWLoRA produced a non-finite loss")
        loss.backward()

        if self.current_task > 0:
            project_current_gradients(self.owlora_modules)
        classifier_snapshot = self._protect_classifier_rows(start, seen_stop)
        self.opt.step()
        self._restore_classifier_rows(
            start, seen_stop, classifier_snapshot[0], classifier_snapshot[1]
        )
        return {
            "loss": float(loss.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_orthogonal": float(loss_orthogonal.detach().cpu()),
            "adapted_linear_count": float(len(self.owlora_modules)),
            "active_adapter_count": float(self._task_adapter_count()),
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def end_task(self, dataset=None) -> None:
        task = self.current_task
        if self.completed_tasks >= task + 1:
            raise RuntimeError(f"OWLoRA task {task} has already been finalized")
        if task == 0:
            initialize_references(self.owlora_modules, self.svd_energy)
            self.reference_initialized = True
        if task < self.n_tasks - 1:
            expand_owlora(self.owlora_modules)
        self.completed_tasks = task + 1
        print(
            f"[owlora] finalized task {task}: "
            f"task_adapters={self._task_adapter_count()}, "
            f"total_parameters={sum(p.numel() for p in self.net.parameters()):,}"
        )

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "backbone": str(self.args.backbone).lower(),
            "rank": self.rank,
            "svd_energy": self.svd_energy,
            "orthogonal_weight": self.orthogonal_weight,
            "num_classes": self.num_classes,
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "task_order": list(self.task_order),
            "module_paths": list(self.owlora_modules),
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "config": self._checkpoint_config(),
            "current_task": self.current_task,
            "completed_tasks": self.completed_tasks,
            "reference_initialized": self.reference_initialized,
            "adapter_counts": self._adapter_counts(),
            "active_adapter_count": self._task_adapter_count(),
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        reconstruct_adapter_layout_from_state_dict(
            self.owlora_modules, state_dict, model_prefix="net."
        )
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            if assign:
                raise
            return super().load_state_dict(state_dict, strict=strict)

    def load_checkpoint_state(self, state, strict=True) -> None:
        if not isinstance(state, dict):
            raise ValueError("OWLoRA checkpoint method_state must be a dictionary")
        expected = self._checkpoint_config()
        if int(state.get("version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError("OWLoRA checkpoint version mismatch")
        if state.get("method") != self.NAME:
            raise ValueError("OWLoRA checkpoint method mismatch")
        if strict and state.get("config") != expected:
            raise ValueError(
                "OWLoRA checkpoint configuration mismatch: "
                f"saved={state.get('config')!r}, expected={expected!r}"
            )
        actual_counts = self._adapter_counts()
        saved_counts = {
            str(path): int(count)
            for path, count in state.get("adapter_counts", {}).items()
        }
        if saved_counts != actual_counts:
            raise ValueError(
                "OWLoRA reconstructed adapter layout mismatch: "
                f"saved={saved_counts}, actual={actual_counts}"
            )
        self.current_task = int(state.get("current_task", 0))
        self.completed_tasks = int(state.get("completed_tasks", 0))
        self.reference_initialized = bool(
            state.get("reference_initialized", False)
        )
        active = int(state.get("active_adapter_count", -1))
        if active != self._task_adapter_count():
            raise ValueError(
                f"OWLoRA active adapter mismatch: saved={active}, "
                f"actual={self._task_adapter_count()}"
            )
        if self.reference_initialized != all(
            module.has_reference for module in self.owlora_modules.values()
        ):
            raise ValueError("OWLoRA reference metadata does not match model state")
