"""ATLAS-v3: frozen FEATHER prototype and distribution classifiers.

ATLAS-v3 is the training-free subset of ATLAS-v2.  The slide encoder is
always frozen and each task is learned from train-split slide embeddings at
the task boundary.  The implementation intentionally contains no parameter-
efficient adaptation or exemplar-memory path.
"""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone.generic_mil import build_mil_backbone
from backbone.pretrained_mil import FEATHER_MODEL_ID, FEATHER_REVISION
from models.atlas_distribution import (
    DISTRIBUTION_STATE_VERSION,
    FrozenDistributionHead,
)
from models.utils.continual_model import ContinualModel
from utils.args import add_experiment_args, add_management_args


CHECKPOINT_VERSION = 1
DISTRIBUTION_MODES = (
    "diag",
    "diag_shrink",
    "lowrank",
    "task_centroid",
    "task_lme",
    "multi",
    "pt_only",
    "atlas_tf",
    "atlas_pt",
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ATLAS-v3 frozen prototype learning")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument(
        "--atlasv3_distribution_mode",
        choices=("prototype", "oas_lda", *DISTRIBUTION_MODES),
        default="prototype",
    )
    parser.add_argument("--atlasv3_distribution_alpha", type=float, default=0.1)
    parser.add_argument("--atlasv3_distribution_rho", type=float, default=0.5)
    parser.add_argument("--atlasv3_distribution_rank", type=int, default=8)
    parser.add_argument("--atlasv3_distribution_clusters", type=int, default=3)
    parser.add_argument("--atlasv3_distribution_tau_multi", type=float, default=0.1)
    parser.add_argument("--atlasv3_distribution_beta", type=float, default=0.25)
    parser.add_argument("--atlasv3_distribution_tau_task", type=float, default=0.1)
    parser.add_argument("--atlasv3_pt_steps", type=int, default=200)
    parser.add_argument("--atlasv3_pt_lr", type=float, default=0.05)
    parser.add_argument("--atlasv3_pt_samples_per_class", type=int, default=32)
    parser.add_argument("--atlasv3_pt_margin", type=float, default=0.2)
    return parser


def validate_args(args) -> None:
    if str(getattr(args, "backbone", "")).lower() != "feather":
        raise ValueError("ATLAS-v3 requires the pretrained FEATHER backbone")
    if int(getattr(args, "feature_dim", 768)) != 768:
        raise ValueError("ATLAS-v3 requires 768-D CONCH patch features")
    if bool(getattr(args, "backbone_freeze", False)):
        raise ValueError("ATLAS-v3 owns FEATHER freezing; omit --backbone_freeze")
    if int(getattr(args, "backbone_max_patches", 0) or 0) != 0:
        raise ValueError("ATLAS-v3 requires backbone_max_patches=0 (full bags)")

    mode = str(getattr(args, "atlasv3_distribution_mode", "prototype"))
    if mode not in {"prototype", "oas_lda", *DISTRIBUTION_MODES}:
        raise ValueError(f"Unknown ATLAS-v3 distribution mode {mode!r}")
    if int(getattr(args, "atlasv3_distribution_rank", 0)) not in (2, 4, 8):
        raise ValueError("ATLAS-v3 distribution rank must be one of {2,4,8}")
    if int(getattr(args, "atlasv3_distribution_clusters", 0)) not in (2, 3):
        raise ValueError("ATLAS-v3 distribution clusters must be one of {2,3}")
    if int(getattr(args, "atlasv3_pt_steps", 0)) <= 0:
        raise ValueError("ATLAS-v3 PT steps must be positive")
    for name in (
        "atlasv3_distribution_alpha",
        "atlasv3_distribution_tau_multi",
        "atlasv3_distribution_tau_task",
        "atlasv3_pt_lr",
    ):
        if float(getattr(args, name, 0.0)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= float(getattr(args, "atlasv3_distribution_rho", -1.0)) <= 1.0:
        raise ValueError("atlasv3_distribution_rho must be in [0,1]")
    if float(getattr(args, "atlasv3_distribution_beta", -1.0)) < 0.0:
        raise ValueError("atlasv3_distribution_beta must be non-negative")
    if int(getattr(args, "atlasv3_pt_samples_per_class", 0)) <= 0:
        raise ValueError("atlasv3_pt_samples_per_class must be positive")


class AtlasV3Network(nn.Module):
    """Frozen FEATHER encoder plus one of the supported prototype heads."""

    supports_ssl = False

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        embedding_dim: int,
        mode: str,
        distribution_head: FrozenDistributionHead | None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = backbone.get_classifier()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.mode = str(mode)
        self.distribution_head = distribution_head
        self.distribution_enabled = distribution_head is not None
        self.prototype_lda_enabled = self.mode == "oas_lda"

        if not self.distribution_enabled:
            self.register_buffer(
                "prototype_bank", torch.zeros(num_classes, embedding_dim)
            )
            self.register_buffer(
                "prototype_valid", torch.zeros(num_classes, dtype=torch.bool)
            )
        if self.prototype_lda_enabled:
            self.register_buffer("lda_means", torch.zeros(num_classes, embedding_dim))
            self.register_buffer(
                "lda_counts", torch.zeros(num_classes, dtype=torch.long)
            )
            self.register_buffer(
                "lda_scatter", torch.zeros(embedding_dim, embedding_dim)
            )
            self.register_buffer("lda_degrees", torch.zeros((), dtype=torch.long))
            self.register_buffer("lda_weight", torch.zeros(num_classes, embedding_dim))
            self.register_buffer("lda_bias", torch.zeros(num_classes))
            self.register_buffer("lda_shrinkage", torch.ones(()))
            self.register_buffer("lda_fitted", torch.zeros((), dtype=torch.bool))

    def encode(self, features, coords=None, patch_size_level0=None) -> torch.Tensor:
        output = self.backbone.forward_with_embedding(
            features, coords, patch_size_level0
        )
        embedding = output.get("embedding") if isinstance(output, dict) else None
        if not torch.is_tensor(embedding) or embedding.shape != (1, self.embedding_dim):
            raise ValueError(
                f"ATLAS-v3 expects slide embedding [1,{self.embedding_dim}]"
            )
        return embedding.float()

    def inference_logits(
        self,
        embedding: torch.Tensor,
        seen_classes: int,
        use_prototype: bool = True,
    ) -> torch.Tensor:
        # Before any task has been finalized (for optional FWT evaluation), use
        # FEATHER's frozen classifier because no prototype statistics exist yet.
        if not use_prototype:
            logits = self.classifier(embedding)
        elif self.distribution_enabled:
            logits = self.distribution_head.scores(embedding, seen_classes)
        elif self.prototype_lda_enabled:
            if not bool(self.lda_fitted):
                raise RuntimeError("ATLAS-v3 train-only LDA has not been fitted")
            normalized = F.normalize(embedding, dim=1, eps=1.0e-6)
            logits = F.linear(normalized, self.lda_weight, self.lda_bias)
            logits = logits.masked_fill(
                ~self.prototype_valid.unsqueeze(0), float("-inf")
            )
        else:
            normalized = F.normalize(embedding, dim=1, eps=1.0e-6)
            prototypes = F.normalize(
                self.prototype_bank, dim=1, eps=1.0e-6
            )
            logits = normalized @ prototypes.t()
            logits = logits.masked_fill(
                ~self.prototype_valid.unsqueeze(0), float("-inf")
            )
        if seen_classes < self.num_classes:
            logits = logits.clone()
            logits[:, seen_classes:] = float("-inf")
        return logits

    def forward_with_embedding(
        self,
        features,
        coords=None,
        patch_size_level0=None,
        seen_classes=None,
        use_prototype=True,
    ) -> Dict[str, torch.Tensor | None]:
        embedding = self.encode(features, coords, patch_size_level0)
        limit = self.num_classes if seen_classes is None else int(seen_classes)
        task_scores = None
        if self.distribution_enabled and bool(use_prototype):
            logits, task_scores = self.distribution_head.scores_and_task_scores(
                embedding, limit
            )
        else:
            logits = self.inference_logits(embedding, limit, bool(use_prototype))
        return {
            "embedding": embedding,
            "logits": logits,
            "soft_task_scores": task_scores,
        }

    def forward(self, features, coords=None, patch_size_level0=None, **kwargs):
        if isinstance(features, (list, tuple)):
            values = features
            features = values[0]
            coords = values[1] if len(values) > 1 else coords
            patch_size_level0 = values[2] if len(values) > 2 else patch_size_level0
        output = self.forward_with_embedding(
            features, coords, patch_size_level0, **kwargs
        )
        logits = output["logits"]
        attention = torch.full(
            (1, features.shape[0]),
            1.0 / features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )
        return logits, logits.softmax(1), logits.argmax(1), attention, logits.sum() * 0.0

    @torch.no_grad()
    def set_prototype(self, label: int, embeddings: torch.Tensor) -> None:
        values = F.normalize(
            embeddings.detach().float().reshape(-1, self.embedding_dim), dim=1
        )
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("Cannot create a prototype from empty/non-finite embeddings")
        prototype = F.normalize(values.mean(dim=0), dim=0)
        self.prototype_bank[int(label)].copy_(prototype.to(self.prototype_bank))
        self.prototype_valid[int(label)] = True

    @torch.no_grad()
    def update_lda_statistics(self, label: int, embeddings: torch.Tensor) -> None:
        if not self.prototype_lda_enabled:
            raise RuntimeError("ATLAS-v3 train-only LDA is disabled")
        label = int(label)
        if int(self.lda_counts[label]) != 0:
            raise RuntimeError(f"ATLAS-v3 LDA class {label} was already finalized")
        values = F.normalize(
            embeddings.detach().float().reshape(-1, self.embedding_dim),
            dim=1,
            eps=1.0e-6,
        ).to(self.lda_scatter)
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("Cannot fit LDA from empty/non-finite train embeddings")
        mean = values.mean(0)
        centered = values - mean
        self.lda_means[label].copy_(mean)
        self.lda_counts[label] = int(values.shape[0])
        self.lda_scatter.add_(centered.t() @ centered)
        self.lda_degrees.add_(max(int(values.shape[0]) - 1, 0))

    @torch.no_grad()
    def fit_lda(self, seen_classes: int) -> None:
        if not self.prototype_lda_enabled:
            raise RuntimeError("ATLAS-v3 train-only LDA is disabled")
        seen_classes = int(seen_classes)
        if seen_classes <= 0 or bool((self.lda_counts[:seen_classes] <= 0).any()):
            raise RuntimeError("ATLAS-v3 LDA is missing a seen class's train statistics")

        dimensions = self.embedding_dim
        degrees = int(self.lda_degrees.item())
        empirical = self.lda_scatter.float() / float(max(degrees, 1))
        mu = empirical.trace() / float(dimensions)
        epsilon = torch.finfo(empirical.dtype).eps
        if degrees == 0 or not torch.isfinite(mu) or float(mu) <= epsilon:
            mu = empirical.new_tensor(1.0)
            shrinkage = empirical.new_tensor(1.0)
        else:
            alpha = empirical.square().mean()
            numerator = alpha + mu.square()
            denominator = float(degrees + 1) * (
                alpha - mu.square() / float(dimensions)
            )
            shrinkage = (
                empirical.new_tensor(1.0)
                if not torch.isfinite(denominator) or float(denominator) <= epsilon
                else (numerator / denominator).clamp(0.0, 1.0)
            )
        covariance = (1.0 - shrinkage) * empirical
        covariance.diagonal().add_(shrinkage * mu)
        covariance.diagonal().add_(torch.clamp(mu * 1.0e-6, min=1.0e-6))
        means = self.lda_means[:seen_classes].float()
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if int(info.item()) == 0:
            precision_means = torch.cholesky_solve(means.t(), cholesky).t()
        else:
            precision_means = means @ torch.linalg.pinv(covariance)
        biases = -0.5 * (means * precision_means).sum(1)
        if not torch.isfinite(precision_means).all() or not torch.isfinite(biases).all():
            raise FloatingPointError("ATLAS-v3 LDA produced non-finite parameters")
        self.lda_weight.zero_()
        self.lda_bias.zero_()
        self.lda_weight[:seen_classes].copy_(precision_means.to(self.lda_weight))
        self.lda_bias[:seen_classes].copy_(biases.to(self.lda_bias))
        self.lda_shrinkage.copy_(shrinkage.to(self.lda_shrinkage))
        self.lda_fitted.fill_(True)


class AtlasV3(ContinualModel):
    NAME = "atlas_v3"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("feather",)
    REQUIRED_FEATURE_DIM = 768
    CHECKPOINT_INCLUDE_OPTIMIZER = False
    CHECKPOINT_VERSION = CHECKPOINT_VERSION
    TRAINING_FREE = True

    def __init__(self, backbone, loss, args: Namespace, transform):
        validate_args(args)
        if not callable(getattr(backbone, "forward_with_embedding", None)):
            raise TypeError("ATLAS-v3 FEATHER must expose forward_with_embedding()")
        classifier = backbone.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError("ATLAS-v3 requires a linear FEATHER classifier")
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

        requested_mode = str(args.atlasv3_distribution_mode)
        self.mode = requested_mode
        self.prototype_lda_enabled = self.mode == "oas_lda"
        self.distribution_enabled = self.mode in DISTRIBUTION_MODES

        class_task = []
        for task, count in enumerate(args.task_num_classes):
            class_task.extend([task] * int(count))
        distribution_head = (
            FrozenDistributionHead(
                int(args.num_classes),
                int(classifier.in_features),
                class_task,
                self.mode,
                seed=int(getattr(args, "seed", 0) or 0)
                + 1009 * int(getattr(args, "fold", 0) or 0),
                alpha=float(args.atlasv3_distribution_alpha),
                rho=float(args.atlasv3_distribution_rho),
                rank=int(args.atlasv3_distribution_rank),
                clusters=int(args.atlasv3_distribution_clusters),
                tau_multi=float(args.atlasv3_distribution_tau_multi),
                beta=float(args.atlasv3_distribution_beta),
                tau_task=float(args.atlasv3_distribution_tau_task),
            )
            if self.distribution_enabled
            else None
        )
        network = AtlasV3Network(
            backbone,
            int(args.num_classes),
            int(classifier.in_features),
            self.mode,
            distribution_head,
        )
        super().__init__(network, loss, args, transform)
        self.num_classes = int(args.num_classes)
        self.embedding_dim = int(classifier.in_features)
        if (
            str(getattr(args, "backbone_model_id", "")) == FEATHER_MODEL_ID
            and str(getattr(args, "backbone_revision", "")) == FEATHER_REVISION
            and self.embedding_dim != 512
        ):
            raise ValueError(
                "Pinned FEATHER must expose runtime slide_embedding_dim=512, "
                f"got {self.embedding_dim}"
            )
        self.task_num_classes = tuple(int(value) for value in args.task_num_classes)
        self.class_offsets = tuple(int(value) for value in args.class_offsets)
        self.task_order = tuple(str(value) for value in args.task_order)
        self.n_tasks = int(args.n_tasks)
        self._validate_layout()
        self.current_task = 0
        self.completed_tasks = 0
        self.old_class_count = 0
        self.seen_class_count = self.task_num_classes[0]
        self.calibration_history: List[Dict[str, Any]] = []

    def _validate_layout(self) -> None:
        if not (
            len(self.task_num_classes)
            == len(self.class_offsets)
            == len(self.task_order)
            == self.n_tasks
        ):
            raise ValueError("ATLAS-v3 task metadata lengths differ")
        expected = 0
        for offset, count in zip(self.class_offsets, self.task_num_classes):
            if offset != expected or count <= 0:
                raise ValueError("ATLAS-v3 requires contiguous positive task classes")
            expected += count
        if expected != self.num_classes:
            raise ValueError("ATLAS-v3 task classes do not cover the global classifier")

    def _bounds(self, task: int) -> Tuple[int, int]:
        start = self.class_offsets[int(task)]
        return start, start + self.task_num_classes[int(task)]

    def begin_task(self, dataset) -> None:
        task = int(dataset.current_task) - 1
        if task != self.completed_tasks or not 0 <= task < self.n_tasks:
            raise RuntimeError("ATLAS-v3 tasks must be learned sequentially")
        self.current_task = task
        self.old_class_count, self.seen_class_count = self._bounds(task)

    @torch.no_grad()
    def _encode_loader_object(self, source_loader) -> Dict[int, List[torch.Tensor]]:
        loader = DataLoader(
            source_loader.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=source_loader.collate_fn,
        )
        values: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for batch in loader:
            features, coords, patch_size = self.prepare_inputs(
                batch.features,
                batch.coords,
                batch.patch_size_level0,
                training=False,
            )
            label = int(batch.labels.reshape(-1)[0])
            values[label].append(
                self.net.encode(features, coords, patch_size).cpu()
            )
        return values

    @torch.no_grad()
    def _encode_loader(self, dataset) -> Dict[int, List[torch.Tensor]]:
        return self._encode_loader_object(dataset.train_loader)

    @torch.no_grad()
    def _validation_cache(self, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        loaders = getattr(dataset, "val_loaders", None)
        if not isinstance(loaders, list) or len(loaders) <= self.current_task:
            raise RuntimeError(
                "ATLAS-v3 distribution calibration requires retained validation loaders"
            )
        values, labels = [], []
        for loader in loaders[: self.current_task + 1]:
            encoded = self._encode_loader_object(loader)
            for label in sorted(encoded):
                class_values = torch.cat(encoded[label])
                values.append(class_values)
                labels.append(
                    torch.full((class_values.shape[0],), label, dtype=torch.long)
                )
        if not values:
            raise RuntimeError("ATLAS-v3 distribution calibration cache is empty")
        return torch.cat(values).to(self.device), torch.cat(labels).to(self.device)

    def _calibrate_distribution(self, dataset, current) -> None:
        head = self.net.distribution_head
        validation_raw, labels = self._validation_cache(dataset)
        validation_norm = F.normalize(validation_raw, dim=1, eps=1.0e-8)
        stages = {
            "diag": ("diag",),
            "diag_shrink": ("diag_shrink",),
            "lowrank": ("lowrank",),
            "task_centroid": ("diag_shrink", "task_centroid"),
            "task_lme": ("diag_shrink", "task_lme"),
            "multi": ("multi",),
            "pt_only": ("diag_shrink",),
            "atlas_tf": ("lowrank", "multi", "atlas_tf"),
            "atlas_pt": ("lowrank", "multi", "atlas_tf"),
        }[self.mode]
        selected = []
        for stage in stages:
            if stage == "multi" and self.current_task == 0:
                current_values = {
                    label: torch.cat(current[label]).to(self.device)
                    for label in range(*self._bounds(self.current_task))
                }
                original_mode = head.mode
                best_key, best = None, None
                for index, (clusters, tau) in enumerate(
                    product((2, 3), (0.05, 0.1, 0.2))
                ):
                    head.refit_subprototypes(current_values, clusters)
                    head.selected_clusters.fill_(clusters)
                    head.selected_tau_multi.fill_(tau)
                    head.mode = "multi"
                    predictions = head.scores(validation_norm).argmax(1)
                    recalls = [
                        (predictions[labels == label] == label).float().mean()
                        for label in torch.unique(labels)
                    ]
                    metric = float(torch.stack(recalls).mean())
                    key = (metric, -index)
                    if best_key is None or key > best_key:
                        best_key, best = key, (clusters, tau)
                clusters, tau = best
                head.refit_subprototypes(current_values, clusters)
                head.selected_clusters.fill_(clusters)
                head.selected_tau_multi.fill_(tau)
                head.mode = original_mode
                selected.append(
                    {
                        "stage": "multi",
                        "clusters": clusters,
                        "tau_multi": tau,
                        "validation_bacc": best_key[0],
                        "candidate_count": 6,
                    }
                )
            elif stage == "multi":
                selected.append(
                    {
                        "stage": "multi",
                        "locked_from_task": 0,
                        "clusters": int(head.selected_clusters),
                        "tau_multi": float(head.selected_tau_multi),
                        "candidate_count": 0,
                    }
                )
            else:
                selected.append(
                    {"stage": stage, **head.select(validation_norm, labels, stage)}
                )
        self.calibration_history.append(
            {
                "distribution_state_version": DISTRIBUTION_STATE_VERSION,
                "fold": int(getattr(self.args, "fold", 0) or 0),
                "task": self.current_task,
                "split": "validation",
                "contains_test_cache": False,
                "runtime_slide_embedding_dim": self.embedding_dim,
                "stages": selected,
            }
        )

    def _calibrate_prototype_tuning(self, dataset, current) -> None:
        head = self.net.distribution_head
        validation_raw, labels = self._validation_cache(dataset)
        validation_norm = F.normalize(validation_raw, dim=1, eps=1.0e-8)
        base_offset = head.class_offset.detach().clone()
        best_key, best_offset, best_hp = None, None, None
        candidates = [
            (temperature, anchor, margin)
            for temperature in (0.05, 0.1, 0.2)
            for anchor in (0.01, 0.1, 1.0)
            for margin in (0.0, 0.1)
        ]
        current_tensors = {
            label: torch.cat(current[label]).to(self.device)
            for label in range(*self._bounds(self.current_task))
        }
        for index, (temperature, anchor, margin_weight) in enumerate(candidates):
            with torch.no_grad():
                head.class_offset.copy_(base_offset)
            head.tune_offsets(
                current_tensors,
                task=self.current_task,
                steps=int(self.args.atlasv3_pt_steps),
                learning_rate=float(self.args.atlasv3_pt_lr),
                samples_per_class=int(self.args.atlasv3_pt_samples_per_class),
                temperature=temperature,
                anchor_weight=anchor,
                margin_weight=margin_weight,
                margin=float(self.args.atlasv3_pt_margin),
            )
            with torch.no_grad():
                metric = float(
                    sum(
                        (
                            head.scores(validation_norm).argmax(1)[labels == label]
                            == label
                        )
                        .float()
                        .mean()
                        for label in torch.unique(labels)
                    )
                    / len(torch.unique(labels))
                )
            key = (metric, -index)
            if best_key is None or key > best_key:
                best_key = key
                best_offset = head.class_offset.detach().clone()
                best_hp = {
                    "ce_temperature": temperature,
                    "anchor_weight": anchor,
                    "margin_weight": margin_weight,
                }
        with torch.no_grad():
            head.class_offset.copy_(best_offset)
        self.calibration_history[-1]["stages"].append(
            {
                "stage": "prototype_tuning",
                **best_hp,
                "validation_bacc": float(best_key[0]),
                "candidate_count": len(candidates),
            }
        )

    def end_task(self, dataset=None) -> None:
        if dataset is None:
            raise RuntimeError("ATLAS-v3 end_task requires the current train dataset")
        was_training = self.net.training
        self.net.eval()
        try:
            current = self._encode_loader(dataset)
            start, stop = self._bounds(self.current_task)
            for label in range(start, stop):
                if not current[label]:
                    raise RuntimeError(f"Current train split has no class {label}")
                embeddings = torch.cat(current[label])
                if self.distribution_enabled:
                    self.net.distribution_head.fit_class(
                        label,
                        F.normalize(
                            embeddings.to(self.device), dim=1, eps=1.0e-8
                        ),
                    )
                else:
                    self.net.set_prototype(label, embeddings)
                    if self.prototype_lda_enabled:
                        self.net.update_lda_statistics(label, embeddings)
            if self.distribution_enabled:
                self._calibrate_distribution(dataset, current)
                if self.mode in {"pt_only", "atlas_pt"}:
                    self._calibrate_prototype_tuning(dataset, current)
            elif self.prototype_lda_enabled:
                self.net.fit_lda(stop)
        finally:
            self.net.train(was_training)
        self.completed_tasks = self.current_task + 1
        print(f"[atlas_v3] finalized task {self.current_task}: retained_wsis=0")

    def forward(self, x, coords=None, patch_size_level0=None):
        use_prototype = self.completed_tasks > self.current_task
        with self.autocast_context():
            if isinstance(x, (list, tuple)):
                values = x
                return self.net(
                    values[0],
                    values[1],
                    values[2],
                    seen_classes=self.seen_class_count,
                    use_prototype=use_prototype,
                )
            return self.net(
                x,
                coords,
                patch_size_level0,
                seen_classes=self.seen_class_count,
                use_prototype=use_prototype,
            )

    def forward_with_distribution_diagnostics(
        self, x, coords=None, patch_size_level0=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        use_prototype = self.completed_tasks > self.current_task
        if isinstance(x, (list, tuple)):
            values = x
            x, coords, patch_size_level0 = values[0], values[1], values[2]
        with self.autocast_context():
            output = self.net.forward_with_embedding(
                x,
                coords,
                patch_size_level0,
                seen_classes=self.seen_class_count,
                use_prototype=use_prototype,
            )
        return output["logits"], output["soft_task_scores"]

    def _config(self) -> Dict[str, Any]:
        config = {
            "version": CHECKPOINT_VERSION,
            "backbone": "feather",
            "training_free": True,
            "classifier": self.mode,
            "ablation_id": getattr(self.args, "ablation_id", None),
            "ablation_config_hash": getattr(
                self.args, "ablation_config_hash", None
            ),
        }
        if self.prototype_lda_enabled:
            config.update(
                {
                    "prototype_metric": "oas_shrinkage_lda",
                    "prototype_fit_data": "train_only",
                    "test_time_adaptation": False,
                }
            )
        if self.distribution_enabled:
            head = self.net.distribution_head
            config.update(
                {
                    "distribution_state_version": DISTRIBUTION_STATE_VERSION,
                    "distribution_fit_data": "train_only_normalized_slide_embeddings",
                    "runtime_slide_embedding_dim": self.embedding_dim,
                    "expected_pinned_feather_embedding_dim": 512,
                    "test_time_adaptation": False,
                    "alpha": float(head.selected_alpha),
                    "rho": float(head.selected_rho),
                    "rank": int(head.selected_rank),
                    "clusters": int(head.selected_clusters),
                    "tau_multi": float(head.selected_tau_multi),
                    "beta": float(head.selected_beta),
                    "tau_task": float(head.selected_tau_task),
                }
            )
        return config

    def get_run_metadata(self) -> Dict[str, Any]:
        config = self._config()
        return {
            "atlas_v3_config": config,
            "atlas_v3_config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "prototype_metric_accounting": (
                {
                    "metric": "oas_shrinkage_lda",
                    "train_samples": int(self.net.lda_counts.sum().item()),
                    "within_class_degrees": int(self.net.lda_degrees.item()),
                    "oas_shrinkage": float(self.net.lda_shrinkage.item()),
                    "test_time_adaptation": False,
                }
                if self.prototype_lda_enabled
                else None
            ),
            "distribution_accounting": (
                {
                    **self.net.distribution_head.diagnostics(),
                    "mode": self.mode,
                    "stored_slide_embeddings": 0,
                    "test_time_adaptation": False,
                }
                if self.distribution_enabled
                else None
            ),
            "calibration_manifest": (
                list(self.calibration_history) if self.distribution_enabled else None
            ),
        }

    def get_calibration_manifest(self) -> Dict[str, Any] | None:
        if not self.distribution_enabled:
            return None
        return {
            "distribution_state_version": DISTRIBUTION_STATE_VERSION,
            "ablation_id": getattr(self.args, "ablation_id", None),
            "fold": int(getattr(self.args, "fold", 0) or 0),
            "runtime_slide_embedding_dim": self.embedding_dim,
            "expected_pinned_feather_embedding_dim": 512,
            "backbone_model_id": getattr(self.args, "backbone_model_id", None),
            "backbone_revision": getattr(self.args, "backbone_revision", None),
            "selection_protocol": "fold-specific isolated validation; staged non-Cartesian DAG",
            "contains_train_embeddings": False,
            "contains_validation_embeddings": False,
            "contains_test_embeddings": False,
            "history": list(self.calibration_history),
        }

    def get_task_diagnostics(self) -> Dict[str, Any] | None:
        if not self.distribution_enabled:
            return None
        return {
            "task": self.current_task,
            **self.net.distribution_head.diagnostics(),
            "selected_hyperparameters": json.dumps(
                self.calibration_history[-1]["stages"]
                if self.calibration_history
                else [],
                sort_keys=True,
            ),
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        state = {
            "version": CHECKPOINT_VERSION,
            "method": self.NAME,
            "config": self._config(),
            "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "current_task": self.current_task,
            "completed_tasks": self.completed_tasks,
            "old_class_count": self.old_class_count,
            "seen_class_count": self.seen_class_count,
        }
        if self.distribution_enabled:
            state["calibration_history"] = list(self.calibration_history)
        return state

    def load_checkpoint_state(
        self, state: Mapping[str, Any], strict: bool = True
    ) -> None:
        expected = {
            "version": CHECKPOINT_VERSION,
            "method": self.NAME,
            "config": self._config(),
            "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(f"ATLAS-v3 checkpoint mismatch for {key}")
        completed = int(state.get("completed_tasks", -1))
        task = int(state.get("current_task", -1))
        if not 0 <= task < self.n_tasks or completed not in {task, task + 1}:
            raise ValueError("ATLAS-v3 checkpoint task state is invalid")
        old_count, seen_count = self._bounds(task)
        if strict and (
            int(state.get("old_class_count", -1)) != old_count
            or int(state.get("seen_class_count", -1)) != seen_count
        ):
            raise ValueError("ATLAS-v3 checkpoint class boundaries are invalid")
        self.current_task = task
        self.completed_tasks = completed
        self.old_class_count = int(state["old_class_count"])
        self.seen_class_count = int(state["seen_class_count"])
        if self.distribution_enabled:
            calibration = [
                dict(row) for row in state.get("calibration_history", [])
            ]
            if strict and len(calibration) != completed:
                raise ValueError(
                    "ATLAS-v3 distribution checkpoint calibration history is incomplete"
                )
            self.calibration_history = calibration


def build_model_from_components(args, loss, transform, backbone) -> AtlasV3:
    return AtlasV3(backbone, loss, args, transform)


def build_model(args: Namespace, loss, transform) -> AtlasV3:
    validate_args(args)
    backbone = build_mil_backbone(args, int(args.num_classes))
    return build_model_from_components(args, loss, transform, backbone)


ATLASV3 = AtlasV3
