"""Replay-free ACL adaptation with historical feature transport for FEATHER."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone.generic_mil import build_mil_backbone
from backbone.pretrained_mil import FEATHER_MODEL_ID, FEATHER_REVISION
from models.utils.atlas_transport import (
    bootstrap_gates,
    distribution_coverage,
    fit_ldc,
    fit_lowrank_residual,
    fit_sldc,
    mean_coverage,
    summarize,
)
from models.utils.continual_model import ContinualModel
from utils.args import add_experiment_args, add_management_args
from utils.optim import build_optimizer


CHECKPOINT_VERSION = 1
ACL_MODES = (
    "acl",
    "histneg",
    "sdc",
    "ldc",
    "sldc",
    "lowrank",
    "histneg_lowrank",
    "gated",
    "gated_oas",
    "gated_task_margin",
    "frozen_raw_oas",
    "oas_static",
    "oas_transport",
    "oas_oracle",
    "normalized_oas_static",
    "transport_normalized_oas",
    "gated_transport_normalized_oas_no_histneg",
    "histneg_normalized_oas_static",
    "histneg_transport_normalized_oas",
    "gated_normalized_oas",
)
TRANSPORT_MODES = {
    "sdc", "ldc", "sldc", "lowrank", "histneg_lowrank",
    "gated", "gated_oas", "gated_task_margin", "oas_transport",
    "transport_normalized_oas", "gated_transport_normalized_oas_no_histneg",
    "histneg_transport_normalized_oas", "gated_normalized_oas",
}
HISTNEG_MODES = {
    "histneg", "histneg_lowrank", "gated", "gated_oas",
    "gated_task_margin", "oas_static", "oas_transport", "oas_oracle",
    "histneg_normalized_oas_static", "histneg_transport_normalized_oas",
    "gated_normalized_oas",
}
GATED_MODES = {
    "gated", "gated_oas", "gated_task_margin",
    "gated_transport_normalized_oas_no_histneg", "gated_normalized_oas",
}
OAS_MODES = {
    "gated_oas", "frozen_raw_oas", "oas_static", "oas_transport", "oas_oracle",
    "normalized_oas_static", "histneg_normalized_oas_static",
    "transport_normalized_oas", "gated_transport_normalized_oas_no_histneg",
    "histneg_transport_normalized_oas", "gated_normalized_oas",
}
RAW_TRANSPORT_MODES = {"gated_oas", "oas_transport"}
NORMALIZED_OAS_MODES = {
    "normalized_oas_static", "histneg_normalized_oas_static",
    "transport_normalized_oas", "gated_transport_normalized_oas_no_histneg",
    "histneg_transport_normalized_oas", "gated_normalized_oas",
}
OAS_TRANSPORT_MODES = RAW_TRANSPORT_MODES | {
    "transport_normalized_oas", "gated_transport_normalized_oas_no_histneg",
    "histneg_transport_normalized_oas", "gated_normalized_oas",
}
OAS_DIAGNOSTIC_SEMANTICS = {
    "oas_static": "acl_histneg_raw_oas_static_no_transport_no_gate_v1",
    "oas_transport": "acl_histneg_raw_oas_ungated_lowrank_transport_v1",
    "oas_oracle": "acl_histneg_raw_oas_oracle_recompute_with_drift_probe_v1",
    "normalized_oas_static": "acl_only_normalized_oas_static_no_transport_v1",
    "transport_normalized_oas": "acl_only_normalized_oas_ungated_lowrank_transport_v1",
    "gated_transport_normalized_oas_no_histneg": "acl_only_normalized_oas_gated_lowrank_transport_v1",
    "histneg_normalized_oas_static": "acl_histneg_normalized_oas_static_no_transport_v1",
    "histneg_transport_normalized_oas": "acl_histneg_normalized_oas_ungated_lowrank_transport_v1",
    "gated_normalized_oas": "acl_histneg_normalized_oas_gated_lowrank_transport_v1",
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ATLAS-v3 ACL historical transport")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument(
        "--bags_per_update",
        type=int,
        default=1,
        help="Current-task WSI bags accumulated into one ACL optimizer update.",
    )
    parser.add_argument("--atlasv3_acl_mode", choices=ACL_MODES, default="acl")
    parser.add_argument("--atlasv3_acl_temperature", type=float, default=0.1)
    parser.add_argument("--atlasv3_acl_hist_weight", type=float, default=1.0)
    parser.add_argument("--atlasv3_acl_hist_temperature", type=float, default=0.1)
    parser.add_argument("--atlasv3_acl_hist_margin", type=float, default=0.2)
    parser.add_argument("--atlasv3_acl_hist_topk", type=int, default=8)
    parser.add_argument("--atlasv3_acl_transport_rank", type=int, default=8)
    parser.add_argument("--atlasv3_acl_transport_ridge", type=float, default=1.0e-3)
    parser.add_argument(
        "--atlasv3_acl_transport_mean_scale",
        type=float,
        default=1.0,
        help="Multiplier in [0,1] applied to the estimated historical-mean transport step.",
    )
    parser.add_argument(
        "--atlasv3_acl_transport_cov_scale",
        type=float,
        default=1.0,
        help="Multiplier in [0,1] applied to the estimated historical-covariance transport step.",
    )
    parser.add_argument("--atlasv3_acl_ldc_steps", type=int, default=100)
    parser.add_argument("--atlasv3_acl_ldc_lr", type=float, default=1.0e-3)
    parser.add_argument("--atlasv3_acl_sdc_sigma", type=float, default=0.3)
    parser.add_argument("--atlasv3_acl_coverage_energy", type=float, default=0.95)
    parser.add_argument("--atlasv3_acl_bootstrap_samples", type=int, default=20)
    parser.add_argument("--atlasv3_acl_uncertainty_beta", type=float, default=10.0)
    parser.add_argument("--atlasv3_acl_reliability_floor", type=float, default=0.1)
    parser.add_argument("--atlasv3_acl_reliability_momentum", type=float, default=0.5)
    parser.add_argument("--atlasv3_acl_task_margin", type=float, default=0.2)
    parser.add_argument("--atlasv3_acl_task_weight", type=float, default=0.1)
    return parser


def validate_args(args) -> None:
    if str(getattr(args, "backbone", "")).lower() != "feather":
        raise ValueError("ATLAS-v3 ACL requires FEATHER")
    if int(getattr(args, "feature_dim", 768)) != 768:
        raise ValueError("ATLAS-v3 ACL requires 768-D CONCH patch features")
    if bool(getattr(args, "backbone_freeze", False)):
        raise ValueError("ATLAS-v3 ACL owns the task-boundary FEATHER freeze lifecycle")
    if int(getattr(args, "backbone_max_patches", 0) or 0) != 0:
        raise ValueError("ATLAS-v3 ACL requires full WSI bags")
    if int(getattr(args, "bags_per_update", 0)) <= 0:
        raise ValueError("bags_per_update must be positive")
    if str(getattr(args, "atlasv3_acl_mode", "")) not in ACL_MODES:
        raise ValueError("Unknown ATLAS-v3 ACL mode")
    positive = (
        "atlasv3_acl_temperature", "atlasv3_acl_hist_temperature",
        "atlasv3_acl_transport_ridge", "atlasv3_acl_ldc_lr",
        "atlasv3_acl_sdc_sigma",
    )
    for name in positive:
        if float(getattr(args, name, 0.0)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(getattr(args, "atlasv3_acl_hist_topk", 0)) <= 0:
        raise ValueError("atlasv3_acl_hist_topk must be positive")
    if int(getattr(args, "atlasv3_acl_transport_rank", 0)) <= 0:
        raise ValueError("atlasv3_acl_transport_rank must be positive")
    for name in (
        "atlasv3_acl_transport_mean_scale",
        "atlasv3_acl_transport_cov_scale",
    ):
        value = float(getattr(args, name, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if int(getattr(args, "atlasv3_acl_ldc_steps", 0)) <= 0:
        raise ValueError("atlasv3_acl_ldc_steps must be positive")
    if int(getattr(args, "atlasv3_acl_bootstrap_samples", -1)) < 0:
        raise ValueError("atlasv3_acl_bootstrap_samples must be non-negative")
    for name in ("atlasv3_acl_coverage_energy", "atlasv3_acl_reliability_floor", "atlasv3_acl_reliability_momentum"):
        value = float(getattr(args, name, -1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0,1]")
    for name in ("atlasv3_acl_hist_weight", "atlasv3_acl_hist_margin", "atlasv3_acl_uncertainty_beta", "atlasv3_acl_task_margin", "atlasv3_acl_task_weight"):
        if float(getattr(args, name, -1.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")


class AtlasV3ACLNetwork(nn.Module):
    supports_ssl = False

    def __init__(self, backbone: nn.Module, num_classes: int, embedding_dim: int, class_task: List[int], mode: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = backbone.get_classifier()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.mode = str(mode)
        self.register_buffer("class_task", torch.as_tensor(class_task, dtype=torch.long))
        self.register_buffer("prototype_bank", torch.zeros(num_classes, embedding_dim))
        self.register_buffer("prototype_valid", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("hist_reliability", torch.ones(num_classes))
        self.register_buffer("uncertainty_ema", torch.zeros(num_classes))
        self.register_buffer("last_coverage", torch.zeros(num_classes))
        self.register_buffer("last_step_gate", torch.zeros(num_classes))
        oas_classes = num_classes if str(mode) in OAS_MODES else 0
        self.register_buffer("raw_count", torch.zeros(oas_classes, dtype=torch.long))
        self.register_buffer("raw_mean", torch.zeros(oas_classes, embedding_dim))
        self.register_buffer("raw_scatter", torch.zeros(oas_classes, embedding_dim, embedding_dim))
        self.register_buffer("lda_weight", torch.zeros(oas_classes, embedding_dim))
        self.register_buffer("lda_bias", torch.zeros(oas_classes))
        self.register_buffer("lda_shrinkage", torch.ones(()))
        self.register_buffer("lda_fitted", torch.zeros((), dtype=torch.bool))

    def encode(self, features, coords=None, patch_size_level0=None) -> torch.Tensor:
        output = self.backbone.forward_with_embedding(features, coords, patch_size_level0)
        embedding = output.get("embedding") if isinstance(output, dict) else None
        if not torch.is_tensor(embedding) or embedding.shape != (1, self.embedding_dim):
            raise ValueError(f"ATLAS-v3 ACL expects slide embedding [1,{self.embedding_dim}]")
        return embedding.float()

    def logits(self, embedding: torch.Tensor, seen_classes: int, *, use_statistics: bool, use_lda: bool) -> torch.Tensor:
        if not use_statistics:
            logits = self.classifier(embedding)
        elif use_lda:
            if not bool(self.lda_fitted):
                raise RuntimeError("ATLAS-v3 ACL OAS-LDA has not been fitted")
            features = embedding.float()
            if self.mode in NORMALIZED_OAS_MODES:
                features = F.normalize(features, dim=1, eps=1.0e-8)
            logits = F.linear(features, self.lda_weight, self.lda_bias)
        else:
            z = F.normalize(embedding.float(), dim=1, eps=1.0e-8)
            prototypes = F.normalize(self.prototype_bank, dim=1, eps=1.0e-8)
            logits = z @ prototypes.t()
        logits = logits.clone()
        if use_statistics:
            logits = logits.masked_fill(~self.prototype_valid.unsqueeze(0), float("-inf"))
        if int(seen_classes) < self.num_classes:
            logits[:, int(seen_classes):] = float("-inf")
        return logits

    def forward(self, features, coords=None, patch_size_level0=None, *, seen_classes=None, use_statistics=True, use_lda=False):
        if isinstance(features, (list, tuple)):
            values = features
            features = values[0]
            coords = values[1] if len(values) > 1 else coords
            patch_size_level0 = values[2] if len(values) > 2 else patch_size_level0
        embedding = self.encode(features, coords, patch_size_level0)
        limit = self.num_classes if seen_classes is None else int(seen_classes)
        logits = self.logits(embedding, limit, use_statistics=bool(use_statistics), use_lda=bool(use_lda))
        attention = torch.full((1, features.shape[0]), 1.0 / features.shape[0], device=features.device, dtype=features.dtype)
        return logits, logits.softmax(1), logits.argmax(1), attention, logits.sum() * 0.0


class AtlasV3ACL(ContinualModel):
    NAME = "atlas_v3_acl"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("feather",)
    REQUIRED_FEATURE_DIM = 768
    REQUIRES_TRAINABLE_BACKBONE = True
    CHECKPOINT_INCLUDE_OPTIMIZER = False
    CHECKPOINT_VERSION = CHECKPOINT_VERSION

    def __init__(self, backbone, loss, args: Namespace, transform):
        validate_args(args)
        classifier = backbone.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError("ATLAS-v3 ACL requires a linear FEATHER classifier")
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        class_task: List[int] = []
        for task, count in enumerate(args.task_num_classes):
            class_task.extend([task] * int(count))
        network = AtlasV3ACLNetwork(
            backbone,
            int(args.num_classes),
            int(classifier.in_features),
            class_task,
            str(args.atlasv3_acl_mode),
        )
        super().__init__(network, loss, args, transform)
        self.mode = str(args.atlasv3_acl_mode)
        self.TRAINING_FREE = self.mode == "frozen_raw_oas"
        self.num_classes = int(args.num_classes)
        self.embedding_dim = int(classifier.in_features)
        if str(getattr(args, "backbone_model_id", "")) == FEATHER_MODEL_ID and str(getattr(args, "backbone_revision", "")) == FEATHER_REVISION and self.embedding_dim != 512:
            raise ValueError(f"Pinned FEATHER must expose slide_embedding_dim=512, got {self.embedding_dim}")
        self.task_num_classes = tuple(int(value) for value in args.task_num_classes)
        self.class_offsets = tuple(int(value) for value in args.class_offsets)
        self.task_order = tuple(str(value) for value in args.task_order)
        self.n_tasks = int(args.n_tasks)
        self._validate_layout()
        self.current_task = 0
        self.completed_tasks = 0
        self.old_class_count = 0
        self.seen_class_count = self.task_num_classes[0]
        self.transport_history: List[Dict[str, Any]] = []
        self._pair_pre_raw: torch.Tensor | None = None
        self._pair_labels: torch.Tensor | None = None
        self._pair_indices: torch.Tensor | None = None
        # Expose the true adaptation parameter count to the shared resource
        # manifest before the first begin_task hook. FWT evaluation is no-grad.
        self._set_encoder_trainable(not self.TRAINING_FREE)
        self._reset_optimizer()

    def _validate_layout(self) -> None:
        if not (len(self.task_num_classes) == len(self.class_offsets) == len(self.task_order) == self.n_tasks):
            raise ValueError("ATLAS-v3 ACL task metadata lengths differ")
        expected = 0
        for offset, count in zip(self.class_offsets, self.task_num_classes):
            if offset != expected or count <= 0:
                raise ValueError("ATLAS-v3 ACL requires contiguous positive task classes")
            expected += count
        if expected != self.num_classes:
            raise ValueError("ATLAS-v3 ACL task classes do not cover the classifier")

    def _bounds(self, task: int) -> Tuple[int, int]:
        start = self.class_offsets[int(task)]
        return start, start + self.task_num_classes[int(task)]

    def _set_encoder_trainable(self, enabled: bool) -> None:
        self.net.backbone.requires_grad_(bool(enabled))
        self.net.classifier.requires_grad_(False)

    def _reset_optimizer(self) -> None:
        trainable = [parameter for parameter in self.net.backbone.parameters() if parameter.requires_grad]
        self.opt = build_optimizer(trainable or [self.net.classifier.weight], self.args)

    @torch.no_grad()
    def _collect_loader(self, source_loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loader = DataLoader(source_loader.dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=source_loader.collate_fn)
        embeddings, labels = [], []
        was_training = self.net.training
        self.net.eval()
        try:
            for batch in loader:
                features, coords, patch_size = self.prepare_inputs(batch.features, batch.coords, batch.patch_size_level0, training=False)
                embeddings.append(self.net.encode(features, coords, patch_size).cpu())
                labels.append(int(batch.labels.reshape(-1)[0]))
        finally:
            self.net.train(was_training)
        if not embeddings:
            raise RuntimeError("ATLAS-v3 ACL current train loader is empty")
        values = torch.cat(embeddings)
        target = torch.as_tensor(labels, dtype=torch.long)
        return values, target, torch.arange(values.shape[0], dtype=torch.long)

    @torch.no_grad()
    def _set_current_anchors(self, raw: torch.Tensor, labels: torch.Tensor) -> None:
        normalized = F.normalize(raw.float(), dim=1, eps=1.0e-8)
        start, stop = self._bounds(self.current_task)
        for label in range(start, stop):
            members = normalized[labels == label]
            if members.shape[0] == 0:
                raise RuntimeError(f"Current train split has no class {label}")
            self.net.prototype_bank[label].copy_(F.normalize(members.mean(0), dim=0).to(self.net.prototype_bank))
            self.net.prototype_valid[label] = True

    def begin_task(self, dataset) -> None:
        task = int(dataset.current_task) - 1
        if task != self.completed_tasks or not 0 <= task < self.n_tasks:
            raise RuntimeError("ATLAS-v3 ACL tasks must be learned sequentially")
        self.current_task = task
        self.old_class_count, self.seen_class_count = self._bounds(task)
        if self.TRAINING_FREE:
            self._set_encoder_trainable(False)
            return
        raw, labels, indices = self._collect_loader(dataset.train_loader)
        self._pair_pre_raw, self._pair_labels, self._pair_indices = raw, labels, indices
        self._set_current_anchors(raw, labels)
        self._set_encoder_trainable(True)
        self._reset_optimizer()

    def _acl_loss(self, embedding: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start, stop = self._bounds(self.current_task)
        z = F.normalize(embedding.float(), dim=1, eps=1.0e-8)
        anchors = F.normalize(self.net.prototype_bank[start:stop], dim=1, eps=1.0e-8)
        logits = z @ anchors.t() / float(self.args.atlasv3_acl_temperature)
        local = label.long().reshape(-1) - start
        return F.cross_entropy(logits, local), z

    def _historical_loss(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.mode not in HISTNEG_MODES or self.old_class_count == 0:
            return z.sum() * 0.0
        positive = F.normalize(self.net.prototype_bank[label.long().reshape(-1)], dim=1, eps=1.0e-8)
        positive_similarity = (z * positive).sum(1)
        old = F.normalize(self.net.prototype_bank[:self.old_class_count], dim=1, eps=1.0e-8)
        similarities = z @ old.t()
        count = min(int(self.args.atlasv3_acl_hist_topk), self.old_class_count)
        values, indices = similarities.topk(count, dim=1)
        reliability = self.net.hist_reliability[:self.old_class_count][indices].clamp_min(float(self.args.atlasv3_acl_reliability_floor))
        scaled = (values - positive_similarity[:, None] + float(self.args.atlasv3_acl_hist_margin)) / float(self.args.atlasv3_acl_hist_temperature)
        log_mass = torch.logsumexp(scaled + reliability.log(), dim=1) - torch.log(scaled.new_tensor(float(count)))
        return F.softplus(log_mass).mean()

    def _task_margin_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode != "gated_task_margin" or self.current_task == 0:
            return z.sum() * 0.0
        start, stop = self._bounds(self.current_task)
        current = F.normalize(self.net.prototype_bank[start:stop], dim=1, eps=1.0e-8).mean(0)
        current = F.normalize(current, dim=0, eps=1.0e-8)
        old_centroids = []
        for task in range(self.current_task):
            left, right = self._bounds(task)
            centroid = F.normalize(self.net.prototype_bank[left:right], dim=1, eps=1.0e-8).mean(0)
            old_centroids.append(F.normalize(centroid, dim=0, eps=1.0e-8))
        old = torch.stack(old_centroids)
        hardest = (z @ old.t()).max(1).values
        current_similarity = z @ current
        return F.relu(float(self.args.atlasv3_acl_task_margin) + hardest - current_similarity).mean()

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("ATLAS-v3 ACL has no SSL phase")
        if self.TRAINING_FREE:
            raise RuntimeError("Frozen raw-OAS control has no adaptation phase")
        if not batches:
            raise ValueError("ATLAS-v3 ACL observe_many requires current WSI bags")
        if task is not None and int(task) != self.current_task:
            raise RuntimeError("ATLAS-v3 ACL received a non-active task")
        acl_values, hist_values, task_values = [], [], []
        for features, coords, patch_size, labels in batches:
            label = labels.long().reshape(-1)
            if not self.old_class_count <= int(label.item()) < self.seen_class_count:
                raise ValueError("Current label is outside the active task")
            embedding = self.net.encode(features, coords, patch_size)
            acl, z = self._acl_loss(embedding, label)
            acl_values.append(acl)
            hist_values.append(self._historical_loss(z, label))
            task_values.append(self._task_margin_loss(z))
        loss_acl = torch.stack(acl_values).mean()
        loss_hist = torch.stack(hist_values).mean()
        loss_task = torch.stack(task_values).mean()
        total = loss_acl + float(self.args.atlasv3_acl_hist_weight) * loss_hist + float(self.args.atlasv3_acl_task_weight) * loss_task
        if not torch.isfinite(total):
            raise FloatingPointError("ATLAS-v3 ACL produced non-finite loss")
        self.opt.zero_grad(set_to_none=True)
        self.backward_loss(total)
        self.optimizer_step()
        return {
            "loss": float(total.detach()),
            "loss_acl": float(loss_acl.detach()),
            "loss_histneg": float(loss_hist.detach()),
            "loss_task_margin": float(loss_task.detach()),
            "replay_bags": 0.0,
            "buffer_size": 0.0,
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many([(features, coords, patch_size, labels)], task=task, ssl=ssl)

    @torch.no_grad()
    def _sdc(self, old: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        drift = target - source
        sigma = float(self.args.atlasv3_acl_sdc_sigma)
        distances = torch.cdist(old, source).square()
        weights = torch.exp(-distances / (2.0 * sigma * sigma))
        update = weights @ drift / weights.sum(1, keepdim=True).clamp_min(1.0e-8)
        return F.normalize(old + update, dim=1, eps=1.0e-8)

    @torch.no_grad()
    def _update_reliability(self, gates: torch.Tensor, uncertainty: torch.Tensor) -> None:
        old = slice(0, self.old_class_count)
        momentum = float(self.args.atlasv3_acl_reliability_momentum)
        floor = float(self.args.atlasv3_acl_reliability_floor)
        reliability = (1.0 - momentum) * self.net.hist_reliability[old] + momentum * gates.to(self.net.hist_reliability)
        self.net.hist_reliability[old].copy_(reliability.clamp(floor, 1.0))
        finite = torch.isfinite(uncertainty)
        if bool(finite.any()):
            current = self.net.uncertainty_ema[old]
            updated = current.clone()
            updated[finite] = 0.9 * current[finite] + 0.1 * uncertainty.to(current)[finite]
            self.net.uncertainty_ema[old].copy_(updated)

    @torch.no_grad()
    def _transport_old(self, pre_raw: torch.Tensor, post_raw: torch.Tensor) -> Dict[str, Any]:
        row: Dict[str, Any] = {"task": self.current_task, "transport_kind": self.mode, "old_class_count": self.old_class_count}
        if self.old_class_count == 0:
            row.update({"effective_rank": 0.0, "transport_fallback_reason": "no_old_classes"})
            return row
        if self.mode in {
            "oas_static", "normalized_oas_static", "histneg_normalized_oas_static",
        }:
            row.update({"effective_rank": 0.0, "transport_fallback_reason": "static_old_statistics"})
            return row
        if self.mode not in TRANSPORT_MODES:
            row.update({"effective_rank": 0.0, "transport_fallback_reason": "static_bank"})
            return row
        device = self.device
        source_norm = F.normalize(pre_raw.to(device), dim=1, eps=1.0e-8)
        target_norm = F.normalize(post_raw.to(device), dim=1, eps=1.0e-8)
        old = self.net.prototype_bank[:self.old_class_count].detach().float().to(device)
        ridge = float(self.args.atlasv3_acl_transport_ridge)
        rank = int(self.args.atlasv3_acl_transport_rank)
        if self.mode == "sdc":
            updated = self._sdc(old, source_norm, target_norm)
            self.net.prototype_bank[:self.old_class_count].copy_(updated.to(self.net.prototype_bank))
            row.update({"effective_rank": 0.0, "pair_train_mse": "", "transport_fallback_reason": "local_class_weighted_drift"})
            return row
        if self.mode == "ldc":
            matrix, diagnostics = fit_ldc(source_norm, target_norm, steps=int(self.args.atlasv3_acl_ldc_steps), learning_rate=float(self.args.atlasv3_acl_ldc_lr))
            updated = F.normalize(old @ matrix, dim=1, eps=1.0e-8)
            self.net.prototype_bank[:self.old_class_count].copy_(updated.to(self.net.prototype_bank))
            row.update(diagnostics)
            row.update({"effective_rank": float(self.embedding_dim), "transport_fallback_reason": ""})
            return row
        if self.mode == "sldc":
            matrix, diagnostics = fit_sldc(source_norm, target_norm, ridge=ridge)
            updated = F.normalize(old @ matrix, dim=1, eps=1.0e-8)
            self.net.prototype_bank[:self.old_class_count].copy_(updated.to(self.net.prototype_bank))
            row.update(diagnostics)
            row.update({"effective_rank": float(self.embedding_dim), "transport_fallback_reason": ""})
            return row

        oas_transport = self.mode in OAS_TRANSPORT_MODES
        raw_transport = self.mode in RAW_TRANSPORT_MODES
        source = pre_raw.to(device) if raw_transport else source_norm
        target = post_raw.to(device) if raw_transport else target_norm
        points = (
            self.net.raw_mean[:self.old_class_count].detach().float().to(device)
            if oas_transport
            else old
        )
        main = fit_lowrank_residual(source, target, rank=rank, ridge=ridge)
        row.update(main.diagnostics)
        if self.mode in GATED_MODES:
            coverage = (
                distribution_coverage(
                    points,
                    self.net.raw_scatter[:self.old_class_count].to(device),
                    self.net.raw_count[:self.old_class_count].to(device),
                    source,
                    energy=float(self.args.atlasv3_acl_coverage_energy),
                )
                if oas_transport
                else mean_coverage(points, source, energy=float(self.args.atlasv3_acl_coverage_energy))
            )
            gates, uncertainty, bootstrap = bootstrap_gates(
                points, source, target, coverage, main,
                rank=rank, ridge=ridge,
                samples=int(self.args.atlasv3_acl_bootstrap_samples),
                beta=float(self.args.atlasv3_acl_uncertainty_beta),
                seed=int(getattr(self.args, "seed", 0) or 0) + 1009 * int(getattr(self.args, "fold", 0) or 0) + 104729 * self.current_task,
            )
            row.update(bootstrap)
            row.update(summarize(coverage, "coverage"))
            row.update(summarize(gates, "step_gate"))
            row.update(summarize(uncertainty, "bootstrap_uncertainty"))
            self.net.last_coverage[:self.old_class_count].copy_(coverage.to(self.net.last_coverage))
        else:
            gates = torch.ones(self.old_class_count, device=device)
            uncertainty = torch.full_like(gates, float("nan"))
        mean_gates = gates * float(self.args.atlasv3_acl_transport_mean_scale)
        covariance_gates = gates * float(self.args.atlasv3_acl_transport_cov_scale)
        row.update(summarize(mean_gates, "applied_mean_gate"))
        if oas_transport:
            row.update(summarize(covariance_gates, "applied_covariance_gate"))
        self.net.last_step_gate[:self.old_class_count].copy_(
            mean_gates.to(self.net.last_step_gate)
        )
        self._update_reliability(mean_gates, uncertainty)
        mapped = main.map(points, mean_gates)
        if not torch.isfinite(mapped).all():
            raise FloatingPointError("ATLAS-v3 ACL transport produced non-finite means")
        if oas_transport:
            self.net.raw_mean[:self.old_class_count].copy_(mapped.to(self.net.raw_mean))
            identity = torch.eye(self.embedding_dim, device=device)
            for label in range(self.old_class_count):
                transform = identity + covariance_gates[label] * main.delta
                scatter = self.net.raw_scatter[label].to(device)
                transported = transform.t() @ scatter @ transform
                transported = 0.5 * (transported + transported.t())
                if not torch.isfinite(transported).all():
                    raise FloatingPointError("ATLAS-v3 ACL transport produced non-finite scatter")
                self.net.raw_scatter[label].copy_(transported.to(self.net.raw_scatter))
            normalized = F.normalize(mapped, dim=1, eps=1.0e-8)
        else:
            normalized = F.normalize(mapped, dim=1, eps=1.0e-8)
        self.net.prototype_bank[:self.old_class_count].copy_(normalized.to(self.net.prototype_bank))
        row.update(summarize(self.net.hist_reliability[:self.old_class_count], "hist_reliability"))
        row["transport_fallback_reason"] = "" if main.effective_rank > 0 else "zero_effective_rank"
        return row

    @torch.no_grad()
    def _fit_current_statistics(self, post_raw: torch.Tensor, labels: torch.Tensor) -> None:
        normalized = F.normalize(post_raw.float(), dim=1, eps=1.0e-8)
        start, stop = self._bounds(self.current_task)
        for label in range(start, stop):
            mask = labels == label
            raw = post_raw[mask].float().to(self.net.raw_mean)
            norm = normalized[mask].to(self.net.prototype_bank)
            if raw.shape[0] == 0:
                raise RuntimeError(f"Post-adaptation train split has no class {label}")
            if not torch.isfinite(raw).all():
                raise FloatingPointError(f"Post-adaptation class {label} contains non-finite embeddings")
            statistics = norm.to(self.net.raw_mean) if self.mode in NORMALIZED_OAS_MODES else raw
            prototype = F.normalize(
                statistics.mean(0) if self.mode in OAS_MODES else norm.mean(0),
                dim=0,
                eps=1.0e-8,
            )
            self.net.prototype_bank[label].copy_(prototype.to(self.net.prototype_bank))
            self.net.prototype_valid[label] = True
            self.net.hist_reliability[label] = 1.0
            self.net.uncertainty_ema[label] = 0.0
            self.net.last_coverage[label] = 1.0
            self.net.last_step_gate[label] = 1.0
            if self.mode in OAS_MODES:
                mean = statistics.mean(0)
                centered = statistics - mean
                self.net.raw_count[label] = int(statistics.shape[0])
                self.net.raw_mean[label].copy_(mean)
                self.net.raw_scatter[label].copy_(centered.t() @ centered)

    @torch.no_grad()
    def _fit_oracle_statistics(
        self,
        dataset,
        current_raw: torch.Tensor,
        current_labels: torch.Tensor,
    ) -> int:
        """Recompute all seen statistics with old train data (diagnostic only)."""

        if not hasattr(dataset, "_datasets_for_task"):
            raise RuntimeError(
                "OAS oracle requires seq-wsi task datasets to revisit old train splits"
            )
        raw_parts = []
        label_parts = []
        revisited = 0
        collate_fn = dataset.train_loader.collate_fn
        fold = int(getattr(self.args, "fold", 0))
        for task_id in range(self.current_task):
            train_set = dataset._datasets_for_task(task_id, fold)[0]
            source = DataLoader(
                train_set,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
            raw, labels, _ = self._collect_loader(source)
            raw_parts.append(raw)
            label_parts.append(labels)
            revisited += int(raw.shape[0])
        raw_parts.append(current_raw)
        label_parts.append(current_labels)
        all_raw = torch.cat(raw_parts).float()
        all_labels = torch.cat(label_parts).long()
        for label in range(self.seen_class_count):
            values = all_raw[all_labels == label].to(self.net.raw_mean)
            if values.shape[0] == 0:
                raise RuntimeError(f"OAS oracle has no train samples for class {label}")
            mean = values.mean(0)
            centered = values - mean
            self.net.prototype_bank[label].copy_(
                F.normalize(mean, dim=0, eps=1.0e-8).to(self.net.prototype_bank)
            )
            self.net.prototype_valid[label] = True
            self.net.raw_count[label] = int(values.shape[0])
            self.net.raw_mean[label].copy_(mean)
            self.net.raw_scatter[label].copy_(centered.t() @ centered)
            self.net.hist_reliability[label] = 1.0
            self.net.uncertainty_ema[label] = 0.0
            self.net.last_coverage[label] = 1.0
            self.net.last_step_gate[label] = 1.0
        return revisited

    @torch.no_grad()
    def _oracle_drift_diagnostics(
        self,
        pre_raw: torch.Tensor,
        post_raw: torch.Tensor,
        old_mean: torch.Tensor,
        old_scatter: torch.Tensor,
        old_count: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compare no correction, ungated transport, and gated transport to oracle statistics."""

        if self.old_class_count == 0:
            return {
                "oracle_class_diagnostics": [],
                "oracle_probe_status": "no_old_classes",
            }
        device = self.device
        source = pre_raw.detach().float().to(device)
        target = post_raw.detach().float().to(device)
        means_before = old_mean.detach().float().to(device)
        scatters_before = old_scatter.detach().float().to(device)
        counts_before = old_count.detach().long().to(device)
        rank = int(self.args.atlasv3_acl_transport_rank)
        ridge = float(self.args.atlasv3_acl_transport_ridge)
        fitted = fit_lowrank_residual(source, target, rank=rank, ridge=ridge)
        coverage = distribution_coverage(
            means_before,
            scatters_before,
            counts_before,
            source,
            energy=float(self.args.atlasv3_acl_coverage_energy),
        )
        gates, uncertainty, bootstrap = bootstrap_gates(
            means_before,
            source,
            target,
            coverage,
            fitted,
            rank=rank,
            ridge=ridge,
            samples=int(self.args.atlasv3_acl_bootstrap_samples),
            beta=float(self.args.atlasv3_acl_uncertainty_beta),
            seed=int(getattr(self.args, "seed", 0) or 0)
            + 1009 * int(getattr(self.args, "fold", 0) or 0)
            + 104729 * self.current_task,
        )
        ones = torch.ones_like(gates)
        ungated_mean = fitted.map(means_before, ones)
        gated_mean = fitted.map(means_before, gates)
        identity = torch.eye(self.embedding_dim, device=device)

        def transport_scatters(step_gates: torch.Tensor) -> torch.Tensor:
            outputs = []
            for label in range(self.old_class_count):
                transform = identity + step_gates[label] * fitted.delta
                value = transform.t() @ scatters_before[label] @ transform
                outputs.append(0.5 * (value + value.t()))
            return torch.stack(outputs)

        ungated_scatter = transport_scatters(ones)
        gated_scatter = transport_scatters(gates)
        oracle_mean = self.net.raw_mean[:self.old_class_count].detach().float().to(device)
        oracle_scatter = self.net.raw_scatter[:self.old_class_count].detach().float().to(device)
        oracle_count = self.net.raw_count[:self.old_class_count].detach().long().to(device)
        class_rows: List[Dict[str, Any]] = []
        metric_names = (
            "mean_drift_cosine",
            "mean_residual_ungated_cosine",
            "mean_residual_gated_cosine",
            "mean_error_ungated_l2",
            "mean_error_gated_l2",
            "mean_error_ungated_relative_l2",
            "mean_error_gated_relative_l2",
            "covariance_drift_frobenius",
            "covariance_residual_ungated_frobenius",
            "covariance_residual_gated_frobenius",
            "covariance_drift_relative_frobenius",
            "covariance_residual_ungated_relative_frobenius",
            "covariance_residual_gated_relative_frobenius",
            "mean_gain_ungated",
            "mean_gain_gated",
            "covariance_gain_ungated",
            "covariance_gain_gated",
        )
        collected: Dict[str, List[float]] = {name: [] for name in metric_names}
        for label in range(self.old_class_count):
            degrees_before = max(int(counts_before[label]) - 1, 1)
            degrees_oracle = max(int(oracle_count[label]) - 1, 1)
            covariance_before = scatters_before[label] / float(degrees_before)
            covariance_ungated = ungated_scatter[label] / float(degrees_before)
            covariance_gated = gated_scatter[label] / float(degrees_before)
            covariance_oracle = oracle_scatter[label] / float(degrees_oracle)
            oracle_mean_norm = torch.linalg.vector_norm(oracle_mean[label]).clamp_min(1.0e-8)
            oracle_covariance_norm = torch.linalg.matrix_norm(covariance_oracle).clamp_min(1.0e-8)

            def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
                left = F.normalize(left, dim=0, eps=1.0e-8)
                right = F.normalize(right, dim=0, eps=1.0e-8)
                return float(1.0 - (left * right).sum().clamp(-1.0, 1.0))

            mean_drift = cosine_distance(means_before[label], oracle_mean[label])
            mean_residual_ungated = cosine_distance(ungated_mean[label], oracle_mean[label])
            mean_residual_gated = cosine_distance(gated_mean[label], oracle_mean[label])
            mean_l2_ungated = float(torch.linalg.vector_norm(ungated_mean[label] - oracle_mean[label]))
            mean_l2_gated = float(torch.linalg.vector_norm(gated_mean[label] - oracle_mean[label]))
            covariance_drift = float(torch.linalg.matrix_norm(covariance_before - covariance_oracle))
            covariance_residual_ungated = float(torch.linalg.matrix_norm(covariance_ungated - covariance_oracle))
            covariance_residual_gated = float(torch.linalg.matrix_norm(covariance_gated - covariance_oracle))
            values = {
                "mean_drift_cosine": mean_drift,
                "mean_residual_ungated_cosine": mean_residual_ungated,
                "mean_residual_gated_cosine": mean_residual_gated,
                "mean_error_ungated_l2": mean_l2_ungated,
                "mean_error_gated_l2": mean_l2_gated,
                "mean_error_ungated_relative_l2": mean_l2_ungated / float(oracle_mean_norm),
                "mean_error_gated_relative_l2": mean_l2_gated / float(oracle_mean_norm),
                "covariance_drift_frobenius": covariance_drift,
                "covariance_residual_ungated_frobenius": covariance_residual_ungated,
                "covariance_residual_gated_frobenius": covariance_residual_gated,
                "covariance_drift_relative_frobenius": covariance_drift / float(oracle_covariance_norm),
                "covariance_residual_ungated_relative_frobenius": covariance_residual_ungated / float(oracle_covariance_norm),
                "covariance_residual_gated_relative_frobenius": covariance_residual_gated / float(oracle_covariance_norm),
                "mean_gain_ungated": mean_drift - mean_residual_ungated,
                "mean_gain_gated": mean_drift - mean_residual_gated,
                "covariance_gain_ungated": (covariance_drift - covariance_residual_ungated) / float(oracle_covariance_norm),
                "covariance_gain_gated": (covariance_drift - covariance_residual_gated) / float(oracle_covariance_norm),
            }
            for name, value in values.items():
                collected[name].append(float(value))
            origin_task = int(self.net.class_task[label])
            class_rows.append({
                "after_task": int(self.current_task),
                "class_id": int(label),
                "origin_task": origin_task,
                "class_age": int(self.current_task - origin_task),
                "sample_count": int(oracle_count[label]),
                "coverage": float(coverage[label]),
                "step_gate": float(gates[label]),
                "bootstrap_uncertainty": (
                    float(uncertainty[label])
                    if bool(torch.isfinite(uncertainty[label]))
                    else ""
                ),
                **values,
            })
        row: Dict[str, Any] = {
            "oracle_class_diagnostics": class_rows,
            "oracle_probe_status": "ok",
            "oracle_probe_scope": "one_step_from_oracle_previous_statistics",
            **{f"oracle_probe_{key}": value for key, value in fitted.diagnostics.items()},
            **{f"oracle_probe_{key}": value for key, value in bootstrap.items()},
        }
        row.update({f"oracle_probe_{key}": value for key, value in summarize(coverage, "coverage").items()})
        row.update({f"oracle_probe_{key}": value for key, value in summarize(gates, "step_gate").items()})
        row.update({f"oracle_probe_{key}": value for key, value in summarize(uncertainty, "bootstrap_uncertainty").items()})
        for name, values in collected.items():
            row[f"oracle_{name}_mean"] = float(sum(values) / len(values))
        return row

    @torch.no_grad()
    def _fit_oas(self) -> None:
        seen = self.seen_class_count
        counts = self.net.raw_count[:seen]
        if bool((counts <= 0).any()):
            raise RuntimeError("Raw OAS-LDA is missing seen-class statistics")
        degrees = int((counts - 1).clamp_min(0).sum())
        scatter = self.net.raw_scatter[:seen].sum(0).float()
        empirical = scatter / float(max(degrees, 1))
        dimensions = self.embedding_dim
        mu = empirical.trace() / float(dimensions)
        epsilon = torch.finfo(empirical.dtype).eps
        if degrees == 0 or not torch.isfinite(mu) or float(mu) <= epsilon:
            mu = empirical.new_tensor(1.0)
            shrinkage = empirical.new_tensor(1.0)
        else:
            alpha = empirical.square().mean()
            denominator = float(degrees + 1) * (alpha - mu.square() / float(dimensions))
            shrinkage = empirical.new_tensor(1.0) if not torch.isfinite(denominator) or float(denominator) <= epsilon else ((alpha + mu.square()) / denominator).clamp(0.0, 1.0)
        covariance = (1.0 - shrinkage) * empirical
        covariance.diagonal().add_(shrinkage * mu)
        covariance = 0.5 * (covariance + covariance.t())
        covariance.diagonal().add_(torch.clamp(mu * 1.0e-6, min=1.0e-6))
        means = self.net.raw_mean[:seen].float()
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        precision_means = torch.cholesky_solve(means.t(), cholesky).t() if int(info.item()) == 0 else means @ torch.linalg.pinv(covariance)
        bias = -0.5 * (means * precision_means).sum(1)
        if not torch.isfinite(precision_means).all() or not torch.isfinite(bias).all():
            raise FloatingPointError("ATLAS-v3 ACL OAS-LDA produced non-finite parameters")
        self.net.lda_weight.zero_()
        self.net.lda_bias.zero_()
        self.net.lda_weight[:seen].copy_(precision_means.to(self.net.lda_weight))
        self.net.lda_bias[:seen].copy_(bias.to(self.net.lda_bias))
        self.net.lda_shrinkage.copy_(shrinkage.to(self.net.lda_shrinkage))
        self.net.lda_fitted.fill_(True)

    def end_task(self, dataset=None) -> None:
        if dataset is None:
            raise RuntimeError("ATLAS-v3 ACL end_task requires current train data")
        self._set_encoder_trainable(False)
        post_raw, labels, indices = self._collect_loader(dataset.train_loader)
        if self.TRAINING_FREE:
            self._fit_current_statistics(post_raw, labels)
            self._fit_oas()
            row = {"task": self.current_task, "transport_kind": "frozen_raw_oas", "transport_fallback_reason": "frozen_control", "effective_rank": 0.0}
        elif self.mode == "oas_oracle":
            if self._pair_pre_raw is None or self._pair_labels is None or self._pair_indices is None:
                raise RuntimeError("OAS oracle pre-adaptation pair cache is missing")
            if not torch.equal(labels, self._pair_labels) or not torch.equal(indices, self._pair_indices):
                raise RuntimeError("Pre/post oracle pairs are misaligned")
            old_mean = self.net.raw_mean[:self.old_class_count].clone()
            old_scatter = self.net.raw_scatter[:self.old_class_count].clone()
            old_count = self.net.raw_count[:self.old_class_count].clone()
            revisited = self._fit_oracle_statistics(dataset, post_raw, labels)
            self._fit_oas()
            row = {
                "task": self.current_task,
                "transport_kind": "oas_oracle",
                "transport_fallback_reason": "diagnostic_old_train_recompute",
                "effective_rank": 0.0,
                "oracle_revisited_wsis": int(revisited),
                "diagnostic_only": True,
            }
            row.update(self._oracle_drift_diagnostics(
                self._pair_pre_raw,
                post_raw,
                old_mean,
                old_scatter,
                old_count,
            ))
        else:
            if self._pair_pre_raw is None or self._pair_labels is None or self._pair_indices is None:
                raise RuntimeError("ATLAS-v3 ACL pre-adaptation pair cache is missing")
            if not torch.equal(labels, self._pair_labels) or not torch.equal(indices, self._pair_indices):
                raise RuntimeError("Pre/post transport pairs are misaligned")
            row = self._transport_old(self._pair_pre_raw, post_raw)
            self._fit_current_statistics(post_raw, labels)
            if self.mode in OAS_MODES:
                self._fit_oas()
        row.update({"retained_wsis": 0, "stored_statistics_bytes": self._statistics_bytes()})
        self.transport_history.append(row)
        self._pair_pre_raw = self._pair_labels = self._pair_indices = None
        self.completed_tasks = self.current_task + 1
        print(f"[atlas_v3_acl] finalized task {self.current_task}: retained_wsis=0")

    def _statistics_bytes(self) -> int:
        names = (
            "prototype_bank", "prototype_valid", "hist_reliability",
            "uncertainty_ema", "last_coverage", "last_step_gate",
        )
        if self.mode in OAS_MODES:
            names += ("raw_count", "raw_mean", "raw_scatter", "lda_weight", "lda_bias")
        return int(sum(getattr(self.net, name).numel() * getattr(self.net, name).element_size() for name in names))

    def forward(self, x, coords=None, patch_size_level0=None):
        use_statistics = bool(self.net.prototype_valid[:self.seen_class_count].all())
        use_lda = self.mode in OAS_MODES and self.completed_tasks > self.current_task
        with self.autocast_context():
            if isinstance(x, (list, tuple)):
                return self.net(x, seen_classes=self.seen_class_count, use_statistics=use_statistics, use_lda=use_lda)
            return self.net(x, coords, patch_size_level0, seen_classes=self.seen_class_count, use_statistics=use_statistics, use_lda=use_lda)

    def _config(self) -> Dict[str, Any]:
        keys = [key for key in vars(self.args) if key.startswith("atlasv3_acl_")]
        config = {
            "version": CHECKPOINT_VERSION,
            "backbone": "feather",
            "mode": self.mode,
            "training_free": self.TRAINING_FREE,
            "hyperparameters": {key: getattr(self.args, key) for key in sorted(keys)},
            "ablation_id": getattr(self.args, "ablation_id", None),
            "ablation_config_hash": getattr(self.args, "ablation_config_hash", None),
        }
        if self.mode == "oas_oracle":
            config.update({
                "diagnostic_only": True,
                "revisits_old_train_data": True,
            })
        if self.mode in OAS_DIAGNOSTIC_SEMANTICS:
            config["implementation_semantics"] = OAS_DIAGNOSTIC_SEMANTICS[self.mode]
        return config

    def get_run_metadata(self) -> Dict[str, Any]:
        config = self._config()
        classifier_parameters = {id(parameter) for parameter in self.net.classifier.parameters()}
        adaptation_parameters = sum(
            parameter.numel()
            for parameter in self.net.backbone.parameters()
            if id(parameter) not in classifier_parameters
        )
        return {
            "atlas_v3_acl_config": config,
            "atlas_v3_acl_config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "transport_accounting": {
                "stored_slide_embeddings": 0,
                "retained_wsis": 0,
                "oracle_revisited_wsis": int(sum(int(row.get("oracle_revisited_wsis", 0)) for row in self.transport_history)),
                "stored_statistics_bytes": self._statistics_bytes(),
                "adaptation_parameter_count": int(adaptation_parameters),
                "history": list(self.transport_history),
            },
        }

    def get_task_diagnostics(self) -> Dict[str, Any] | None:
        if not self.transport_history:
            return None
        return dict(self.transport_history[-1])

    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
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
            "transport_history": list(self.transport_history),
        }

    def load_checkpoint_state(self, state: Mapping[str, Any], strict: bool = True) -> None:
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
                raise ValueError(f"ATLAS-v3 ACL checkpoint mismatch for {key}")
        restored_task = int(state["current_task"])
        restored_completed = int(state["completed_tasks"])
        if not 0 <= restored_task < self.n_tasks or restored_completed not in {
            restored_task,
            restored_task + 1,
        }:
            raise ValueError("ATLAS-v3 ACL checkpoint task state is invalid")
        expected_old, expected_seen = self._bounds(restored_task)
        if strict and (
            int(state.get("old_class_count", -1)) != expected_old
            or int(state.get("seen_class_count", -1)) != expected_seen
        ):
            raise ValueError("ATLAS-v3 ACL checkpoint class boundaries are invalid")
        history = [dict(row) for row in state.get("transport_history", [])]
        if strict and len(history) != restored_completed:
            raise ValueError("ATLAS-v3 ACL checkpoint transport history is incomplete")
        preserve_in_process_pairs = (
            restored_completed == restored_task
            and self._pair_pre_raw is not None
            and self._pair_labels is not None
            and self._pair_indices is not None
        )
        self.current_task = restored_task
        self.completed_tasks = restored_completed
        self.old_class_count = int(state["old_class_count"])
        self.seen_class_count = int(state["seen_class_count"])
        self.transport_history = history
        if not preserve_in_process_pairs:
            self._pair_pre_raw = self._pair_labels = self._pair_indices = None
        self._set_encoder_trainable(not self.TRAINING_FREE and self.completed_tasks == self.current_task)
        self._reset_optimizer()


def build_model_from_components(args, loss, transform, backbone) -> AtlasV3ACL:
    return AtlasV3ACL(backbone, loss, args, transform)


def build_model(args: Namespace, loss, transform) -> AtlasV3ACL:
    validate_args(args)
    backbone = build_mil_backbone(args, int(args.num_classes))
    return build_model_from_components(args, loss, transform, backbone)


ATLASV3ACL = AtlasV3ACL
