"""Replay-free ACL adaptation with historical feature transport for FEATHER."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
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
    fit_full_residual,
    fit_lowrank_residual,
    summarize,
)
from models.utils.continual_model import ContinualModel
from utils.args import add_experiment_args, add_management_args
from utils.optim import build_optimizer


CHECKPOINT_VERSION = 1
ACL_MODES = (
    "acl",
    "normalized_oas_static",
    "transport_normalized_oas",
    "gated_transport_normalized_oas_no_histneg",
)
TRANSPORT_MODES = {
    "transport_normalized_oas", "gated_transport_normalized_oas_no_histneg",
}
GATED_MODES = {
    "gated_transport_normalized_oas_no_histneg",
}
OAS_MODES = {
    "normalized_oas_static",
    "transport_normalized_oas", "gated_transport_normalized_oas_no_histneg",
}
NORMALIZED_OAS_MODES = OAS_MODES
OAS_DIAGNOSTIC_SEMANTICS = {
    "normalized_oas_static": "acl_only_normalized_oas_static_no_transport_v1",
    "transport_normalized_oas": "acl_only_normalized_oas_ungated_lowrank_transport_v1",
    "gated_transport_normalized_oas_no_histneg": "acl_only_normalized_oas_gated_lowrank_transport_v1",
}
FULL_RANK_DIAGNOSTIC_SEMANTICS = (
    "acl_only_normalized_oas_gated_full_ridge_transport_v1"
)


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
    parser.add_argument("--atlasv3_acl_transport_rank", type=int, default=8)
    parser.add_argument(
        "--atlasv3_acl_transport_full_rank",
        action=BooleanOptionalAction,
        default=False,
        help="Use the complete ridge residual without SVD rank truncation.",
    )
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
    parser.add_argument("--atlasv3_acl_coverage_energy", type=float, default=0.95)
    parser.add_argument("--atlasv3_acl_bootstrap_samples", type=int, default=20)
    parser.add_argument("--atlasv3_acl_uncertainty_beta", type=float, default=10.0)
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
    if int(getattr(args, "n_epochs", -1)) < 0:
        raise ValueError("n_epochs must be non-negative")
    if str(getattr(args, "atlasv3_acl_mode", "")) not in ACL_MODES:
        raise ValueError("Unknown ATLAS-v3 ACL mode")
    positive = ("atlasv3_acl_temperature", "atlasv3_acl_transport_ridge")
    for name in positive:
        if float(getattr(args, name, 0.0)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(getattr(args, "atlasv3_acl_transport_rank", 0)) <= 0:
        raise ValueError("atlasv3_acl_transport_rank must be positive")
    for name in (
        "atlasv3_acl_transport_mean_scale",
        "atlasv3_acl_transport_cov_scale",
    ):
        value = float(getattr(args, name, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if int(getattr(args, "atlasv3_acl_bootstrap_samples", -1)) < 0:
        raise ValueError("atlasv3_acl_bootstrap_samples must be non-negative")
    energy = float(getattr(args, "atlasv3_acl_coverage_energy", -1.0))
    if not 0.0 < energy <= 1.0:
        raise ValueError("atlasv3_acl_coverage_energy must be in (0,1]")
    if float(getattr(args, "atlasv3_acl_uncertainty_beta", -1.0)) < 0.0:
        raise ValueError("atlasv3_acl_uncertainty_beta must be non-negative")


class AtlasV3ACLNetwork(nn.Module):
    supports_ssl = False

    def __init__(self, backbone: nn.Module, num_classes: int, embedding_dim: int, mode: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = backbone.get_classifier()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.mode = str(mode)
        self.register_buffer("prototype_bank", torch.zeros(num_classes, embedding_dim))
        self.register_buffer("prototype_valid", torch.zeros(num_classes, dtype=torch.bool))
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
        network = AtlasV3ACLNetwork(
            backbone,
            int(args.num_classes),
            int(classifier.in_features),
            str(args.atlasv3_acl_mode),
        )
        super().__init__(network, loss, args, transform)
        self.mode = str(args.atlasv3_acl_mode)
        # Epoch zero is a deliberate frozen-encoder control in the ACL-epoch
        # ablation.  Marking it training-free also prevents the shared trainer
        # from trying to restore a validation checkpoint that cannot exist.
        self.TRAINING_FREE = int(getattr(args, "n_epochs", 1)) == 0
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
        raw, labels, indices = self._collect_loader(dataset.train_loader)
        self._pair_pre_raw, self._pair_labels, self._pair_indices = raw, labels, indices
        self._set_current_anchors(raw, labels)
        self._set_encoder_trainable(not self.TRAINING_FREE)
        self._reset_optimizer()

    def _acl_loss(self, embedding: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        start, stop = self._bounds(self.current_task)
        z = F.normalize(embedding.float(), dim=1, eps=1.0e-8)
        anchors = F.normalize(self.net.prototype_bank[start:stop], dim=1, eps=1.0e-8)
        logits = z @ anchors.t() / float(self.args.atlasv3_acl_temperature)
        local = label.long().reshape(-1) - start
        return F.cross_entropy(logits, local)

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("ATLAS-v3 ACL has no SSL phase")
        if not batches:
            raise ValueError("ATLAS-v3 ACL observe_many requires current WSI bags")
        if task is not None and int(task) != self.current_task:
            raise RuntimeError("ATLAS-v3 ACL received a non-active task")
        acl_values = []
        for features, coords, patch_size, labels in batches:
            label = labels.long().reshape(-1)
            if not self.old_class_count <= int(label.item()) < self.seen_class_count:
                raise ValueError("Current label is outside the active task")
            embedding = self.net.encode(features, coords, patch_size)
            acl_values.append(self._acl_loss(embedding, label))
        loss_acl = torch.stack(acl_values).mean()
        if not torch.isfinite(loss_acl):
            raise FloatingPointError("ATLAS-v3 ACL produced non-finite loss")
        self.opt.zero_grad(set_to_none=True)
        self.backward_loss(loss_acl)
        self.optimizer_step()
        return {
            "loss": float(loss_acl.detach()),
            "loss_acl": float(loss_acl.detach()),
            "replay_bags": 0.0,
            "buffer_size": 0.0,
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many([(features, coords, patch_size, labels)], task=task, ssl=ssl)

    @torch.no_grad()
    def _transport_old(self, pre_raw: torch.Tensor, post_raw: torch.Tensor) -> Dict[str, Any]:
        row: Dict[str, Any] = {"task": self.current_task, "transport_kind": self.mode, "old_class_count": self.old_class_count}
        if self.old_class_count == 0:
            row.update({"effective_rank": 0.0, "transport_fallback_reason": "no_old_classes"})
            return row
        if self.mode == "normalized_oas_static":
            row.update({"effective_rank": 0.0, "transport_fallback_reason": "static_old_statistics"})
            return row
        if self.mode not in TRANSPORT_MODES:
            row.update({"effective_rank": 0.0, "transport_fallback_reason": "static_bank"})
            return row
        device = self.device
        source_norm = F.normalize(pre_raw.to(device), dim=1, eps=1.0e-8)
        target_norm = F.normalize(post_raw.to(device), dim=1, eps=1.0e-8)
        ridge = float(self.args.atlasv3_acl_transport_ridge)
        rank = int(self.args.atlasv3_acl_transport_rank)
        full_rank = bool(self.args.atlasv3_acl_transport_full_rank)
        source, target = source_norm, target_norm
        points = self.net.raw_mean[:self.old_class_count].detach().float().to(device)
        main = (
            fit_full_residual(source, target, ridge=ridge)
            if full_rank
            else fit_lowrank_residual(source, target, rank=rank, ridge=ridge)
        )
        row.update(main.diagnostics)
        row["requested_rank"] = "full" if full_rank else rank
        if self.mode in GATED_MODES:
            coverage = distribution_coverage(
                points,
                self.net.raw_scatter[:self.old_class_count].to(device),
                self.net.raw_count[:self.old_class_count].to(device),
                source,
                energy=float(self.args.atlasv3_acl_coverage_energy),
            )
            gates, uncertainty, bootstrap = bootstrap_gates(
                points, source, target, coverage, main,
                rank=rank, ridge=ridge,
                samples=int(self.args.atlasv3_acl_bootstrap_samples),
                beta=float(self.args.atlasv3_acl_uncertainty_beta),
                seed=int(getattr(self.args, "seed", 0) or 0) + 1009 * int(getattr(self.args, "fold", 0) or 0) + 104729 * self.current_task,
                full_rank=full_rank,
            )
            row.update(bootstrap)
            row.update(summarize(coverage, "coverage"))
            row.update(summarize(gates, "step_gate"))
            row.update(summarize(uncertainty, "bootstrap_uncertainty"))
            self.net.last_coverage[:self.old_class_count].copy_(coverage.to(self.net.last_coverage))
        else:
            gates = torch.ones(self.old_class_count, device=device)
        mean_gates = gates * float(self.args.atlasv3_acl_transport_mean_scale)
        covariance_gates = gates * float(self.args.atlasv3_acl_transport_cov_scale)
        row.update(summarize(mean_gates, "applied_mean_gate"))
        row.update(summarize(covariance_gates, "applied_covariance_gate"))
        self.net.last_step_gate[:self.old_class_count].copy_(
            mean_gates.to(self.net.last_step_gate)
        )
        mapped = main.map(points, mean_gates)
        if not torch.isfinite(mapped).all():
            raise FloatingPointError("ATLAS-v3 ACL transport produced non-finite means")
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
        self.net.prototype_bank[:self.old_class_count].copy_(normalized.to(self.net.prototype_bank))
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
            self.net.last_coverage[label] = 1.0
            self.net.last_step_gate[label] = 1.0
            if self.mode in OAS_MODES:
                mean = statistics.mean(0)
                centered = statistics - mean
                self.net.raw_count[label] = int(statistics.shape[0])
                self.net.raw_mean[label].copy_(mean)
                self.net.raw_scatter[label].copy_(centered.t() @ centered)

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
            "prototype_bank", "prototype_valid", "last_coverage", "last_step_gate",
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
        if (
            self.mode == "gated_transport_normalized_oas_no_histneg"
            and bool(self.args.atlasv3_acl_transport_full_rank)
        ):
            config["implementation_semantics"] = FULL_RANK_DIAGNOSTIC_SEMANTICS
        elif self.mode in OAS_DIAGNOSTIC_SEMANTICS:
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
