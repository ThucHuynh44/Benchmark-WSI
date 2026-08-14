"""ATLAS-MIL for continual whole-slide classification.

The active implementation combines prompt-anchored latent geometry, masked
latent reconstruction, attention-aware pseudo-bag replay, and fixed-rank
semantic LoRA merging while preserving Benchmark-WSI's global classifier and
variable-length bag contracts.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone.generic_mil import build_mil_backbone
from backbone.pretrained_mil import TITAN_MODEL_ID, TITAN_REVISION, _resolve_snapshot
from configs.qpmil_vl_prompts import prompt_schema_hash, resolve_class_prompts
from models.qpmil_vl import build_class_features
from models.utils.amil_memory import maxminrand_select
from models.utils.atlas_lora import (
    AtlasLoRALinear,
    attach_atlas_lora,
    merge_atlas_lora,
)
from models.utils.atlas_memory import AtlasMemoryPool, AtlasReplayBag
from models.utils.continual_model import ContinualModel
from models.utils.wsi_replay import unpack_prepared_batch
from utils.args import add_experiment_args, add_management_args
from utils.optim import build_optimizer


CHECKPOINT_VERSION = 1


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ATLAS-MIL continual WSI learning")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--minibatch_size", type=int, default=1)
    parser.add_argument("--bags_per_update", type=int, default=1)
    parser.add_argument("--pmp_k", type=int, default=400)
    parser.add_argument("--atlas_rank", type=int, default=8)
    parser.add_argument("--atlas_nce_temperature", type=float, default=0.07)
    parser.add_argument("--atlas_centroid_momentum", type=float, default=0.99)
    parser.add_argument("--latent_mask_ratio", type=float, default=0.5)
    parser.add_argument("--atlas_nce_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--manifold_weight", type=float, default=1.0)
    parser.add_argument("--attention_weight", type=float, default=1.0)
    parser.add_argument("--atlas_prompt_weight", type=float, default=0.5)
    parser.add_argument("--atlas_logit_scale", type=float, default=10.0)
    parser.add_argument("--atlas_lora_rank", type=int, default=8)
    parser.add_argument(
        "--atlas_lora_mode", choices=("semantic", "hard", "none"), default="semantic"
    )
    parser.add_argument("--atlas_lora_merge_scale", type=float, default=1.0)
    parser.add_argument("--atlas_text_model_id", type=str, default=TITAN_MODEL_ID)
    parser.add_argument("--atlas_text_revision", type=str, default=TITAN_REVISION)
    return parser


def validate_args(args) -> None:
    backbone = str(getattr(args, "backbone", "generic_mil")).lower()
    if backbone not in AtlasMil.SUPPORTED_BACKBONES:
        raise ValueError(
            f"ATLAS-MIL supports only {AtlasMil.SUPPORTED_BACKBONES}, got {backbone!r}"
        )
    if int(getattr(args, "feature_dim", 768)) != 768:
        raise ValueError("ATLAS-MIL requires 768-D CONCH patch features")
    if bool(getattr(args, "backbone_freeze", False)):
        raise ValueError("ATLAS-MIL owns backbone freezing; --backbone_freeze is invalid")
    if int(getattr(args, "backbone_max_patches", 0) or 0) != 0:
        raise ValueError("ATLAS-MIL requires backbone_max_patches=0 for full-bag attention")

    for name, default in {
        "buffer_size": 30,
        "minibatch_size": 1,
        "bags_per_update": 1,
        "pmp_k": 400,
        "atlas_rank": 8,
        "atlas_lora_rank": 8,
    }.items():
        if int(getattr(args, name, default)) <= 0:
            raise ValueError(f"ATLAS-MIL {name} must be positive")
    required_capacity = int(getattr(args, "num_classes", 27))
    if int(getattr(args, "buffer_size", 30)) < required_capacity:
        raise ValueError(
            "ATLAS-MIL buffer_size must cover every global class "
            f"(at least {required_capacity})"
        )
    for name, default in {
        "atlas_nce_weight": 1.0,
        "reconstruction_weight": 1.0,
        "manifold_weight": 1.0,
        "attention_weight": 1.0,
    }.items():
        if float(getattr(args, name, default)) < 0.0:
            raise ValueError(f"ATLAS-MIL {name} must be non-negative")
    for name, default in {
        "atlas_nce_temperature": 0.07,
        "atlas_logit_scale": 10.0,
        "atlas_lora_merge_scale": 1.0,
    }.items():
        if float(getattr(args, name, default)) <= 0.0:
            raise ValueError(f"ATLAS-MIL {name} must be positive")
    for name, default in {
        "atlas_centroid_momentum": 0.99,
        "latent_mask_ratio": 0.5,
        "atlas_prompt_weight": 0.5,
    }.items():
        value = float(getattr(args, name, default))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"ATLAS-MIL {name} must be in [0, 1]")
    if not str(getattr(args, "atlas_text_model_id", "")).strip():
        raise ValueError("ATLAS-MIL atlas_text_model_id cannot be empty")
    if not str(getattr(args, "atlas_text_revision", "")).strip():
        raise ValueError("ATLAS-MIL atlas_text_revision cannot be empty")


class AtlasNetwork(nn.Module):
    """Backbone wrapper implementing the hybrid atlas classifier and decoder."""

    supports_ssl = False
    has_genuine_patch_attention = True

    def __init__(
        self,
        backbone: nn.Module,
        semantic_anchors: torch.Tensor,
        *,
        embedding_dim: int,
        atlas_rank: int,
        prompt_weight: float,
        logit_scale: float,
        centroid_momentum: float,
    ) -> None:
        super().__init__()
        anchors = torch.as_tensor(semantic_anchors, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[0] <= 0 or anchors.shape[1] <= 0:
            raise ValueError("ATLAS semantic anchors must have shape [C,D]")
        if not torch.isfinite(anchors).all():
            raise ValueError("ATLAS semantic anchors contain NaN or Inf")
        self.backbone = backbone
        self.num_classes = int(anchors.shape[0])
        self.semantic_dim = int(anchors.shape[1])
        self.embedding_dim = int(embedding_dim)
        self.atlas_rank = int(atlas_rank)
        self.prompt_weight = float(prompt_weight)
        self.logit_scale = float(logit_scale)
        self.centroid_momentum = float(centroid_momentum)

        self.register_buffer("semantic_anchors", F.normalize(anchors, dim=1))
        self.prompt_projector = nn.Linear(
            self.semantic_dim, self.embedding_dim, bias=False
        )
        if self.semantic_dim == self.embedding_dim:
            with torch.no_grad():
                self.prompt_projector.weight.copy_(
                    torch.eye(self.embedding_dim, dtype=self.prompt_projector.weight.dtype)
                )
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.register_buffer(
            "atlas_centroids", torch.zeros(self.num_classes, self.embedding_dim)
        )
        self.register_buffer(
            "atlas_subspaces",
            torch.zeros(self.num_classes, self.embedding_dim, self.atlas_rank),
        )
        self.register_buffer(
            "atlas_variances", torch.zeros(self.num_classes, self.atlas_rank)
        )
        self.register_buffer(
            "atlas_effective_ranks", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "atlas_sample_counts", torch.zeros(self.num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "atlas_valid", torch.zeros(self.num_classes, dtype=torch.bool)
        )
        self.register_buffer(
            "atlas_finalized", torch.zeros(self.num_classes, dtype=torch.bool)
        )

    def projected_prompts(self) -> torch.Tensor:
        return F.normalize(self.prompt_projector(self.semantic_anchors), dim=1, eps=1.0e-6)

    def encode(self, features, coords=None, patch_size_level0=None) -> Dict[str, torch.Tensor]:
        output = self.backbone.forward_with_embedding(
            features, coords, patch_size_level0
        )
        if not isinstance(output, dict):
            raise TypeError("ATLAS backbone forward_with_embedding must return a dictionary")
        embedding = output.get("embedding")
        attention = output.get("attention")
        if not torch.is_tensor(embedding) or embedding.shape != (1, self.embedding_dim):
            shape = tuple(embedding.shape) if torch.is_tensor(embedding) else None
            raise ValueError(
                f"ATLAS expects slide embedding [1,{self.embedding_dim}], got {shape}"
            )
        if not torch.is_tensor(attention) or attention.ndim != 2 or attention.shape[0] != 1:
            raise ValueError("ATLAS requires genuine attention with shape [1,N]")
        attention = attention.float()
        if torch.any(attention < 0) or not torch.isfinite(attention).all():
            raise ValueError("ATLAS attention must be finite and non-negative")
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        return {
            "embedding": embedding.float(),
            "attention": attention,
            "auxiliary_loss": output.get("auxiliary_loss", embedding.sum() * 0.0),
        }

    def logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embedding.float(), dim=1, eps=1.0e-6)
        prompt_scores = normalized @ self.projected_prompts().t()
        centroid_scores = normalized @ F.normalize(
            self.atlas_centroids.float(), dim=1, eps=1.0e-6
        ).t()
        valid = self.atlas_valid.unsqueeze(0)
        blended = (
            self.prompt_weight * prompt_scores
            + (1.0 - self.prompt_weight) * centroid_scores
        )
        scores = torch.where(valid, blended, prompt_scores)
        return self.logit_scale * scores

    def forward_with_embedding(self, features, coords=None, patch_size_level0=None):
        encoded = self.encode(features, coords, patch_size_level0)
        encoded["logits"] = self.logits_from_embedding(encoded["embedding"])
        return encoded

    def forward(self, features, coords=None, patch_size_level0=None, **_):
        if isinstance(features, (list, tuple)):
            values = features
            features = values[0]
            coords = values[1] if len(values) > 1 else coords
            patch_size_level0 = values[2] if len(values) > 2 else patch_size_level0
        output = self.forward_with_embedding(features, coords, patch_size_level0)
        logits = output["logits"]
        return (
            logits,
            F.softmax(logits, dim=1),
            logits.argmax(dim=1),
            output["attention"],
            output["auxiliary_loss"],
        )

    @torch.no_grad()
    def update_running_centroid(self, label: int, embedding: torch.Tensor) -> None:
        label = int(label)
        value = embedding.detach().float().reshape(-1).to(self.atlas_centroids)
        if not bool(self.atlas_valid[label]):
            self.atlas_centroids[label].copy_(value)
            self.atlas_valid[label] = True
            self.atlas_sample_counts[label] = 1
        else:
            momentum = self.centroid_momentum
            self.atlas_centroids[label].mul_(momentum).add_(value, alpha=1.0 - momentum)
            self.atlas_sample_counts[label] += 1

    @torch.no_grad()
    def finalize_class(self, label: int, embeddings: torch.Tensor) -> None:
        label = int(label)
        values = torch.as_tensor(embeddings, dtype=torch.float32).reshape(
            -1, self.embedding_dim
        )
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("ATLAS cannot finalize an empty/non-finite class")
        mean = values.mean(dim=0)
        centered = values - mean
        effective = min(self.atlas_rank, max(0, values.shape[0] - 1), self.embedding_dim)
        subspace = values.new_zeros(self.embedding_dim, self.atlas_rank)
        variance = values.new_zeros(self.atlas_rank)
        if effective > 0 and float(centered.square().sum()) > 0.0:
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            effective = min(effective, int(singular.numel()))
            subspace[:, :effective] = vh[:effective].t()
            variance[:effective] = singular[:effective].square() / float(
                max(values.shape[0] - 1, 1)
            )
        else:
            effective = 0
        self.atlas_centroids[label].copy_(mean.to(self.atlas_centroids))
        self.atlas_subspaces[label].copy_(subspace.to(self.atlas_subspaces))
        self.atlas_variances[label].copy_(variance.to(self.atlas_variances))
        self.atlas_effective_ranks[label] = effective
        self.atlas_sample_counts[label] = int(values.shape[0])
        self.atlas_valid[label] = True
        self.atlas_finalized[label] = True


class AtlasMil(ContinualModel):
    NAME = "atlas_mil"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("generic_mil", "feather")
    REQUIRED_FEATURE_DIM = 768
    REQUIRES_TRAINABLE_BACKBONE = False
    CHECKPOINT_INCLUDE_OPTIMIZER = False
    CHECKPOINT_VERSION = CHECKPOINT_VERSION

    def __init__(
        self,
        backbone: nn.Module,
        loss,
        args: Namespace,
        transform,
        semantic_anchors: torch.Tensor,
    ) -> None:
        validate_args(args)
        self._validate_backbone(backbone, args)
        classifier = backbone.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError("ATLAS-MIL requires an nn.Linear backbone classifier")
        num_classes = int(getattr(args, "num_classes", 0))
        if classifier.out_features != num_classes:
            raise ValueError("ATLAS-MIL classifier/global class count mismatch")
        if int(semantic_anchors.shape[0]) != num_classes:
            raise ValueError("ATLAS-MIL semantic anchor/global class count mismatch")

        backbone_name = str(args.backbone).lower()
        root = backbone.model if backbone_name == "feather" else backbone
        root_name = "backbone.model" if backbone_name == "feather" else "backbone"
        lora_modules = attach_atlas_lora(
            root,
            root_name=root_name,
            classifier=classifier,
            rank=int(args.atlas_lora_rank),
        )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for module in lora_modules.values():
            module.active_down.requires_grad_(True)
            module.active_up.requires_grad_(True)
        classifier.requires_grad_(False)

        network = AtlasNetwork(
            backbone,
            semantic_anchors,
            embedding_dim=int(classifier.in_features),
            atlas_rank=int(args.atlas_rank),
            prompt_weight=float(args.atlas_prompt_weight),
            logit_scale=float(args.atlas_logit_scale),
            centroid_momentum=float(args.atlas_centroid_momentum),
        )
        super().__init__(network, loss, args, transform)
        object.__setattr__(self, "_lora_modules", lora_modules)

        self.num_classes = num_classes
        self.feature_dim = int(args.feature_dim)
        self.embedding_dim = int(classifier.in_features)
        self.task_num_classes = tuple(int(value) for value in args.task_num_classes)
        self.class_offsets = tuple(int(value) for value in args.class_offsets)
        self.task_order = tuple(str(value) for value in args.task_order)
        self.n_tasks = int(args.n_tasks)
        self._validate_task_layout()
        if int(args.buffer_size) < self.num_classes:
            raise ValueError(
                "ATLAS-MIL buffer_size must be at least num_classes so every seen "
                "class retains one pseudo-bag"
            )

        self.minibatch_size = int(args.minibatch_size)
        self.bags_per_update = int(args.bags_per_update)
        self.pmp_k = int(args.pmp_k)
        self.nce_temperature = float(args.atlas_nce_temperature)
        self.mask_ratio = float(args.latent_mask_ratio)
        self.nce_weight = float(args.atlas_nce_weight)
        self.reconstruction_weight = float(args.reconstruction_weight)
        self.manifold_weight = float(args.manifold_weight)
        self.attention_weight = float(args.attention_weight)
        self.lora_mode = str(args.atlas_lora_mode)
        self.lora_merge_scale = float(args.atlas_lora_merge_scale)
        self.text_model_id = str(args.atlas_text_model_id)
        self.text_revision = str(args.atlas_text_revision)
        self.prompt_hash = prompt_schema_hash(self.task_order, self.task_num_classes)

        seed = 0 if getattr(args, "seed", None) is None else int(args.seed)
        self.memory = AtlasMemoryPool(
            int(args.buffer_size),
            pmp_k=self.pmp_k,
            feature_dim=self.feature_dim,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            seed=seed,
        )
        self.mask_generator = torch.Generator(device="cpu")
        self.mask_generator.manual_seed(seed + 113)
        self.current_task = 0
        self.completed_tasks = 0
        self.old_class_count = 0
        self.seen_class_count = self.task_num_classes[0]
        self.semantic_rho_history: List[float] = []
        self._buffer_update_started = False
        self._reset_optimizer()

    @property
    def lora_modules(self) -> Mapping[str, AtlasLoRALinear]:
        return self.__dict__["_lora_modules"]

    @classmethod
    def _validate_backbone(cls, backbone, args) -> None:
        if str(args.backbone).lower() not in cls.SUPPORTED_BACKBONES:
            raise ValueError("ATLAS-MIL received an unsupported backbone")
        if not bool(getattr(backbone, "has_genuine_patch_attention", False)):
            raise ValueError("ATLAS-MIL requires genuine patch attention")
        if not callable(getattr(backbone, "forward_with_embedding", None)):
            raise TypeError("ATLAS-MIL backbone must expose forward_with_embedding()")
        if not callable(getattr(backbone, "get_classifier", None)):
            raise TypeError("ATLAS-MIL backbone must expose get_classifier()")

    def _validate_task_layout(self) -> None:
        if (
            len(self.task_num_classes) != self.n_tasks
            or len(self.class_offsets) != self.n_tasks
            or len(self.task_order) != self.n_tasks
        ):
            raise ValueError("ATLAS-MIL task metadata lengths do not match n_tasks")
        expected = 0
        for offset, count in zip(self.class_offsets, self.task_num_classes):
            if int(count) <= 0 or int(offset) != expected:
                raise ValueError("ATLAS-MIL requires contiguous positive task classes")
            expected += int(count)
        if expected != self.num_classes:
            raise ValueError("ATLAS-MIL task classes do not cover the global classifier")

    def _reset_optimizer(self) -> None:
        trainable = [parameter for parameter in self.net.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("ATLAS-MIL has no trainable adaptation parameters")
        self.opt = build_optimizer(trainable, self.args)

    def _task_bounds(self, task: int) -> Tuple[int, int]:
        start = self.class_offsets[int(task)]
        return start, start + self.task_num_classes[int(task)]

    def _configure_task(self, task: int) -> None:
        task = int(task)
        if not 0 <= task < self.n_tasks:
            raise ValueError("ATLAS-MIL task index is outside the stream")
        if self.completed_tasks != task:
            raise RuntimeError(
                f"ATLAS-MIL task sequence mismatch: completed={self.completed_tasks}, task={task}"
            )
        self.current_task = task
        self.old_class_count, self.seen_class_count = self._task_bounds(task)

    def begin_task(self, dataset) -> None:
        if self._buffer_update_started:
            raise RuntimeError("ATLAS-MIL cannot begin during memory refresh")
        task = int(getattr(dataset, "current_task", 0)) - 1
        self._configure_task(task)
        expected_snapshot = task - 1
        if task == 0:
            if len(self.memory) != 0:
                raise RuntimeError("ATLAS-MIL task 0 requires empty memory")
        elif len(self.memory) == 0 or self.memory.target_snapshot_task != expected_snapshot:
            raise RuntimeError("ATLAS-MIL memory snapshot is missing or stale")
        self._reset_optimizer()

    def _unpack(self, batch: Sequence[torch.Tensor]):
        return unpack_prepared_batch(batch, feature_dim=self.feature_dim)

    def _forward_bag(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        features, coords, patch_size, label = self._unpack(batch)
        output = self.net.forward_with_embedding(features, coords, patch_size)
        output["label"] = label.long().reshape(-1)
        if output["label"].numel() != 1:
            raise ValueError("ATLAS-MIL expects one label per WSI")
        return output

    @staticmethod
    def attention_distillation_loss(
        student: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if student.shape != target.shape:
            raise ValueError("ATLAS student/teacher attention shapes differ")
        epsilon = torch.finfo(torch.float32).eps
        student = student.float().clamp_min(epsilon)
        target = target.to(student).float().clamp_min(epsilon)
        student = student / student.sum(dim=1, keepdim=True)
        target = target / target.sum(dim=1, keepdim=True)
        return F.kl_div(student.log(), target, reduction="batchmean")

    def _atlas_nce_loss(self, embedding: torch.Tensor, label: int) -> torch.Tensor:
        label = int(label)
        z = F.normalize(embedding.float(), dim=1, eps=1.0e-6)
        prompts = self.net.projected_prompts()
        positives = [prompts[label : label + 1]]
        if bool(self.net.atlas_valid[label]):
            positives.append(self.net.atlas_centroids[label : label + 1])
        positives.extend(self.memory.embeddings_for_label(label, self.device))
        negatives: List[torch.Tensor] = []
        for other in range(self.seen_class_count):
            if other == label:
                continue
            negatives.append(prompts[other : other + 1])
            if bool(self.net.atlas_valid[other]):
                negatives.append(self.net.atlas_centroids[other : other + 1])
        positive_tensor = F.normalize(
            torch.cat([value.reshape(1, -1).to(z) for value in positives], dim=0),
            dim=1,
            eps=1.0e-6,
        )
        negative_tensor = (
            F.normalize(
                torch.cat([value.reshape(1, -1).to(z) for value in negatives], dim=0),
                dim=1,
                eps=1.0e-6,
            )
            if negatives
            else z.new_zeros((0, self.embedding_dim))
        )
        positive_logits = (z @ positive_tensor.t()).reshape(-1) / self.nce_temperature
        all_logits = torch.cat(
            [positive_logits, (z @ negative_tensor.t()).reshape(-1) / self.nce_temperature]
        )
        return -(torch.logsumexp(positive_logits, dim=0) - torch.logsumexp(all_logits, dim=0))

    def _masked_reconstruction(
        self, embedding: torch.Tensor, label: int
    ) -> torch.Tensor:
        random = torch.rand(
            embedding.shape,
            generator=self.mask_generator,
            device="cpu",
            dtype=torch.float32,
        ).to(embedding.device)
        keep = (random >= self.mask_ratio).to(embedding.dtype)
        prompt = self.net.projected_prompts()[int(label) : int(label) + 1]
        return self.net.decoder(torch.cat((embedding * keep, prompt), dim=1))

    def _manifold_loss(self, reconstructed: torch.Tensor, label: int) -> torch.Tensor:
        label = int(label)
        rank = int(self.net.atlas_effective_ranks[label])
        if rank <= 0 or not bool(self.net.atlas_finalized[label]):
            return reconstructed.sum() * 0.0
        centered = reconstructed - self.net.atlas_centroids[label : label + 1]
        basis = self.net.atlas_subspaces[label, :, :rank]
        residual = centered - (centered @ basis) @ basis.t()
        return residual.square().mean()

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("ATLAS-MIL does not define a separate SSL phase")
        if not batches:
            raise ValueError("ATLAS-MIL observe_many requires current WSI bags")
        if self._buffer_update_started:
            raise RuntimeError("ATLAS-MIL cannot train during memory refresh")
        if task is not None and int(task) != self.current_task:
            raise RuntimeError("ATLAS-MIL received a non-active task")

        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        current_outputs = [self._forward_bag(batch) for batch in batches]
        replay_entries = (
            self.memory.sample(self.minibatch_size, self.device)
            if self.old_class_count > 0
            else []
        )
        replay_outputs = [
            (entry, self._forward_bag(
                (entry.features, entry.coords, entry.patch_size, entry.label)
            ))
            for entry in replay_entries
        ]

        ce_losses: List[torch.Tensor] = []
        nce_losses: List[torch.Tensor] = []
        rec_losses: List[torch.Tensor] = []
        manifold_losses: List[torch.Tensor] = []
        attention_losses: List[torch.Tensor] = []

        for output in current_outputs:
            label = int(output["label"].item())
            if not self.old_class_count <= label < self.seen_class_count:
                raise ValueError("ATLAS-MIL current label is outside the active task")
            ce_losses.append(
                F.cross_entropy(output["logits"][:, : self.seen_class_count], output["label"])
            )
            nce_losses.append(self._atlas_nce_loss(output["embedding"], label))
            reconstructed = self._masked_reconstruction(output["embedding"], label)
            rec_losses.append(F.mse_loss(reconstructed, output["embedding"].detach()))

        for entry, output in replay_outputs:
            label = int(output["label"].item())
            if label >= self.old_class_count:
                raise ValueError("ATLAS-MIL replay label is not from an old task")
            if entry.target_attention is None or entry.target_embedding is None:
                raise RuntimeError("ATLAS-MIL replay entry lacks teacher targets")
            ce_losses.append(
                F.cross_entropy(output["logits"][:, : self.seen_class_count], output["label"])
            )
            nce_losses.append(self._atlas_nce_loss(output["embedding"], label))
            reconstructed = self._masked_reconstruction(output["embedding"], label)
            rec_losses.append(F.mse_loss(reconstructed, entry.target_embedding.to(reconstructed)))
            manifold_losses.append(self._manifold_loss(reconstructed, label))
            attention_losses.append(
                self.attention_distillation_loss(
                    output["attention"], entry.target_attention
                )
            )

        zero = current_outputs[0]["embedding"].sum() * 0.0
        loss_ce = torch.stack(ce_losses).mean()
        loss_nce = torch.stack(nce_losses).mean() if nce_losses else zero
        loss_rec = torch.stack(rec_losses).mean() if rec_losses else zero
        loss_manifold = (
            torch.stack(manifold_losses).mean() if manifold_losses else zero
        )
        loss_attention = (
            torch.stack(attention_losses).mean() if attention_losses else zero
        )
        loss = (
            loss_ce
            + self.nce_weight * loss_nce
            + self.reconstruction_weight * loss_rec
            + self.manifold_weight * loss_manifold
            + self.attention_weight * loss_attention
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("ATLAS-MIL produced a non-finite loss")
        loss.backward()
        self.opt.step()
        for output in current_outputs:
            self.net.update_running_centroid(
                int(output["label"].item()), output["embedding"]
            )
        return {
            "loss": float(loss.detach().cpu()),
            "loss_cls": float(loss_ce.detach().cpu()),
            "loss_atlas_nce": float(loss_nce.detach().cpu()),
            "loss_reconstruction": float(loss_rec.detach().cpu()),
            "loss_manifold": float(loss_manifold.detach().cpu()),
            "loss_attention": float(loss_attention.detach().cpu()),
            "replay_bags": float(len(replay_entries)),
            "buffer_size": float(len(self.memory)),
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def _start_memory_update(self) -> None:
        if self._buffer_update_started:
            return
        if self.current_task == 0:
            if len(self.memory) != 0:
                raise RuntimeError("ATLAS task 0 memory must start empty")
        elif self.memory.target_snapshot_task != self.current_task - 1:
            raise RuntimeError("ATLAS memory snapshot is stale before update")
        self.memory.start_update(
            self.seen_class_count, task_id=self.current_task
        )
        self._buffer_update_started = True

    def save_buffer(self, features, coords, patch_size, labels, task=None) -> int:
        if task is not None and int(task) != self.current_task:
            raise RuntimeError("ATLAS save_buffer task does not match active task")
        features, coords, patch_size, label = self._unpack(
            (features, coords, patch_size, labels)
        )
        value = int(label.item())
        if not self.old_class_count <= value < self.seen_class_count:
            raise ValueError("ATLAS save_buffer accepts only current-task WSIs")
        self._start_memory_update()
        decision = self.memory.consider(value)
        if decision is None:
            return -1
        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                output = self.net.forward_with_embedding(features, coords, patch_size)
                indices = maxminrand_select(
                    output["attention"],
                    self.pmp_k,
                    generator=self.memory.selection_generator,
                ).to(features.device)
                entry = AtlasReplayBag(
                    features=features.index_select(0, indices),
                    coords=coords.index_select(0, indices),
                    patch_size=patch_size,
                    label=label,
                    origin_task_id=self.current_task,
                )
            self.memory.commit(decision, entry)
        finally:
            self.net.train(was_training)
        return 1

    def _semantic_rho(self) -> float:
        if self.current_task == 0:
            return 0.0
        anchors = self.net.semantic_anchors.float()
        start, stop = self._task_bounds(self.current_task)
        current = F.normalize(anchors[start:stop].mean(dim=0), dim=0)
        similarities = []
        for task in range(self.current_task):
            old_start, old_stop = self._task_bounds(task)
            old = F.normalize(anchors[old_start:old_stop].mean(dim=0), dim=0)
            similarities.append(F.cosine_similarity(current, old, dim=0))
        return float(torch.stack(similarities).max().clamp(0.0, 1.0).cpu())

    def _finalize_current_atlas(self, dataset) -> None:
        loader = DataLoader(
            dataset.train_loader.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.train_loader.collate_fn,
        )
        values: Dict[int, List[torch.Tensor]] = defaultdict(list)
        with torch.no_grad():
            for batch in loader:
                features, coords, patch_size = self.prepare_inputs(
                    batch.features, batch.coords, batch.patch_size_level0, training=False
                )
                label = int(batch.labels.reshape(-1)[0])
                output = self.net.forward_with_embedding(features, coords, patch_size)
                values[label].append(output["embedding"].detach().cpu())
        start, stop = self._task_bounds(self.current_task)
        for label in range(start, stop):
            if not values[label]:
                raise RuntimeError(f"ATLAS train split contains no class {label}")
            self.net.finalize_class(label, torch.cat(values[label], dim=0))

    def _refresh_memory_targets(self) -> None:
        targets = []
        with torch.no_grad():
            for entry in self.memory.all(self.device, require_targets=False):
                output = self.net.forward_with_embedding(
                    entry.features, entry.coords, entry.patch_size
                )
                targets.append(
                    (
                        output["attention"].detach().cpu().clone(),
                        output["embedding"].detach().cpu().clone(),
                    )
                )
        self.memory.refresh_targets(
            targets, target_snapshot_task=self.current_task
        )

    def end_task(self, dataset=None) -> None:
        if dataset is None or not self._buffer_update_started:
            raise RuntimeError("ATLAS end_task requires dataset and completed buffer update")
        semantic_rho = self._semantic_rho()
        merge_rho = {
            "semantic": semantic_rho,
            "hard": 0.0,
            "none": 1.0,
        }[self.lora_mode]
        was_training = self.net.training
        self.net.eval()
        try:
            merge_atlas_lora(
                self.lora_modules,
                rho=merge_rho,
                scale=self.lora_merge_scale,
            )
            self._finalize_current_atlas(dataset)
            self._refresh_memory_targets()
        finally:
            self.net.train(was_training)
        self.semantic_rho_history.append(semantic_rho)
        self.completed_tasks = self.current_task + 1
        self._buffer_update_started = False
        print(
            f"[atlas_mil] finalized task {self.current_task}: rho={semantic_rho:.4f}, "
            f"atlas_classes={int(self.net.atlas_finalized.sum())}, memory={len(self.memory)}"
        )

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "backbone": str(self.args.backbone).lower(),
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "buffer_size": self.memory.capacity,
            "minibatch_size": self.minibatch_size,
            "bags_per_update": self.bags_per_update,
            "pmp_k": self.pmp_k,
            "atlas_rank": int(self.args.atlas_rank),
            "nce_temperature": self.nce_temperature,
            "centroid_momentum": float(self.args.atlas_centroid_momentum),
            "mask_ratio": self.mask_ratio,
            "nce_weight": self.nce_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "manifold_weight": self.manifold_weight,
            "attention_weight": self.attention_weight,
            "prompt_weight": float(self.args.atlas_prompt_weight),
            "logit_scale": float(self.args.atlas_logit_scale),
            "lora_rank": int(self.args.atlas_lora_rank),
            "lora_mode": self.lora_mode,
            "lora_merge_scale": self.lora_merge_scale,
            "lora_module_paths": list(self.lora_modules),
            "text_model_id": self.text_model_id,
            "text_revision": self.text_revision,
            "backbone_model_id": getattr(self.args, "backbone_model_id", None),
            "backbone_revision": getattr(self.args, "backbone_revision", None),
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        if self._buffer_update_started or self.memory.refresh_required:
            raise RuntimeError("ATLAS cannot checkpoint an unfinished memory refresh")
        self.memory.all("cpu", require_targets=True)
        return {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "config": self._checkpoint_config(),
            "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "prompt_hash": self.prompt_hash,
            "current_task": self.current_task,
            "completed_tasks": self.completed_tasks,
            "old_class_count": self.old_class_count,
            "seen_class_count": self.seen_class_count,
            "semantic_rho_history": list(self.semantic_rho_history),
            "mask_generator_state": self.mask_generator.get_state().clone(),
            "memory": self.memory.state_dict(),
        }

    def load_checkpoint_state(self, state: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("ATLAS checkpoint method_state must be a dictionary")
        expected = {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "prompt_hash": self.prompt_hash,
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(
                    f"ATLAS checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )
        if strict and state.get("config") != self._checkpoint_config():
            raise ValueError("ATLAS checkpoint hyperparameters do not match this run")
        task = int(state.get("current_task", -1))
        if not 0 <= task < self.n_tasks:
            raise ValueError("ATLAS checkpoint current task is invalid")
        completed = int(state.get("completed_tasks", -1))
        if completed not in {task, task + 1}:
            raise ValueError("ATLAS checkpoint completed-task count is invalid")
        old_count, seen_count = self._task_bounds(task)
        if strict and (
            int(state.get("old_class_count", -1)) != old_count
            or int(state.get("seen_class_count", -1)) != seen_count
        ):
            raise ValueError("ATLAS checkpoint class boundaries are invalid")
        history = [float(value) for value in state.get("semantic_rho_history", [])]
        if len(history) != completed:
            raise ValueError("ATLAS checkpoint semantic history length is invalid")
        self.memory.load_state_dict(state.get("memory"), strict=strict)
        expected_snapshot = completed - 1 if completed > 0 else None
        if len(self.memory) and self.memory.target_snapshot_task != expected_snapshot:
            raise ValueError("ATLAS checkpoint memory snapshot is incompatible")
        self.mask_generator.set_state(
            torch.as_tensor(state["mask_generator_state"], dtype=torch.uint8).cpu()
        )
        self.current_task = task
        self.completed_tasks = completed
        self.old_class_count = old_count
        self.seen_class_count = seen_count
        self.semantic_rho_history = history
        self._buffer_update_started = False
        self._reset_optimizer()


def _load_titan_for_anchors(args: Namespace, device: torch.device):
    from transformers import AutoModel

    model_id = str(args.atlas_text_model_id)
    revision = str(args.atlas_text_revision)
    _resolve_snapshot(
        model_id,
        revision,
        getattr(args, "backbone_cache_dir", None),
        bool(getattr(args, "backbone_allow_download", False)),
    )
    allow_download = bool(getattr(args, "backbone_allow_download", False))
    kwargs = {
        "revision": revision,
        "trust_remote_code": True,
        "local_files_only": not allow_download,
    }
    if getattr(args, "backbone_cache_dir", None) is not None:
        kwargs["cache_dir"] = args.backbone_cache_dir
    if allow_download and os.environ.get("HF_TOKEN"):
        kwargs["token"] = os.environ["HF_TOKEN"]
    model = AutoModel.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model


def build_model_from_components(
    args: Namespace,
    loss,
    transform,
    backbone: nn.Module,
    semantic_anchors: torch.Tensor,
) -> AtlasMil:
    return AtlasMil(backbone, loss, args, transform, semantic_anchors)


def build_model(args: Namespace, loss, transform) -> AtlasMil:
    validate_args(args)
    backbone = build_mil_backbone(args, int(args.num_classes))
    prompts = resolve_class_prompts(args.task_order, args.task_num_classes)
    # Anchor extraction is a one-off build step.  Keeping TITAN on CPU avoids a
    # transient GPU peak while FEATHER is also resident during construction.
    device = torch.device("cpu")
    titan = _load_titan_for_anchors(args, device)
    anchors = build_class_features(titan, prompts, device).detach().cpu()
    del titan
    return build_model_from_components(
        args, loss, transform, backbone, anchors
    )


ATLASMIL = AtlasMil
