"""Active QPMIL-VL integration for TITAN-compatible WSI patch features.

The immutable upstream source is stored under
``third_party/upstream/qpmil_vl``.  This module adapts its queryable prototype
pool, continuous prompt learning, and tunable class vectors to ConSlide's
continual-model interface.  It never imports or executes the upstream snapshot.

QPMIL-VL is licensed CC-BY-NC-ND-4.0 upstream.  This adapted implementation is
for internal research use only and must not be redistributed without a separate
rights review.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.qpmil_vl_prompts import (
    TEMPLATES,
    prompt_schema_hash,
    resolve_class_prompts,
)
from models.utils.continual_model import ContinualModel
from utils.args import add_experiment_args, add_management_args
from utils.conf import get_device
from utils.optim import build_optimizer


DEFAULT_TITAN_MODEL_ID = "MahmoodLab/TITAN"
DEFAULT_TITAN_REVISION = "dac6773d9961cfc75503440676ff157a2c6e8d2e"
ADAPTATION_SCHEMA_VERSION = 1
METHOD_STATE_SCHEMA_VERSION = 1


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="QPMIL-VL with a frozen TITAN text tower"
    )
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--prompt_length", type=int, default=24)
    parser.add_argument("--match_size", type=int, default=5)
    parser.add_argument("--bags_per_update", type=int, default=16)
    parser.add_argument("--pooling", choices=("max", "mean"), default="max")
    parser.add_argument("--csm_logit_scale", type=float, default=100.0)
    parser.add_argument("--classification_logit_scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--matching_loss_weight", type=float, default=0.5)
    parser.add_argument("--class_similarity_loss_weight", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.set_defaults(
        backbone="titan",
        feature_dim=768,
        backbone_max_patches=400,
        lr=1.0e-5,
        optim_wd=1.0e-4,
    )
    return parser


class TitanPromptEncoder(nn.Module):
    """Encode learned token embeddings through TITAN's frozen text tower."""

    _REQUIRED_ATTRIBUTES = (
        "token_embedding",
        "positional_embedding",
        "transformer",
        "ln_final",
        "cls_emb",
        "attn_mask",
        "build_cls_mask",
        "text_projection",
        "tokenizer",
        "pad_id",
    )

    def __init__(self, text_encoder: nn.Module, prompt_length: int):
        super().__init__()
        missing = [
            name for name in self._REQUIRED_ATTRIBUTES
            if not hasattr(text_encoder, name)
        ]
        if missing:
            raise ValueError(
                f"Unsupported TITAN text encoder; missing attributes: {missing}"
            )
        if text_encoder.cls_emb is None:
            raise ValueError("QPMIL-VL requires TITAN's appended CLS text token")

        self.text_encoder = text_encoder
        self.prompt_length = int(prompt_length)
        self.context_length = int(text_encoder.positional_embedding.shape[0])
        self.embedding_dim = int(text_encoder.token_embedding.embedding_dim)
        self.output_dim = self._projection_output_dim(
            text_encoder.text_projection
        )
        if self.prompt_length <= 0 or self.prompt_length >= self.context_length - 1:
            raise ValueError(
                "prompt_length must leave room for BOS, suffix, and appended CLS; "
                f"got {self.prompt_length} for context length {self.context_length}"
            )

        placeholder = " ".join(["x"] * self.prompt_length) + "."
        token_ids = text_encoder.tokenizer([placeholder])
        if not torch.is_tensor(token_ids) or token_ids.ndim != 2 or token_ids.shape[0] != 1:
            shape = tuple(token_ids.shape) if torch.is_tensor(token_ids) else type(token_ids).__name__
            raise ValueError(
                "TITAN tokenizer must return [1, context_length], "
                f"got {shape}"
            )
        if token_ids.shape[1] != self.context_length:
            raise ValueError(
                f"TITAN tokenizer returned {token_ids.shape[1]} tokens; "
                f"expected {self.context_length}"
            )

        pseudo_tokens = token_ids[:, :-1].detach()
        non_padding = int(
            (pseudo_tokens != int(text_encoder.pad_id)).sum().item()
        )
        if non_padding < self.prompt_length + 2:
            raise ValueError(
                "Placeholder prompt was truncated or tokenized into too few "
                f"tokens ({non_padding}) for prompt_length={self.prompt_length}"
            )
        embedding_device = text_encoder.token_embedding.weight.device
        with torch.no_grad():
            base_embedding = text_encoder.token_embedding(
                pseudo_tokens.to(embedding_device)
            ).detach()
        self.register_buffer("pseudo_tokens", pseudo_tokens, persistent=False)
        self.register_buffer(
            "embedding_prefix", base_embedding[:, :1], persistent=False
        )
        self.register_buffer(
            "embedding_suffix",
            base_embedding[:, 1 + self.prompt_length :],
            persistent=False,
        )

        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    @staticmethod
    def _projection_output_dim(projection) -> int:
        if isinstance(projection, nn.Linear):
            return int(projection.out_features)
        if torch.is_tensor(projection) and projection.ndim == 2:
            return int(projection.shape[1])
        raise ValueError("Unsupported TITAN text_projection type")

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def build_prompt_embeddings(
        self, prompt_core: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prompt_core.ndim != 3:
            raise ValueError(
                "prompt_core must have shape [B,L,D], "
                f"got {tuple(prompt_core.shape)}"
            )
        if prompt_core.shape[1:] != (self.prompt_length, self.embedding_dim):
            raise ValueError(
                "prompt_core shape mismatch: expected "
                f"[B,{self.prompt_length},{self.embedding_dim}], "
                f"got {tuple(prompt_core.shape)}"
            )
        batch_size = prompt_core.shape[0]
        prefix = self.embedding_prefix.to(
            device=prompt_core.device, dtype=prompt_core.dtype
        ).expand(batch_size, -1, -1)
        suffix = self.embedding_suffix.to(
            device=prompt_core.device, dtype=prompt_core.dtype
        ).expand(batch_size, -1, -1)
        embeddings = torch.cat((prefix, prompt_core, suffix), dim=1)
        pseudo_tokens = self.pseudo_tokens.to(prompt_core.device).expand(
            batch_size, -1
        )
        return embeddings, pseudo_tokens

    def encode_embeddings(
        self, embeddings: torch.Tensor, pseudo_tokens: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                "embeddings must have shape "
                f"[B,L,{self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        if pseudo_tokens.shape != embeddings.shape[:2]:
            raise ValueError(
                "pseudo_tokens must match the first two embedding dimensions"
            )
        if embeddings.shape[1] + 1 > self.context_length:
            raise ValueError(
                f"Embedding sequence is too long for context {self.context_length}"
            )

        cast_dtype = self.text_encoder.transformer.get_cast_dtype()
        x = embeddings.to(cast_dtype)
        seq_len = x.shape[1] + 1
        cls = self.text_encoder.cls_emb.reshape(1, 1, -1).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat((x, cls.to(device=x.device, dtype=x.dtype)), dim=1)

        cls_mask = self.text_encoder.build_cls_mask(pseudo_tokens, cast_dtype)
        attention_mask = self.text_encoder.attn_mask
        if attention_mask is None:
            attention_mask = cls_mask[:, :seq_len, :seq_len]
        else:
            attention_mask = attention_mask.to(x.device)
            attention_mask = (
                attention_mask[None, :seq_len, :seq_len]
                + cls_mask[:, :seq_len, :seq_len]
            )
        positional = self.text_encoder.positional_embedding[:seq_len].to(
            device=x.device, dtype=cast_dtype
        )
        x = self.text_encoder.transformer(
            x + positional, attn_mask=attention_mask
        )
        pooled = self.text_encoder.ln_final(x[:, -1])
        projection = self.text_encoder.text_projection
        if isinstance(projection, nn.Linear):
            pooled = projection(pooled)
        else:
            pooled = pooled @ projection
        return pooled

    def forward(self, prompt_core: torch.Tensor) -> torch.Tensor:
        embeddings, pseudo_tokens = self.build_prompt_embeddings(prompt_core)
        return self.encode_embeddings(embeddings, pseudo_tokens)


class QPMILVLTitanCore(nn.Module):
    """Queryable prototype MIL with trainable prompts and class residuals."""

    def __init__(
        self,
        text_encoder: nn.Module,
        class_features: torch.Tensor,
        task_num_classes: Sequence[int],
        pool_size: int = 20,
        prompt_length: int = 24,
        match_size: int = 5,
        pooling: str = "max",
        csm_logit_scale: float = 100.0,
        classification_logit_scale: float = 1.0,
        alpha: float = 0.5,
        normalize_eps: float = 1.0e-6,
    ):
        super().__init__()
        self.pool_size = int(pool_size)
        self.prompt_length = int(prompt_length)
        self.match_size = int(match_size)
        self.pooling = str(pooling)
        self.csm_logit_scale = float(csm_logit_scale)
        self.classification_logit_scale = float(classification_logit_scale)
        self.alpha = float(alpha)
        self.normalize_eps = float(normalize_eps)
        self.task_num_classes = [int(value) for value in task_num_classes]

        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.match_size <= 0 or self.match_size > self.pool_size:
            raise ValueError("match_size must be in [1,pool_size]")
        if self.pooling not in {"max", "mean"}:
            raise ValueError("pooling must be 'max' or 'mean'")
        if not self.task_num_classes or any(
            value <= 0 for value in self.task_num_classes
        ):
            raise ValueError("task_num_classes must contain positive integers")
        for name, value in (
            ("csm_logit_scale", self.csm_logit_scale),
            ("classification_logit_scale", self.classification_logit_scale),
            ("alpha", self.alpha),
            ("normalize_eps", self.normalize_eps),
        ):
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite")
        if self.normalize_eps <= 0:
            raise ValueError("normalize_eps must be positive")

        self.prompt_encoder = TitanPromptEncoder(text_encoder, self.prompt_length)
        self.feature_dim = self.prompt_encoder.output_dim
        self.embedding_dim = self.prompt_encoder.embedding_dim
        class_features = torch.as_tensor(class_features).detach().float()
        expected_classes = sum(self.task_num_classes)
        expected_shape = (expected_classes, self.feature_dim)
        if class_features.shape != expected_shape:
            raise ValueError(
                f"class_features must have shape {expected_shape}, "
                f"got {tuple(class_features.shape)}"
            )
        if not torch.isfinite(class_features).all():
            raise ValueError("class_features contains NaN or Inf")
        self.register_buffer(
            "class_features",
            F.normalize(class_features, dim=-1, eps=self.normalize_eps),
        )

        # Keep each pool entry as its own Parameter.  This lets unmatched
        # entries have grad=None so AdamW cannot change them via weight decay.
        self.keys = nn.ParameterList(
            [
                nn.Parameter(0.02 * torch.randn(1, self.feature_dim))
                for _ in range(self.pool_size)
            ]
        )
        self.prompts = nn.ParameterList(
            [
                nn.Parameter(
                    0.02
                    * torch.randn(
                        1, self.prompt_length, self.embedding_dim
                    )
                )
                for _ in range(self.pool_size)
            ]
        )
        self.tunable_vectors = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(count, self.feature_dim))
                for count in self.task_num_classes
            ]
        )
        self.current_task = -1
        self.register_buffer("penalty_table", None, persistent=False)

    @property
    def total_classes(self) -> int:
        return sum(self.task_num_classes)

    @property
    def seen_classes(self) -> int:
        if self.current_task < 0:
            return 0
        return sum(self.task_num_classes[: self.current_task + 1])

    def train(self, mode: bool = True):
        super().train(mode)
        self.prompt_encoder.text_encoder.eval()
        return self

    def begin_task(
        self,
        task_id: int,
        previous_key_frequencies: Optional[Sequence[torch.Tensor]] = None,
    ) -> None:
        task_id = int(task_id)
        if task_id < 0 or task_id >= len(self.task_num_classes):
            raise ValueError(
                f"task_id must be in [0,{len(self.task_num_classes) - 1}], "
                f"got {task_id}"
            )
        frequencies = list(previous_key_frequencies or [])
        if len(frequencies) != task_id:
            raise ValueError(
                f"Expected {task_id} previous key-frequency tensors, "
                f"got {len(frequencies)}"
            )
        if task_id == 0:
            self.penalty_table = None
        else:
            normalized = []
            for index, frequency in enumerate(frequencies):
                frequency = torch.as_tensor(frequency, dtype=torch.float32)
                if frequency.shape != (self.pool_size,):
                    raise ValueError(
                        f"Key frequency {index} must have shape "
                        f"[{self.pool_size}], got {tuple(frequency.shape)}"
                    )
                total = frequency.sum()
                if not torch.isfinite(frequency).all() or total <= 0:
                    raise ValueError(
                        f"Key frequency {index} must be finite with positive total"
                    )
                normalized.append(frequency / total)
            self.penalty_table = torch.stack(normalized).mean(dim=0).to(
                self.class_features.device
            )

        self.current_task = task_id
        for index, vector in enumerate(self.tunable_vectors):
            vector.requires_grad_(index == task_id)
            if index != task_id:
                vector.grad = None
        self.prompt_encoder.text_encoder.requires_grad_(False)
        self.prompt_encoder.text_encoder.eval()

    def _validate_bags(self, bags: Sequence[torch.Tensor]) -> None:
        if not bags:
            raise ValueError("bags must contain at least one WSI")
        for index, bag in enumerate(bags):
            if not torch.is_tensor(bag) or bag.ndim != 2:
                shape = tuple(bag.shape) if torch.is_tensor(bag) else type(bag).__name__
                raise ValueError(f"Bag {index} must have shape [N,D], got {shape}")
            if bag.shape[0] == 0:
                raise ValueError(f"Bag {index} is empty")
            if bag.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Bag {index} has feature dimension {bag.shape[1]}; "
                    f"expected {self.feature_dim}"
                )
            if not torch.isfinite(bag).all():
                raise ValueError(f"Bag {index} contains NaN or Inf")

    def _query_prototype_pool(
        self, bags: Sequence[torch.Tensor], use_penalty: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = []
        for bag in bags:
            bag_float = bag.float()
            query = (
                bag_float.max(dim=0).values
                if self.pooling == "max"
                else bag_float.mean(dim=0)
            )
            queries.append(query)
        queries_tensor = F.normalize(
            torch.stack(queries), dim=-1, eps=self.normalize_eps
        )
        normalized_keys = F.normalize(
            torch.cat(list(self.keys), dim=0).float(),
            dim=-1,
            eps=self.normalize_eps,
        )
        cosine = queries_tensor @ normalized_keys.t()
        if use_penalty:
            if self.current_task <= 0 or self.penalty_table is None:
                raise RuntimeError(
                    "Penalty requested before a later continual task began"
                )
            penalty = self.penalty_table.to(
                device=cosine.device, dtype=cosine.dtype
            )
            candidates = ((1.0 - cosine) * penalty).topk(
                self.match_size, dim=1, largest=False
            ).indices
        else:
            candidates = cosine.topk(
                self.match_size, dim=1, largest=True
            ).indices

        key_ids, counts = torch.unique(
            candidates, sorted=True, return_counts=True
        )
        majority = key_ids[counts.topk(self.match_size).indices]
        indices = majority.unsqueeze(0).expand(len(bags), -1)
        matched_keys = normalized_keys[indices]
        matching_loss = 1.0 - (
            queries_tensor.unsqueeze(1) * matched_keys
        ).sum(dim=-1).mean()
        return indices, matching_loss

    def _aggregate_bags(
        self, bags: Sequence[torch.Tensor], indices: torch.Tensor
    ) -> torch.Tensor:
        merged_prompts = torch.cat(list(self.prompts), dim=0)
        prompt_core = merged_prompts[indices].reshape(
            len(bags) * self.match_size,
            self.prompt_length,
            self.embedding_dim,
        )
        prototypes = self.prompt_encoder(prompt_core)
        prototypes = F.normalize(
            prototypes.float(), dim=-1, eps=self.normalize_eps
        ).reshape(len(bags), self.match_size, self.feature_dim)

        bag_features = []
        for bag, bag_prototypes in zip(bags, prototypes):
            bag_float = bag.float()
            normalized_patches = F.normalize(
                bag_float, dim=-1, eps=self.normalize_eps
            )
            similarities = (
                self.csm_logit_scale
                * normalized_patches
                @ bag_prototypes.transpose(0, 1)
            )
            patch_weights = torch.softmax(similarities, dim=0)
            aggregated = patch_weights.transpose(0, 1) @ bag_float
            bag_features.append(aggregated.mean(dim=0))
        return F.normalize(
            torch.stack(bag_features), dim=-1, eps=self.normalize_eps
        )

    def _enhanced_class_features(self) -> torch.Tensor:
        if self.current_task < 0:
            raise RuntimeError("Call begin_task() before forward()")
        residuals = torch.cat(
            list(self.tunable_vectors[: self.current_task + 1]), dim=0
        )
        enhanced = (
            self.class_features[: self.seen_classes]
            + self.alpha * residuals
        )
        return F.normalize(enhanced.float(), dim=-1, eps=self.normalize_eps)

    @staticmethod
    def _class_similarity_loss(class_features: torch.Tensor) -> torch.Tensor:
        if class_features.shape[0] <= 1:
            return class_features.new_zeros(())
        similarity = class_features @ class_features.t() + 1.0
        off_diagonal = ~torch.eye(
            class_features.shape[0],
            dtype=torch.bool,
            device=class_features.device,
        )
        return similarity[off_diagonal].mean()

    def forward(
        self,
        bags: Sequence[torch.Tensor],
        compute_aux_losses: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        self._validate_bags(bags)
        if self.current_task < 0:
            raise RuntimeError("Call begin_task() before forward()")
        use_penalty = bool(compute_aux_losses and self.current_task > 0)
        indices, matching_loss = self._query_prototype_pool(bags, use_penalty)
        bag_features = self._aggregate_bags(bags, indices)
        class_features = self._enhanced_class_features()
        logits = (
            self.classification_logit_scale
            * bag_features
            @ class_features.t()
        )
        return {
            "logits": logits.float(),
            "key_indices": indices,
            "matching_loss": (
                matching_loss.float() if compute_aux_losses else None
            ),
            "class_similarity_loss": (
                self._class_similarity_loss(class_features).float()
                if compute_aux_losses
                else None
            ),
        }

    def adaptation_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.keys.parameters()
        yield from self.prompts.parameters()
        yield from self.tunable_vectors.parameters()

    def clear_zero_adaptation_gradients(self) -> None:
        """Keep AdamW from decaying pool entries unused by this update."""
        for parameter in self.adaptation_parameters():
            if parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0:
                parameter.grad = None

    def adaptation_state_dict(self) -> dict:
        return {
            "schema_version": ADAPTATION_SCHEMA_VERSION,
            "keys": torch.cat(list(self.keys), dim=0).detach().cpu().clone(),
            "prompts": (
                torch.cat(list(self.prompts), dim=0).detach().cpu().clone()
            ),
            "tunable_vectors": [
                value.detach().cpu().clone() for value in self.tunable_vectors
            ],
            "class_features": self.class_features.detach().cpu().clone(),
            "current_task": int(self.current_task),
        }

    def load_adaptation_state_dict(
        self, state: dict, strict: bool = True
    ) -> None:
        required = {
            "schema_version",
            "keys",
            "prompts",
            "tunable_vectors",
            "class_features",
            "current_task",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"Adaptation checkpoint is missing keys: {sorted(missing)}"
            )
        unexpected = set(state).difference(required)
        if strict and unexpected:
            raise ValueError(
                f"Unexpected adaptation checkpoint keys: {sorted(unexpected)}"
            )
        if int(state["schema_version"]) != ADAPTATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported adaptation schema {state['schema_version']}"
            )
        tensors = {
            "keys": (
                torch.as_tensor(state["keys"]),
                torch.Size((self.pool_size, self.feature_dim)),
            ),
            "prompts": (
                torch.as_tensor(state["prompts"]),
                torch.Size(
                    (
                        self.pool_size,
                        self.prompt_length,
                        self.embedding_dim,
                    )
                ),
            ),
            "class_features": (
                torch.as_tensor(state["class_features"]),
                self.class_features.shape,
            ),
        }
        for name, (value, expected_shape) in tensors.items():
            if value.shape != expected_shape:
                raise ValueError(
                    f"Checkpoint {name} shape mismatch: {tuple(value.shape)}; "
                    f"expected {tuple(expected_shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"Checkpoint {name} contains NaN or Inf")
        tunable = list(state["tunable_vectors"])
        if len(tunable) != len(self.tunable_vectors):
            raise ValueError("Checkpoint tunable-vector task count mismatch")

        with torch.no_grad():
            for index, parameter in enumerate(self.keys):
                parameter.copy_(
                    tensors["keys"][0][index : index + 1].to(parameter)
                )
            for index, parameter in enumerate(self.prompts):
                parameter.copy_(
                    tensors["prompts"][0][index : index + 1].to(parameter)
                )
            self.class_features.copy_(
                tensors["class_features"][0].to(self.class_features)
            )
            for index, (parameter, raw_value) in enumerate(
                zip(self.tunable_vectors, tunable)
            ):
                value = torch.as_tensor(raw_value)
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"Checkpoint tunable vector {index} shape mismatch: "
                        f"{tuple(value.shape)}; expected {tuple(parameter.shape)}"
                    )
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"Checkpoint tunable vector {index} contains NaN or Inf"
                    )
                parameter.copy_(value.to(parameter))
        current_task = int(state["current_task"])
        if current_task < -1 or current_task >= len(self.task_num_classes):
            raise ValueError(
                f"Checkpoint current_task is out of range: {current_task}"
            )
        self.current_task = current_task


class QpmilVl(ContinualModel):
    """ConSlide wrapper around :class:`QPMILVLTitanCore`."""

    NAME = "qpmil_vl"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("titan",)
    REQUIRED_FEATURE_DIM = 768
    REQUIRES_TRAINABLE_BACKBONE = False
    CHECKPOINT_USES_STATE_DICT = False
    CHECKPOINT_INCLUDE_OPTIMIZER = False

    def __init__(
        self,
        core: QPMILVLTitanCore,
        loss,
        args: Namespace,
        transform=None,
    ):
        super().__init__(core, loss, args, transform)
        self.task_order = [str(value) for value in args.task_order]
        self.task_num_classes = [int(value) for value in args.task_num_classes]
        self.total_classes = int(args.num_classes)
        if sum(self.task_num_classes) != self.total_classes:
            raise ValueError(
                "task_num_classes does not sum to args.num_classes"
            )
        if self.task_num_classes != core.task_num_classes:
            raise ValueError(
                "Core and ConSlide task class counts do not match"
            )
        self.model_id = str(
            getattr(args, "backbone_model_id", None) or DEFAULT_TITAN_MODEL_ID
        )
        self.model_revision = str(
            getattr(args, "backbone_revision", None) or DEFAULT_TITAN_REVISION
        )
        self.prompt_hash = prompt_schema_hash(
            self.task_order, self.task_num_classes
        )
        self.completed_key_frequencies: List[torch.Tensor] = []
        self.active_key_frequency = torch.zeros(
            core.pool_size, dtype=torch.long
        )
        self.active_epoch: Optional[int] = None
        self._task_finalized = False

        # A defined first-task state keeps construction and direct smoke forwards
        # deterministic; begin_task(dataset) resets it before training.
        self.net.begin_task(0, [])
        self._reset_optimizer()

    @property
    def current_task(self) -> int:
        return int(self.net.current_task)

    def _reset_optimizer(self) -> None:
        parameters = [
            parameter
            for parameter in self.net.adaptation_parameters()
            if parameter.requires_grad
        ]
        self.opt = build_optimizer(parameters, self.args)

    def begin_task(self, dataset) -> None:
        task_id = int(dataset.current_task) - 1
        if task_id < 0:
            raise ValueError(
                "Dataset must load the current task before QPMIL-VL begin_task"
            )
        if len(self.completed_key_frequencies) != task_id:
            raise ValueError(
                f"Task {task_id} requires {task_id} completed key-frequency "
                f"vectors; found {len(self.completed_key_frequencies)}"
            )
        self.net.begin_task(task_id, self.completed_key_frequencies)
        self.active_key_frequency = torch.zeros(
            self.net.pool_size, dtype=torch.long
        )
        self.active_epoch = None
        self._task_finalized = False
        self._reset_optimizer()

    def begin_epoch(self, task_id: int, epoch: int) -> None:
        if int(task_id) != self.current_task:
            raise ValueError(
                f"begin_epoch task {task_id} does not match active task "
                f"{self.current_task}"
            )
        self.active_key_frequency.zero_()
        self.active_epoch = int(epoch)

    def end_epoch(self, task_id: int, epoch: int) -> dict:
        if int(task_id) != self.current_task or int(epoch) != self.active_epoch:
            raise ValueError("end_epoch does not match the active task/epoch")
        return {
            "key_frequency": self.active_key_frequency.clone(),
            "key_matches": int(self.active_key_frequency.sum().item()),
        }

    @staticmethod
    def _feature_tensor(features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(features):
            raise TypeError("QPMIL-VL features must be a tensor")
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(
                f"QPMIL-VL expects a raw patch bag [N,D], got {tuple(features.shape)}"
            )
        return features

    @staticmethod
    def _split_training_item(item) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(item, "features") and hasattr(item, "labels"):
            return item.features, item.labels
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            raise ValueError(
                "observe_many items must be "
                "(features,coords,patch_size_level0,labels)"
            )
        return item[0], item[3]

    def observe_many(
        self,
        batches,
        task=None,
        ssl: bool = False,
    ) -> Dict[str, float]:
        if ssl:
            raise ValueError("QPMIL-VL does not define an SSL phase")
        if not batches:
            raise ValueError("observe_many requires at least one WSI")
        if task is not None and int(task) != self.current_task:
            raise ValueError(
                f"observe_many task {task} does not match active task "
                f"{self.current_task}"
            )

        bags, label_tensors = [], []
        for item in batches:
            features, labels = self._split_training_item(item)
            bag = self._feature_tensor(features).to(self.device)
            bags.append(bag)
            label_tensors.append(
                torch.as_tensor(labels, dtype=torch.long, device=self.device).reshape(-1)
            )
        targets = torch.cat(label_tensors)
        if targets.numel() != len(bags):
            raise ValueError("Each WSI in observe_many must have exactly one label")
        if targets.min() < 0 or targets.max() >= self.net.seen_classes:
            raise ValueError(
                "QPMIL-VL labels must be global IDs within the seen-class slice"
            )

        self.opt.zero_grad(set_to_none=True)
        output = self.net(bags, compute_aux_losses=True)
        classification_loss = self.loss(output["logits"], targets)
        matching_loss = output["matching_loss"]
        class_similarity_loss = output["class_similarity_loss"]
        total_loss = (
            classification_loss
            + float(self.args.matching_loss_weight) * matching_loss
            + float(self.args.class_similarity_loss_weight)
            * class_similarity_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("QPMIL-VL produced a non-finite loss")
        total_loss.backward()
        trainable = [
            parameter
            for parameter in self.net.adaptation_parameters()
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(
            trainable, float(self.args.max_grad_norm)
        )
        self.net.clear_zero_adaptation_gradients()
        self.opt.step()

        self.active_key_frequency += torch.bincount(
            output["key_indices"].detach().cpu().reshape(-1),
            minlength=self.net.pool_size,
        )
        return {
            "loss": float(total_loss.detach().item()),
            "classification_loss": float(classification_loss.detach().item()),
            "matching_loss": float(matching_loss.detach().item()),
            "class_similarity_loss": float(
                class_similarity_loss.detach().item()
            ),
        }

    def observe(
        self,
        features,
        coords,
        patch_size,
        labels,
        task=None,
        ssl: bool = False,
    ) -> Dict[str, float]:
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def forward(
        self, x, coords=None, patch_size_level0=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del patch_size_level0
        if isinstance(x, (tuple, list)):
            if not x:
                raise ValueError("QPMIL-VL input cannot be empty")
            features = x[0]
        else:
            features = x
        del coords
        bag = self._feature_tensor(features).to(self.device)
        output = self.net([bag], compute_aux_losses=False)
        seen_logits = output["logits"]
        logits = seen_logits.new_full(
            (seen_logits.shape[0], self.total_classes), -float("inf")
        )
        logits[:, : self.net.seen_classes] = seen_logits
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        attention = torch.full(
            (1, bag.shape[0]),
            1.0 / bag.shape[0],
            device=bag.device,
            dtype=bag.dtype,
        )
        auxiliary = seen_logits.sum() * 0.0
        return logits, probabilities, predictions, attention, auxiliary

    def end_task(self, dataset) -> None:
        del dataset
        if self._task_finalized:
            raise RuntimeError("QPMIL-VL task was already finalized")
        if len(self.completed_key_frequencies) != self.current_task:
            raise RuntimeError("Completed key-frequency history is inconsistent")
        if self.active_key_frequency.sum() <= 0:
            raise RuntimeError(
                "Cannot finalize QPMIL-VL without key matches from the selected epoch"
            )
        self.completed_key_frequencies.append(
            self.active_key_frequency.clone().cpu()
        )
        self._task_finalized = True

    def get_checkpoint_state(self) -> dict:
        """Return adaptation-only state; the frozen TITAN tower is excluded."""
        return {
            "schema_version": METHOD_STATE_SCHEMA_VERSION,
            "method": self.NAME,
            "state_type": "adaptation_only",
            "titan_model_id": self.model_id,
            "titan_revision": self.model_revision,
            "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "total_classes": self.total_classes,
            "prompt_schema_hash": self.prompt_hash,
            "pool_size": self.net.pool_size,
            "prompt_length": self.net.prompt_length,
            "match_size": self.net.match_size,
            "adaptation_state": self.net.adaptation_state_dict(),
            "completed_key_frequencies": [
                value.clone().cpu()
                for value in self.completed_key_frequencies
            ],
            "active_key_frequency": self.active_key_frequency.clone().cpu(),
            "active_epoch": self.active_epoch,
            "task_finalized": self._task_finalized,
        }

    def load_checkpoint_state(self, state: dict, strict: bool = True) -> None:
        required = {
            "schema_version",
            "method",
            "state_type",
            "titan_model_id",
            "titan_revision",
            "task_order",
            "task_num_classes",
            "total_classes",
            "prompt_schema_hash",
            "pool_size",
            "prompt_length",
            "match_size",
            "adaptation_state",
            "completed_key_frequencies",
            "active_key_frequency",
            "active_epoch",
            "task_finalized",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"QPMIL-VL method state is missing keys: {sorted(missing)}"
            )
        unexpected = set(state).difference(required)
        if strict and unexpected:
            raise ValueError(
                f"Unexpected QPMIL-VL method state keys: {sorted(unexpected)}"
            )
        expected_metadata = {
            "schema_version": METHOD_STATE_SCHEMA_VERSION,
            "method": self.NAME,
            "state_type": "adaptation_only",
            "titan_model_id": self.model_id,
            "titan_revision": self.model_revision,
            "task_order": self.task_order,
            "task_num_classes": self.task_num_classes,
            "total_classes": self.total_classes,
            "prompt_schema_hash": self.prompt_hash,
            "pool_size": self.net.pool_size,
            "prompt_length": self.net.prompt_length,
            "match_size": self.net.match_size,
        }
        for key, expected in expected_metadata.items():
            actual = state[key]
            if isinstance(expected, list):
                actual = list(actual)
            if actual != expected:
                raise ValueError(
                    f"QPMIL-VL checkpoint mismatch for {key}: "
                    f"saved={actual!r}, expected={expected!r}"
                )

        frequencies = []
        for index, raw_frequency in enumerate(
            state["completed_key_frequencies"]
        ):
            frequency = torch.as_tensor(raw_frequency, dtype=torch.long).cpu()
            if frequency.shape != (self.net.pool_size,):
                raise ValueError(
                    f"Completed key frequency {index} has shape "
                    f"{tuple(frequency.shape)}"
                )
            if not torch.isfinite(frequency.float()).all() or frequency.sum() <= 0:
                raise ValueError(
                    f"Completed key frequency {index} must have a positive total"
                )
            frequencies.append(frequency)
        active = torch.as_tensor(
            state["active_key_frequency"], dtype=torch.long
        ).cpu()
        if active.shape != (self.net.pool_size,):
            raise ValueError(
                f"Active key frequency has shape {tuple(active.shape)}"
            )
        if (active < 0).any():
            raise ValueError("Active key frequency cannot contain negatives")

        self.net.load_adaptation_state_dict(
            state["adaptation_state"], strict=strict
        )
        task_id = self.net.current_task
        finalized = bool(state["task_finalized"])
        expected_frequency_count = task_id + int(finalized)
        if len(frequencies) != expected_frequency_count:
            raise ValueError(
                "Completed key-frequency count is incompatible with "
                "adaptation current_task/task_finalized"
            )
        # Recreate non-persistent penalty and gradient-scope state.  A finalized
        # task excludes its own frequency because its penalty was trained only
        # against tasks that preceded it.
        self.net.begin_task(task_id, frequencies[:task_id])
        self.completed_key_frequencies = frequencies
        self.active_key_frequency = active
        self.active_epoch = (
            None if state["active_epoch"] is None else int(state["active_epoch"])
        )
        self._task_finalized = finalized
        self._reset_optimizer()


def build_class_features(
    titan_model,
    class_prompts: Sequence[Sequence[str]],
    device: torch.device,
) -> torch.Tensor:
    """Create normalized TITAN zero-shot features once for every class."""
    with torch.inference_mode():
        classifier = titan_model.zero_shot_classifier(
            class_prompts, list(TEMPLATES), device=str(device)
        )
    if not torch.is_tensor(classifier) or classifier.ndim != 2:
        shape = tuple(classifier.shape) if torch.is_tensor(classifier) else type(classifier).__name__
        raise ValueError(
            "TITAN zero_shot_classifier must return a rank-2 tensor, "
            f"got {shape}"
        )
    expected_classes = len(class_prompts)
    if classifier.shape[1] == expected_classes:
        class_features = classifier.t()
    elif classifier.shape[0] == expected_classes:
        class_features = classifier
    else:
        raise ValueError(
            "TITAN zero-shot classifier does not contain the expected "
            f"{expected_classes} classes: {tuple(classifier.shape)}"
        )
    return F.normalize(class_features.float(), dim=-1, eps=1.0e-6)


def _load_titan(args: Namespace, device: torch.device):
    """Lazily resolve and load the pinned full model needed for its text tower."""
    import os

    from backbone.pretrained_mil import _resolve_snapshot
    from transformers import AutoModel

    model_id = str(
        getattr(args, "backbone_model_id", None) or DEFAULT_TITAN_MODEL_ID
    )
    revision = str(
        getattr(args, "backbone_revision", None) or DEFAULT_TITAN_REVISION
    )
    # Resolve first so missing offline weights fail with the same actionable
    # message as the native backbone adapters.  Loading by repository ID (with
    # the pinned revision and local_files_only policy) is intentional: the
    # transformers dynamic-module loader does not preserve all of TITAN's
    # relative remote-code imports when given its symlinked snapshot path.
    _resolve_snapshot(
        model_id,
        revision,
        getattr(args, "backbone_cache_dir", None),
        bool(getattr(args, "backbone_allow_download", False)),
    )
    allow_download = bool(getattr(args, "backbone_allow_download", False))
    load_kwargs = {
        "revision": revision,
        "trust_remote_code": True,
        "local_files_only": not allow_download,
    }
    cache_dir = getattr(args, "backbone_cache_dir", None)
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    if allow_download and os.environ.get("HF_TOKEN"):
        load_kwargs["token"] = os.environ["HF_TOKEN"]
    model = AutoModel.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.to(device)
    model.eval()
    return model


def build_model_from_components(
    args: Namespace,
    loss,
    transform,
    text_encoder: nn.Module,
    class_features: torch.Tensor,
) -> QpmilVl:
    """Dependency-injected builder used by unit tests and cached-model tooling."""
    core = QPMILVLTitanCore(
        text_encoder=text_encoder,
        class_features=class_features,
        task_num_classes=args.task_num_classes,
        pool_size=int(args.pool_size),
        prompt_length=int(args.prompt_length),
        match_size=int(args.match_size),
        pooling=str(args.pooling),
        csm_logit_scale=float(args.csm_logit_scale),
        classification_logit_scale=float(args.classification_logit_scale),
        alpha=float(args.alpha),
    )
    if core.feature_dim != int(args.feature_dim):
        raise ValueError(
            f"TITAN text/patch space has dimension {core.feature_dim}; "
            f"configured features have dimension {args.feature_dim}"
        )
    return QpmilVl(core, loss, args, transform)


def build_model(args: Namespace, loss, transform) -> QpmilVl:
    """Special model-factory hook; no slide vision backbone is constructed."""
    if str(args.backbone).lower() != "titan":
        raise ValueError("QPMIL-VL supports only --backbone titan")
    if int(args.feature_dim) != 768:
        raise ValueError("QPMIL-VL requires raw 768-D TITAN-compatible patch bags")
    task_order = [str(value) for value in args.task_order]
    task_num_classes = [int(value) for value in args.task_num_classes]
    class_prompts = resolve_class_prompts(task_order, task_num_classes)
    if len(class_prompts) != int(args.num_classes):
        raise ValueError(
            "Prompt registry class count does not match args.num_classes"
        )

    device = get_device()
    titan_model = _load_titan(args, device)
    class_features = build_class_features(
        titan_model, class_prompts, device
    )
    if not hasattr(titan_model, "text_encoder"):
        raise ValueError("Pinned TITAN model does not expose text_encoder")
    text_encoder = titan_model.text_encoder
    model = build_model_from_components(
        args, loss, transform, text_encoder, class_features
    )
    # QpmilVl retains text_encoder as a registered child; the unreferenced vision
    # tower and full-model wrapper can now be reclaimed.
    del titan_model
    return model


# Backward-friendly alias for tests or research code that used the standalone
# Benchmark implementation's class name.
QPMILVLTitan = QPMILVLTitanCore
