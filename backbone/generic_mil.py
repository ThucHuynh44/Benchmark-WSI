"""Backbone-agnostic MIL interface for pre-extracted WSI features.

The training code in this repository expects a common five-item output:
``(logits, probabilities, predictions, instance_attention, auxiliary_loss)``.
This module provides a small default MIL backbone and an adapter for custom
backbones so datasets are not tied to HIT, CLAM, TransMIL, or DSMIL.
"""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _unpack_input(features, coords=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(features, (list, tuple)):
        if len(features) == 0:
            raise ValueError("The MIL input list cannot be empty.")
        coords = features[1] if len(features) > 1 else coords
        features = features[0]
    if not torch.is_tensor(features):
        raise TypeError(f"features must be a tensor, got {type(features)!r}")
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if features.ndim != 2:
        raise ValueError(
            "A WSI bag must have shape [num_patches, feature_dim], "
            f"got {tuple(features.shape)}"
        )
    return features.float(), coords


class GenericMILBackbone(nn.Module):
    """Configurable gated-attention MIL model for arbitrary feature vectors."""

    supports_ssl = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 384,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        attention_dim = max(self.hidden_dim // 2, 1)
        self.attention_v = nn.Linear(self.hidden_dim, attention_dim)
        self.attention_u = nn.Linear(self.hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.feature_encoder(features)
        attention_logits = self.attention_w(
            torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        ).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=0)
        slide_embedding = torch.sum(attention.unsqueeze(-1) * encoded, dim=0, keepdim=True)
        return slide_embedding, attention.unsqueeze(0)

    def forward(self, features, coords=None, returnt: str = "out", **_) -> Any:
        features, _ = _unpack_input(features, coords)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Backbone expects feature_dim={self.input_dim}, "
                f"but the HDF5 bag has feature_dim={features.shape[-1]}. "
                "Set --feature_dim to the HDF5 feature dimension."
            )
        embedding, attention = self.forward_features(features)
        if returnt == "features":
            return embedding
        logits = self.classifier(embedding)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        auxiliary_loss = logits.sum() * 0.0
        return logits, probabilities, predictions, attention, auxiliary_loss

    def get_params(self) -> torch.Tensor:
        return torch.cat([parameter.view(-1) for parameter in self.parameters()])

    def get_grads(self) -> torch.Tensor:
        return torch.cat([
            parameter.grad.view(-1) if parameter.grad is not None else torch.zeros_like(parameter).view(-1)
            for parameter in self.parameters()
        ])


class BackboneAdapter(nn.Module):
    """Normalize a custom MIL backbone to ConSlide's output contract.

    A custom backbone should accept ``forward(features, coords=None)`` and may
    return a logits tensor, a dictionary containing ``logits``, or an existing
    ConSlide-style tuple.
    """

    supports_ssl = False

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        parameters = list(inspect.signature(model.forward).parameters.values())
        positional = [
            parameter for parameter in parameters
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self._accepts_coords = (
            len(positional) >= 2
            or any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        )
        self._accepts_patch_size = (
            len(positional) >= 3
            or any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        )

    def forward(self, features, coords=None, patch_size_level0=None, returnt: str = "out", **kwargs):
        if isinstance(features, (list, tuple)) and len(features) > 2:
            patch_size_level0 = features[2]
        features, coords = _unpack_input(features, coords)
        if self._accepts_patch_size:
            output = self.model(features, coords, patch_size_level0, **kwargs)
        elif self._accepts_coords:
            output = self.model(features, coords, **kwargs)
        else:
            output = self.model(features, **kwargs)

        if returnt == "features":
            if isinstance(output, dict) and "features" in output:
                return output["features"]
            if hasattr(self.model, "forward_features"):
                return self.model.forward_features(features)
            raise ValueError("The custom backbone does not expose slide-level features.")

        attention = None
        auxiliary_loss = None
        if torch.is_tensor(output):
            logits = output
        elif isinstance(output, dict):
            logits = output["logits"]
            attention = output.get("attention")
            auxiliary_loss = output.get("auxiliary_loss")
        elif isinstance(output, (tuple, list)):
            if len(output) >= 5:
                return tuple(output[:5])
            logits = output[0]
            attention = output[1] if len(output) > 1 else None
            auxiliary_loss = output[2] if len(output) > 2 else None
        else:
            raise TypeError(f"Unsupported custom backbone output: {type(output)!r}")

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        elif logits.ndim == 2 and logits.shape[0] == features.shape[0] and logits.shape[0] > 1:
            # A patch-level classifier is still usable through mean MIL pooling.
            logits = logits.mean(dim=0, keepdim=True)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                "A custom backbone must produce one logits vector per WSI bag; "
                f"got {tuple(logits.shape)}"
            )
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        if attention is None:
            attention = torch.full(
                (1, features.shape[0]),
                1.0 / features.shape[0],
                dtype=features.dtype,
                device=features.device,
            )
        if auxiliary_loss is None:
            auxiliary_loss = logits.sum() * 0.0
        return logits, probabilities, predictions, attention, auxiliary_loss

    def get_params(self) -> torch.Tensor:
        return torch.cat([parameter.view(-1) for parameter in self.parameters()])

    def get_grads(self) -> torch.Tensor:
        return torch.cat([
            parameter.grad.view(-1) if parameter.grad is not None else torch.zeros_like(parameter).view(-1)
            for parameter in self.parameters()
        ])


def _construct_custom_backbone(spec: str, kwargs: Dict[str, Any]) -> nn.Module:
    if ":" not in spec:
        raise ValueError(
            f"Unknown backbone {spec!r}. Use 'generic_mil' or '<python.module>:<ClassName>'."
        )
    module_name, class_name = spec.split(":", 1)
    backbone_class = getattr(importlib.import_module(module_name), class_name)
    signature = inspect.signature(backbone_class)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    }
    return BackboneAdapter(backbone_class(**accepted))


def build_mil_backbone(args, num_classes: int) -> nn.Module:
    """Build the configured backbone without coupling it to the dataset class."""
    name = getattr(args, "backbone", "generic_mil")
    kwargs: Dict[str, Any] = {
        "input_dim": getattr(args, "feature_dim", 768),
        "num_classes": num_classes,
        "hidden_dim": getattr(args, "backbone_hidden_dim", 384),
        "dropout": getattr(args, "backbone_dropout", 0.0),
    }
    raw_kwargs = getattr(args, "backbone_kwargs", None)
    if raw_kwargs:
        kwargs.update(json.loads(raw_kwargs))

    if name == "generic_mil":
        return GenericMILBackbone(**kwargs)
    if name in {"titan", "feather"}:
        from backbone.pretrained_mil import build_pretrained_backbone

        return build_pretrained_backbone(args, num_classes)
    return _construct_custom_backbone(name, kwargs)
