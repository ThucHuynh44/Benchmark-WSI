"""Native adapters for cached TITAN and FEATHER slide encoders."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


TITAN_MODEL_ID = "MahmoodLab/TITAN"
TITAN_REVISION = "dac6773d9961cfc75503440676ff157a2c6e8d2e"
FEATHER_MODEL_ID = "mahmoodlab/abmil.base.conch_v15.pc108-24k"
FEATHER_REVISION = "423c894c738294e8b0ac38938108180a2c21dd43"


def _resolve_snapshot(
    model_id: str,
    revision: str,
    cache_dir: Optional[str],
    allow_download: bool,
) -> str:
    """Resolve a pinned snapshot, remaining offline unless explicitly allowed."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=not allow_download,
            token=os.environ.get("HF_TOKEN") if allow_download else None,
        )
    except Exception as error:
        location = str(Path(cache_dir).expanduser()) if cache_dir else "the Hugging Face default cache"
        action = (
            "Check HF_TOKEN/network access because --backbone_allow_download was enabled."
            if allow_download
            else "Populate the cache first or explicitly pass --backbone_allow_download."
        )
        raise RuntimeError(
            f"Could not resolve pinned backbone {model_id}@{revision} from {location}. {action}"
        ) from error


def _load_auto_model(snapshot_path: str, **kwargs):
    from transformers import AutoModel

    # Loading remote-code repositories by their local snapshot path also avoids
    # invalid dynamic-module names for FEATHER's dotted repository ID.
    return AutoModel.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        **kwargs,
    )


def _load_titan_vision(snapshot_path: str, revision: str) -> nn.Module:
    """Instantiate only TITAN's vision tower and load only its checkpoint keys.

    Constructing the full remote model initializes its text tokenizer, which can
    attempt a network request even when ``local_files_only`` is set on AutoModel.
    Loading the isolated tower also keeps the text encoder out of memory/training.
    """
    snapshot = Path(snapshot_path)
    package_name = f"_conslide_titan_{revision.replace('-', '_')}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(snapshot)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    configuration = importlib.import_module(f"{package_name}.configuration_titan")
    vision_module = importlib.import_module(f"{package_name}.vision_transformer")
    with (snapshot / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    vision_config = configuration.TitanVisionConfig(**config["vision_config"])
    vision_encoder = vision_module.build_vision_tower(vision_config)

    from safetensors import safe_open

    state_dict = {}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("vision_encoder."):
                state_dict[key.removeprefix("vision_encoder.")] = handle.get_tensor(key)
    missing, unexpected = vision_encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Pinned TITAN vision weights are incompatible: missing={missing}, unexpected={unexpected}"
        )
    return vision_encoder


def _unpack_bag(features, coords=None, patch_size_level0=None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(features, (list, tuple)):
        if not features:
            raise ValueError("The MIL input cannot be empty")
        coords = features[1] if len(features) > 1 else coords
        patch_size_level0 = features[2] if len(features) > 2 else patch_size_level0
        features = features[0]
    if not torch.is_tensor(features) or not torch.is_tensor(coords):
        raise TypeError("TITAN/FEATHER inputs require tensor features and coordinates")
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    if features.ndim != 2 or features.shape[-1] != 768:
        raise ValueError(f"Expected features [N,768], got {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("TITAN/FEATHER inputs require at least one patch")
    if not torch.isfinite(features).all():
        raise ValueError("TITAN/FEATHER features contain NaN or Inf")
    if coords.ndim != 2 or coords.shape != (features.shape[0], 2):
        raise ValueError(f"Expected coords [N,2] matching features, got {tuple(coords.shape)}")
    if torch.is_tensor(patch_size_level0):
        if patch_size_level0.numel() != 1:
            raise ValueError("patch_size_level0 must be scalar")
        patch_size_level0 = int(patch_size_level0.detach().cpu().item())
    if patch_size_level0 is None:
        raise ValueError("TITAN/FEATHER input is missing patch_size_level0 metadata")
    patch_size_level0 = int(patch_size_level0)
    if patch_size_level0 <= 0:
        raise ValueError(f"patch_size_level0 must be positive, got {patch_size_level0}")
    return features.float(), coords.long(), patch_size_level0


class _AdapterBase(nn.Module):
    supports_ssl = False

    def _finish(self, logits, attention, features):
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probabilities = F.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        if attention is None:
            attention = torch.full(
                (1, features.shape[0]),
                1.0 / features.shape[0],
                device=features.device,
                dtype=features.dtype,
            )
        while attention.ndim > 2 and attention.shape[0] == 1:
            attention = attention.squeeze(0)
        if attention.ndim == 1:
            attention = attention.unsqueeze(0)
        return logits, probabilities, predictions, attention, logits.sum() * 0.0

    def _embedding_output(self, logits, embedding, attention, features):
        """Return the shared representation contract used by CL methods.

        ``forward()`` intentionally keeps the original five-value ConSlide
        output.  New methods use this dictionary to avoid running the slide
        encoder twice merely to obtain both logits and embeddings.
        """
        finished = self._finish(logits, attention, features)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(
                "A pretrained MIL backbone must return one slide embedding; "
                f"got {tuple(embedding.shape)}"
            )
        return {
            "logits": finished[0],
            "embedding": embedding,
            "attention": finished[3],
            "auxiliary_loss": finished[4],
        }

    def get_params(self) -> torch.Tensor:
        return torch.cat([parameter.view(-1) for parameter in self.parameters()])

    def get_grads(self) -> torch.Tensor:
        return torch.cat([
            parameter.grad.view(-1) if parameter.grad is not None else torch.zeros_like(parameter).view(-1)
            for parameter in self.parameters()
        ])


class TitanMILBackbone(_AdapterBase):
    """TITAN vision encoder with a stream-wide classification head."""

    def __init__(self, vision_encoder: nn.Module, num_classes: int, freeze: bool = False):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.classifier = nn.Linear(768, int(num_classes))
        if freeze:
            self.vision_encoder.requires_grad_(False)

    def _embedding(self, features, coords, patch_size_level0):
        output = self.vision_encoder(
            features,
            coords,
            patch_size_level0,
            no_proj=True,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            output = output.get("features", output.get("embedding"))
        if output is None:
            raise ValueError("TITAN vision encoder did not return a slide embedding")
        if output.ndim == 1:
            output = output.unsqueeze(0)
        return output

    def forward(self, features, coords=None, patch_size_level0=None, returnt="out", **_):
        features, coords, patch_size_level0 = _unpack_bag(features, coords, patch_size_level0)
        embedding = self._embedding(features, coords, patch_size_level0)
        if returnt == "features":
            return embedding
        logits = self.classifier(embedding)
        return self._finish(logits, None, features)

    def forward_with_embedding(self, features, coords=None, patch_size_level0=None):
        features, coords, patch_size_level0 = _unpack_bag(
            features, coords, patch_size_level0
        )
        embedding = self._embedding(features, coords, patch_size_level0)
        logits = self.classifier(embedding)
        return self._embedding_output(logits, embedding, None, features)

    def get_classifier(self) -> nn.Linear:
        return self.classifier


class FeatherMILBackbone(_AdapterBase):
    """Pretrained FEATHER ABMIL model normalized to ConSlide's contract."""

    def __init__(self, model: nn.Module, freeze: bool = False):
        super().__init__()
        self.model = model
        if freeze:
            self.model.requires_grad_(False)
            classifier = self._classifier()
            classifier.requires_grad_(True)

    def _classifier(self) -> nn.Module:
        candidates = [self.model, getattr(self.model, "model", None)]
        for candidate in candidates:
            classifier = getattr(candidate, "classifier", None) if candidate is not None else None
            if isinstance(classifier, nn.Module):
                return classifier
        raise AttributeError("FEATHER remote model does not expose its classifier")

    def _run(self, features):
        output = self.model(
            features.unsqueeze(0),
            return_attention=True,
            return_slide_feats=True,
        )
        if isinstance(output, (tuple, list)) and len(output) == 2:
            results, log = output
        elif isinstance(output, dict):
            results = output.get("results", output)
            log = output.get("log", {})
        else:
            raise TypeError(f"Unexpected FEATHER output type: {type(output)!r}")
        if not isinstance(results, dict) or not isinstance(log, dict):
            raise TypeError("FEATHER results/log output must contain dictionaries")
        logits = results.get("logits")
        attention = results.get("attention", log.get("attention"))
        slide_features = results.get("slide_feats", log.get("slide_feats"))
        if logits is None:
            raise ValueError("FEATHER output is missing logits")
        return logits, attention, slide_features

    def forward(self, features, coords=None, patch_size_level0=None, returnt="out", **_):
        features, _, _ = _unpack_bag(features, coords, patch_size_level0)
        logits, attention, slide_features = self._run(features)
        if returnt == "features":
            if slide_features is None:
                raise ValueError("FEATHER output is missing slide_feats")
            return slide_features
        return self._finish(logits, attention, features)

    def forward_with_embedding(self, features, coords=None, patch_size_level0=None):
        features, _, _ = _unpack_bag(features, coords, patch_size_level0)
        logits, attention, embedding = self._run(features)
        if embedding is None:
            raise ValueError("FEATHER output is missing slide_feats")
        return self._embedding_output(logits, embedding, attention, features)

    def get_classifier(self) -> nn.Linear:
        classifier = self._classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError(
                "FEATHER classifier accessor must expose nn.Linear, "
                f"got {type(classifier)!r}"
            )
        return classifier


def build_pretrained_backbone(args, num_classes: int) -> nn.Module:
    name = str(args.backbone).lower()
    defaults = {
        "titan": (TITAN_MODEL_ID, TITAN_REVISION),
        "feather": (FEATHER_MODEL_ID, FEATHER_REVISION),
    }
    model_id, revision = defaults[name]
    model_id = getattr(args, "backbone_model_id", None) or model_id
    revision = getattr(args, "backbone_revision", None) or revision
    snapshot = _resolve_snapshot(
        model_id,
        revision,
        getattr(args, "backbone_cache_dir", None),
        bool(getattr(args, "backbone_allow_download", False)),
    )
    freeze = bool(getattr(args, "backbone_freeze", False))
    if name == "titan":
        vision_encoder = _load_titan_vision(snapshot, revision)
        return TitanMILBackbone(vision_encoder, num_classes, freeze=freeze)
    remote_model = _load_auto_model(
        snapshot,
        num_classes=int(num_classes),
        ignore_mismatched_sizes=True,
    )
    return FeatherMILBackbone(remote_model, freeze=freeze)
