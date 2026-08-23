"""Native adapters for cached TITAN, FEATHER, and GigaPath slide encoders."""

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
GIGAPATH_MODEL_ID = "prov-gigapath/prov-gigapath"
GIGAPATH_REVISION = "64f9e26c15019f2d4f6d9113c6822f88bb16b01b"
GIGAPATH_SOURCE_REVISION = "55431e04ff853a08f752a0ba42e6e9c48e60c776"
GIGAPATH_ARCHITECTURE = "gigapath_slide_enc12l768d"
GIGAPATH_INPUT_DIM = 1536
GIGAPATH_EMBED_DIM = 768
GIGAPATH_SLIDE_NGRIDS = 1000
GIGAPATH_CHECKPOINT_FILENAME = "slide_encoder.pth"

# These sets are the exact accepted state-dict deviations for the pinned
# encoder-only release: none.  They are deliberately not a compatibility list
# for raw MAE pretraining checkpoints.  A future release with a different
# contract must be reviewed and pinned explicitly.
GIGAPATH_EXPECTED_MISSING_KEYS = frozenset()
GIGAPATH_EXPECTED_UNEXPECTED_KEYS = frozenset()


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


def _resolve_hf_file(
    model_id: str,
    revision: str,
    filename: str,
    cache_dir: Optional[str],
    allow_download: bool,
) -> str:
    """Resolve one pinned Hub file without downloading unrelated artifacts."""
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=model_id,
            revision=revision,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=not allow_download,
            token=os.environ.get("HF_TOKEN") if allow_download else None,
        )
    except Exception as error:
        location = str(Path(cache_dir).expanduser()) if cache_dir else "the Hugging Face default cache"
        action = (
            "Check gated-model access, HF_TOKEN, and network connectivity because "
            "--backbone_allow_download was enabled."
            if allow_download
            else "Populate the cache first or explicitly pass --backbone_allow_download."
        )
        raise RuntimeError(
            f"Could not resolve pinned backbone file "
            f"{model_id}@{revision}/{filename} from {location}. {action}"
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


def _unpack_bag(
    features,
    coords=None,
    patch_size_level0=None,
    *,
    feature_dim: int = 768,
    backbone_name: str = "TITAN/FEATHER",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(features, (list, tuple)):
        if not features:
            raise ValueError("The MIL input cannot be empty")
        coords = features[1] if len(features) > 1 else coords
        patch_size_level0 = features[2] if len(features) > 2 else patch_size_level0
        features = features[0]
    if not torch.is_tensor(features) or not torch.is_tensor(coords):
        raise TypeError(f"{backbone_name} inputs require tensor features and coordinates")
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    if features.ndim != 2 or features.shape[-1] != int(feature_dim):
        raise ValueError(
            f"Expected {backbone_name} features [N,{int(feature_dim)}], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError(f"{backbone_name} inputs require at least one patch")
    if not torch.isfinite(features).all():
        raise ValueError(f"{backbone_name} features contain NaN or Inf")
    if coords.ndim != 2 or coords.shape != (features.shape[0], 2):
        raise ValueError(f"Expected coords [N,2] matching features, got {tuple(coords.shape)}")
    if torch.is_tensor(patch_size_level0):
        if patch_size_level0.numel() != 1:
            raise ValueError("patch_size_level0 must be scalar")
        patch_size_level0 = int(patch_size_level0.detach().cpu().item())
    if patch_size_level0 is None:
        raise ValueError(f"{backbone_name} input is missing patch_size_level0 metadata")
    patch_size_level0 = int(patch_size_level0)
    if patch_size_level0 <= 0:
        raise ValueError(f"patch_size_level0 must be positive, got {patch_size_level0}")
    return features.float(), coords.long(), patch_size_level0


class _AdapterBase(nn.Module):
    supports_ssl = False
    has_genuine_patch_attention = False

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

    has_genuine_patch_attention = False

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

    has_genuine_patch_attention = True

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
        if attention is None:
            raise ValueError(
                "FEATHER output is missing the requested genuine patch attention"
            )
        if not torch.is_tensor(attention) or not torch.isfinite(attention).all():
            raise ValueError("FEATHER attention logits must be a finite tensor")
        # The pinned FEATHER implementation exposes A_base (pre-softmax
        # attention logits) even though its pooling path uses softmax(A_base).
        # Publish the actual pooling weights through the shared MIL contract.
        attention = F.softmax(attention.float(), dim=-1)
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


class GigaPathMILBackbone(_AdapterBase):
    """Prov-GigaPath LongNet slide encoder with a global classifier."""

    has_genuine_patch_attention = False

    def __init__(self, slide_encoder: nn.Module, num_classes: int, freeze: bool = False):
        super().__init__()
        self.slide_encoder = slide_encoder
        self.classifier = nn.Linear(GIGAPATH_EMBED_DIM, int(num_classes))
        if freeze:
            self.slide_encoder.requires_grad_(False)

    def _embedding(self, features, coords, patch_size_level0):
        # Do not pass the WSI patch size to the constructor: upstream uses its
        # constructor tile_size while choosing LongNet segment lengths.  Only
        # positional quantization is bag-specific.
        self.slide_encoder.tile_size = int(patch_size_level0)
        outputs = self.slide_encoder(
            features.unsqueeze(0),
            coords.unsqueeze(0),
            all_layer_embed=False,
        )
        if not isinstance(outputs, (tuple, list)) or not outputs:
            raise TypeError(
                "GigaPath slide encoder must return a non-empty outcomes list"
            )
        embedding = outputs[0]
        if not torch.is_tensor(embedding):
            raise TypeError("GigaPath outputs[0] must be a tensor")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        expected = (1, GIGAPATH_EMBED_DIM)
        if tuple(embedding.shape) != expected:
            raise ValueError(
                f"GigaPath outputs[0] must have shape {expected}, "
                f"got {tuple(embedding.shape)}"
            )
        if not torch.isfinite(embedding).all():
            raise ValueError("GigaPath slide embedding contains NaN or Inf")
        return embedding.float()

    def forward(self, features, coords=None, patch_size_level0=None, returnt="out", **_):
        features, coords, patch_size_level0 = _unpack_bag(
            features,
            coords,
            patch_size_level0,
            feature_dim=GIGAPATH_INPUT_DIM,
            backbone_name="GigaPath",
        )
        embedding = self._embedding(features, coords, patch_size_level0)
        if returnt == "features":
            return embedding
        logits = self.classifier(embedding)
        return self._finish(logits, None, features)

    def forward_with_embedding(self, features, coords=None, patch_size_level0=None):
        features, coords, patch_size_level0 = _unpack_bag(
            features,
            coords,
            patch_size_level0,
            feature_dim=GIGAPATH_INPUT_DIM,
            backbone_name="GigaPath",
        )
        embedding = self._embedding(features, coords, patch_size_level0)
        logits = self.classifier(embedding)
        return self._embedding_output(logits, embedding, None, features)

    def get_classifier(self) -> nn.Linear:
        return self.classifier


def _load_gigapath_slide(checkpoint_path: str) -> nn.Module:
    """Instantiate the pinned release architecture and validate its weights."""
    try:
        # Instantiate the pinned architecture directly. Loading remains
        # explicit below so the upstream helper cannot download files.
        from gigapath.slide_encoder import gigapath_slide_enc12l768d
    except Exception as error:
        raise RuntimeError(
            "Prov-GigaPath runtime dependencies are unavailable. Use the "
            "pinned benchmark_gigapath environment."
        ) from error

    slide_encoder = gigapath_slide_enc12l768d(
        in_chans=GIGAPATH_INPUT_DIM,
        global_pool=True,
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "The pinned GigaPath integration requires torch.load(weights_only=True)"
        ) from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("GigaPath slide_encoder.pth must contain a 'model' state dictionary")
    state_dict = checkpoint["model"]
    projection = state_dict.get("patch_embed.proj.weight")
    expected_projection_shape = (GIGAPATH_EMBED_DIM, GIGAPATH_INPUT_DIM)
    if not torch.is_tensor(projection) or tuple(projection.shape) != expected_projection_shape:
        shape = tuple(projection.shape) if torch.is_tensor(projection) else None
        raise ValueError(
            "Pinned GigaPath patch projection mismatch: "
            f"expected {expected_projection_shape}, got {shape}"
        )
    invalid = [
        key for key, value in state_dict.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if invalid:
        raise ValueError(
            "Pinned GigaPath checkpoint contains NaN or Inf tensors: "
            + ", ".join(invalid[:5])
        )
    missing, unexpected = slide_encoder.load_state_dict(state_dict, strict=False)
    missing_set, unexpected_set = frozenset(missing), frozenset(unexpected)
    model_key_set = frozenset(slide_encoder.state_dict())
    parameter_key_set = frozenset(dict(slide_encoder.named_parameters()))
    missing_model_keys = missing_set.intersection(model_key_set | parameter_key_set)
    if missing_model_keys:
        raise RuntimeError(
            "Pinned GigaPath checkpoint is missing model/trainable parameters: "
            + ", ".join(sorted(missing_model_keys))
        )
    if missing_set != GIGAPATH_EXPECTED_MISSING_KEYS or unexpected_set != GIGAPATH_EXPECTED_UNEXPECTED_KEYS:
        raise RuntimeError(
            "Pinned GigaPath state-dict contract changed: "
            f"missing={sorted(missing_set)}, unexpected={sorted(unexpected_set)}; "
            f"expected missing={sorted(GIGAPATH_EXPECTED_MISSING_KEYS)}, "
            f"unexpected={sorted(GIGAPATH_EXPECTED_UNEXPECTED_KEYS)}"
        )
    return slide_encoder


def _initialize_feather_classifier(
    remote_model: nn.Module,
    num_classes: int,
) -> nn.Linear:
    """Attach a finite downstream classifier to a native FEATHER model.

    The pinned FEATHER checkpoint was published with ``num_classes=0`` and
    therefore contains zero-sized classifier tensors.  Asking Transformers to
    resize those tensors via ``ignore_mismatched_sizes`` can leave the remote
    classifier backed by uninitialized memory.  Load the pretrained aggregator
    in its native form and use FEATHER's own initializer instead.
    """
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f"FEATHER num_classes must be positive, got {num_classes}")

    initializer = getattr(remote_model, "initialize_classifier", None)
    core = getattr(remote_model, "model", None)
    if not callable(initializer) or core is None:
        raise TypeError(
            "FEATHER remote model must expose model and initialize_classifier()"
        )
    initializer(num_classes)

    classifier = getattr(core, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError(
            "FEATHER initialize_classifier() must create an nn.Linear classifier"
        )
    if classifier.out_features != num_classes:
        raise ValueError(
            "FEATHER classifier output mismatch: "
            f"expected {num_classes}, got {classifier.out_features}"
        )
    if not all(torch.isfinite(parameter).all() for parameter in classifier.parameters()):
        raise RuntimeError("FEATHER classifier initialization produced NaN or Inf")

    # Keep checkpoint/profile metadata consistent with the downstream head.
    config = getattr(remote_model, "config", None)
    if config is not None:
        config.num_classes = num_classes
    if hasattr(core, "num_classes"):
        core.num_classes = num_classes
    return classifier


def build_pretrained_backbone(args, num_classes: int) -> nn.Module:
    name = str(args.backbone).lower()
    defaults = {
        "titan": (TITAN_MODEL_ID, TITAN_REVISION),
        "feather": (FEATHER_MODEL_ID, FEATHER_REVISION),
        "gigapath": (GIGAPATH_MODEL_ID, GIGAPATH_REVISION),
    }
    model_id, revision = defaults[name]
    model_id = getattr(args, "backbone_model_id", None) or model_id
    revision = getattr(args, "backbone_revision", None) or revision
    freeze = bool(getattr(args, "backbone_freeze", False))
    if name == "gigapath":
        from utils.precision import validate_precision_configuration

        precision = validate_precision_configuration(args)
        if not torch.cuda.is_available():
            raise RuntimeError("GigaPath requires CUDA for the pinned FlashAttention runtime")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GigaPath BF16 requires a CUDA GPU with BF16 support")
        checkpoint_path = _resolve_hf_file(
            model_id,
            revision,
            GIGAPATH_CHECKPOINT_FILENAME,
            getattr(args, "backbone_cache_dir", None),
            bool(getattr(args, "backbone_allow_download", False)),
        )
        slide_encoder = _load_gigapath_slide(checkpoint_path)
        return GigaPathMILBackbone(slide_encoder, num_classes, freeze=freeze)

    snapshot = _resolve_snapshot(
        model_id,
        revision,
        getattr(args, "backbone_cache_dir", None),
        bool(getattr(args, "backbone_allow_download", False)),
    )
    if name == "titan":
        vision_encoder = _load_titan_vision(snapshot, revision)
        return TitanMILBackbone(vision_encoder, num_classes, freeze=freeze)
    # Load FEATHER with its native num_classes=0 checkpoint, then attach a
    # freshly initialized downstream head.  This preserves all pretrained
    # patch-embedding and attention weights without trusting mismatched-shape
    # initialization in the Transformers dynamic-module loader.
    remote_model = _load_auto_model(snapshot)
    _initialize_feather_classifier(remote_model, num_classes)
    return FeatherMILBackbone(remote_model, freeze=freeze)
