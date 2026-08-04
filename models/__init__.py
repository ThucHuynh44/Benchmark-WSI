# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic model discovery and construction."""

from __future__ import annotations

import importlib
from pathlib import Path


_MODEL_DIR = Path(__file__).resolve().parent


modules = {}
names = {}
for path in sorted(_MODEL_DIR.glob("*.py")):
    if path.name == "__init__.py":
        continue
    model_name = path.stem
    module = importlib.import_module("models." + model_name)
    normalized_name = model_name.replace("_", "")
    candidates = {
        name.lower(): value
        for name, value in vars(module).items()
        if isinstance(value, type)
    }
    # Helper modules may live next to methods (for example a prompt registry).
    # A runnable method is identified by the historical filename/class-name
    # convention; helpers are imported normally but are not exposed to the CLI.
    if normalized_name not in candidates:
        continue
    modules[model_name] = module
    names[model_name] = candidates[normalized_name]


def get_all_models():
    return sorted(names)


def validate_model_configuration(args) -> None:
    """Reject unsupported method/backbone contracts before loading weights."""
    model_name = str(args.model)
    if model_name not in names:
        raise ValueError(f"Unknown model {model_name!r}")
    model_class = names[model_name]
    backbone = str(getattr(args, "backbone", "generic_mil")).lower()

    supported = getattr(model_class, "SUPPORTED_BACKBONES", None)
    if supported:
        normalized = tuple(str(value).lower() for value in supported)
        if backbone not in normalized:
            raise ValueError(
                f"{model_name} supports backbones {normalized}, got {backbone!r}"
            )

    required_dim = getattr(model_class, "REQUIRED_FEATURE_DIM", None)
    if required_dim is not None:
        feature_dim = int(getattr(args, "feature_dim", required_dim))
        if feature_dim != int(required_dim):
            raise ValueError(
                f"{model_name} requires {int(required_dim)}-D patch features, "
                f"got feature_dim={feature_dim}"
            )

    if (
        bool(getattr(model_class, "REQUIRES_TRAINABLE_BACKBONE", False))
        and bool(getattr(args, "backbone_freeze", False))
    ):
        raise ValueError(
            f"{model_name} requires a trainable {backbone} backbone; "
            "--backbone_freeze is not supported"
        )

    custom_validator = getattr(modules[model_name], "validate_args", None)
    if custom_validator is not None:
        custom_validator(args)


def get_model(args, backbone, loss, transform):
    module = modules[args.model]
    builder = getattr(module, "build_model", None)
    if builder is not None:
        return builder(args, loss, transform)
    return names[args.model](backbone, loss, args, transform)
