"""YAML experiment configuration support for the argparse CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_experiment_config(
    path: Optional[str], method: Optional[str] = None, backbone: Optional[str] = None
) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {config_path}")

    if "common" in raw or "methods" in raw or "backbones" in raw:
        common = raw.get("common", {})
        methods = raw.get("methods", {})
        backbones = raw.get("backbones", {})
        if not isinstance(common, dict) or not isinstance(methods, dict) or not isinstance(backbones, dict):
            raise ValueError("'common', 'backbones', and 'methods' must be mappings")
        if not method:
            available = ", ".join(sorted(methods))
            raise ValueError(
                f"Multi-method config requires --model. Available methods: {available}"
            )
        if method not in methods:
            raise ValueError(
                f"Method {method!r} is not defined in {config_path}; "
                f"available: {', '.join(sorted(methods))}"
            )
        method_config = methods[method]
        if not isinstance(method_config, dict):
            raise ValueError(f"Configuration for method {method!r} must be a mapping")
        selected_backbone = backbone or common.get("backbone", "generic_mil")
        if selected_backbone not in backbones:
            raise ValueError(
                f"Backbone {selected_backbone!r} is not defined in {config_path}; "
                f"available: {', '.join(sorted(backbones))}"
            )
        backbone_config = backbones[selected_backbone]
        if not isinstance(backbone_config, dict):
            raise ValueError(f"Configuration for backbone {selected_backbone!r} must be a mapping")
        config = {
            **common,
            **backbone_config,
            **method_config,
            "model": method,
            "backbone": selected_backbone,
        }
        buffer_size = config.get("buffer_size")
        replacements = {
            "method": method,
            "backbone": selected_backbone,
            # ``buffer_tag`` is filename-friendly for both replay and
            # non-replay methods. ``buffer_size`` remains available when a
            # custom template needs only the raw value.
            "buffer_tag": (
                f"buffer{buffer_size}" if buffer_size is not None else "nobuffer"
            ),
            "buffer_size": buffer_size if buffer_size is not None else "na",
        }
        config = {
            # exp_desc is resolved only after argparse has applied explicit CLI
            # overrides (especially --buffer_size).
            key: (
                value
                if key == "exp_desc"
                else value.format(**replacements) if isinstance(value, str) else value
            )
            for key, value in config.items()
        }
    else:
        config = raw

    nested = [key for key, value in config.items() if isinstance(value, (dict, list)) and key != "backbone_kwargs"]
    if nested:
        raise ValueError(
            "Experiment config uses flat argparse names; nested/list values are invalid for: "
            + ", ".join(nested)
        )
    return config


def config_to_argv(config: Dict[str, Any]) -> List[str]:
    """Convert YAML defaults to CLI tokens; later real CLI tokens override them."""
    arguments: List[str] = []
    boolean_optional = {
        "early_stopping", "early_stopping_verbose",
        "atlas_replay", "atlas_diagnostics",
    }
    for key, value in config.items():
        if value is None:
            continue
        option = "--" + str(key)
        if key in boolean_optional and value is False:
            arguments.append("--no-" + str(key))
            continue
        if value is False:
            continue
        if value is True:
            arguments.append(option)
        else:
            if key == "backbone_kwargs" and isinstance(value, dict):
                value = json.dumps(value)
            arguments.extend([option, str(value)])
    return arguments
