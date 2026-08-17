"""Load and validate the declarative ATLAS-MIL ablation registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from configs.experiment_loader import load_experiment_config


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _value_key(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def load_registry(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("ATLAS ablation registry must be a YAML mapping")
    defaults = raw.get("defaults", {})
    explicit = raw.get("variants", {})
    axes = raw.get("axes", {})
    if not all(isinstance(value, dict) for value in (defaults, explicit, axes)):
        raise ValueError("Registry defaults, variants, and axes must be mappings")
    atlas_defaults = defaults.get("atlas_overrides", {})
    if not isinstance(atlas_defaults, dict):
        raise ValueError("defaults.atlas_overrides must be a mapping")

    variants: Dict[str, Dict[str, Any]] = {}
    required = {"id", "group", "model", "label", "factor", "value", "overrides"}
    for key, entry in explicit.items():
        if not isinstance(entry, dict) or required.difference(entry):
            raise ValueError(f"Explicit variant {key!r} is missing {sorted(required.difference(entry or {}))}")
        if entry["id"] != key:
            raise ValueError(f"Variant key/id mismatch: {key!r} != {entry['id']!r}")
        if not isinstance(entry["overrides"], dict):
            raise ValueError(f"Variant {key!r} overrides must be a mapping")
        overrides = (
            {**atlas_defaults, **entry["overrides"]}
            if entry["model"] == "atlas_mil" else dict(entry["overrides"])
        )
        variants[key] = {**entry, "overrides": overrides, "source": "explicit"}

    axis_members: Dict[str, list] = {}
    for axis_name, axis in axes.items():
        if not isinstance(axis, dict):
            raise ValueError(f"Axis {axis_name!r} must be a mapping")
        for field in ("group", "option", "id_template", "label_template", "values"):
            if field not in axis:
                raise ValueError(f"Axis {axis_name!r} is missing {field!r}")
        reuse = {_value_key(key): value for key, value in axis.get("reuse", {}).items()}
        members = []
        for value in axis["values"]:
            key = _value_key(value)
            if key in reuse:
                variant_id = str(reuse[key])
                if variant_id not in variants:
                    raise ValueError(f"Axis {axis_name!r} reuses unknown variant {variant_id!r}")
            else:
                token = str(value).replace(".", "p")
                variant_id = str(axis["id_template"]).format(value=token)
                if variant_id in variants:
                    raise ValueError(f"Duplicate generated variant {variant_id!r}")
                variants[variant_id] = {
                    "id": variant_id,
                    "group": axis["group"],
                    "model": "atlas_mil",
                    "label": str(axis["label_template"]).format(value=value),
                    "factor": axis["option"],
                    "value": value,
                    "overrides": {**atlas_defaults, axis["option"]: value},
                    "source": "axis",
                }
            members.append({"value": value, "variant_id": variant_id})
        axis_members[axis_name] = members

    expected = int(defaults.get("expected_variants", len(variants)))
    if len(variants) != expected:
        raise ValueError(f"Registry resolves {len(variants)} variants, expected {expected}")
    if variants.get("sgd_ft", {}).get("group") == "additive":
        raise ValueError("sgd_ft must remain outside the additive ladder")

    seen_configs: Dict[str, str] = {}
    config_path = Path(str(defaults.get("config", "configs/methods.yaml")))
    if not config_path.is_absolute():
        config_path = source.parents[1] / config_path
    for variant_id, entry in variants.items():
        identity = {"model": entry["model"], "overrides": entry["overrides"]}
        digest = _stable_hash(identity)
        if digest in seen_configs:
            raise ValueError(
                f"Variants {seen_configs[digest]!r} and {variant_id!r} resolve identically"
            )
        seen_configs[digest] = variant_id
        resolved = load_experiment_config(
            str(config_path),
            method=str(entry["model"]),
            backbone=str(defaults.get("backbone", "feather")),
        )
        resolved.update(entry["overrides"])
        for output_only in ("exp_desc", "folds", "csv_log", "tensorboard"):
            resolved.pop(output_only, None)
        entry["resolved_config"] = resolved
        entry["config_hash"] = _stable_hash({
            "id": variant_id,
            "resolved_config": resolved,
        })

    return {
        "path": source,
        "defaults": defaults,
        "variants": variants,
        "axes": axes,
        "axis_members": axis_members,
    }


def select_variants(registry: Mapping[str, Any], requested: Iterable[str]) -> list[Dict[str, Any]]:
    requested = list(requested)
    ids = list(registry["variants"]) if not requested or requested == ["all"] else requested
    unknown = [value for value in ids if value not in registry["variants"]]
    if unknown:
        raise ValueError(f"Unknown ablation variants: {', '.join(unknown)}")
    return [dict(registry["variants"][value]) for value in ids]
