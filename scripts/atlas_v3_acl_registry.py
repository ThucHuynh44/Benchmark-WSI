"""Load and validate the ATLAS-v3 ACL experiment registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from configs.experiment_loader import load_experiment_config


SETTING_IDS = (
    "atlasv3_acl",
    "atlasv3_acl_histneg",
    "atlasv3_acl_sdc",
    "atlasv3_acl_ldc",
    "atlasv3_acl_sldc",
    "atlasv3_acl_lowrank_transport",
    "atlasv3_acl_histneg_lowrank_transport",
    "atlasv3_acl_gated_transport",
    "atlasv3_acl_gated_transport_oas",
    "atlasv3_acl_gated_transport_task_margin",
    "atlasv3_control_frozen_raw_oas",
    "atlasv3_acl_oas_static",
    "atlasv3_acl_oas_transport",
    "atlasv3_acl_oas_oracle",
)

EXPECTED_MODES = {
    "atlasv3_acl": "acl",
    "atlasv3_acl_histneg": "histneg",
    "atlasv3_acl_sdc": "sdc",
    "atlasv3_acl_ldc": "ldc",
    "atlasv3_acl_sldc": "sldc",
    "atlasv3_acl_lowrank_transport": "lowrank",
    "atlasv3_acl_histneg_lowrank_transport": "histneg_lowrank",
    "atlasv3_acl_gated_transport": "gated",
    "atlasv3_acl_gated_transport_oas": "gated_oas",
    "atlasv3_acl_gated_transport_task_margin": "gated_task_margin",
    "atlasv3_control_frozen_raw_oas": "frozen_raw_oas",
    "atlasv3_acl_oas_static": "oas_static",
    "atlasv3_acl_oas_transport": "oas_transport",
    "atlasv3_acl_oas_oracle": "oas_oracle",
}


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_registry(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("ATLAS-v3 ACL registry must be a YAML mapping")
    defaults, variants = raw.get("defaults", {}), raw.get("variants", {})
    if not isinstance(defaults, dict) or not isinstance(variants, dict):
        raise ValueError("ATLAS-v3 ACL defaults and variants must be mappings")
    if tuple(variants) != SETTING_IDS:
        raise ValueError("ATLAS-v3 ACL registry does not match the ordered schema")
    if int(defaults.get("expected_variants", -1)) != len(SETTING_IDS):
        raise ValueError(f"expected_variants must equal {len(SETTING_IDS)}")
    if str(defaults.get("backbone", "")) != "feather":
        raise ValueError("ATLAS-v3 ACL registry must use FEATHER")
    config_path = Path(str(defaults.get("config", "configs/methods.yaml")))
    if not config_path.is_absolute():
        config_path = source.parents[1] / config_path
    required = {"id", "group", "model", "label", "factor", "value", "overrides"}
    digests: Dict[str, str] = {}
    for variant_id, entry in variants.items():
        if not isinstance(entry, dict) or required.difference(entry):
            raise ValueError(f"ATLAS-v3 ACL variant {variant_id!r} is incomplete")
        if entry["id"] != variant_id or entry["model"] != "atlas_v3_acl":
            raise ValueError(f"ATLAS-v3 ACL identity mismatch for {variant_id}")
        expected = {"atlasv3_acl_mode": EXPECTED_MODES[variant_id]}
        if entry["overrides"] != expected:
            raise ValueError(f"{variant_id} must only select {expected!r}")
        resolved = load_experiment_config(str(config_path), method="atlas_v3_acl", backbone="feather")
        resolved.update(expected)
        for output_only in ("exp_desc", "folds", "csv_log", "tensorboard"):
            resolved.pop(output_only, None)
        digest = stable_hash(resolved)
        if digest in digests:
            raise ValueError(f"Settings {digests[digest]} and {variant_id} resolve identically")
        digests[digest] = variant_id
        entry["resolved_config"] = resolved
        entry["config_hash"] = stable_hash({"id": variant_id, "resolved_config": resolved})
    return {"path": source, "defaults": defaults, "variants": variants}


def select_variants(registry: Mapping[str, Any], requested: Iterable[str]) -> list[Dict[str, Any]]:
    requested = list(requested)
    ids = list(SETTING_IDS) if not requested or requested == ["all"] else requested
    unknown = [value for value in ids if value not in registry["variants"]]
    if unknown:
        raise ValueError("Unknown ATLAS-v3 ACL settings: " + ", ".join(unknown))
    return [dict(registry["variants"][variant_id]) for variant_id in ids]
