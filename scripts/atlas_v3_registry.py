"""Load and validate the compact ATLAS-v3 experiment registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from configs.experiment_loader import load_experiment_config


SETTING_IDS = (
    "atlasv3_frozen_proto_diag",
    "atlasv3_frozen_proto_diag_shrink",
    "atlasv3_frozen_proto_lowrank",
    "atlasv3_frozen_proto_task_centroid",
    "atlasv3_frozen_proto_task_lme",
    "atlasv3_frozen_proto_multi",
    "atlasv3_frozen_proto_pt_only",
    "atlasv3_frozen_atlas_tf",
    "atlasv3_frozen_atlas_pt",
    "atlasv3_frozen_proto_oas_lda",
    "atlasv3_frozen_proto",
)

EXPECTED_MODES = {
    "atlasv3_frozen_proto_diag": "diag",
    "atlasv3_frozen_proto_diag_shrink": "diag_shrink",
    "atlasv3_frozen_proto_lowrank": "lowrank",
    "atlasv3_frozen_proto_task_centroid": "task_centroid",
    "atlasv3_frozen_proto_task_lme": "task_lme",
    "atlasv3_frozen_proto_multi": "multi",
    "atlasv3_frozen_proto_pt_only": "pt_only",
    "atlasv3_frozen_atlas_tf": "atlas_tf",
    "atlasv3_frozen_atlas_pt": "atlas_pt",
    "atlasv3_frozen_proto_oas_lda": "oas_lda",
    "atlasv3_frozen_proto": "prototype",
}


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_registry(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("ATLAS-v3 registry must be a YAML mapping")
    defaults, variants = raw.get("defaults", {}), raw.get("variants", {})
    if not isinstance(defaults, dict) or not isinstance(variants, dict):
        raise ValueError("ATLAS-v3 defaults and variants must be mappings")
    if tuple(variants) != SETTING_IDS:
        raise ValueError("ATLAS-v3 registry does not match the ordered setting schema")
    if int(defaults.get("expected_variants", -1)) != len(SETTING_IDS):
        raise ValueError(f"ATLAS-v3 expected_variants must equal {len(SETTING_IDS)}")
    if str(defaults.get("backbone", "")) != "feather":
        raise ValueError("ATLAS-v3 registry must use the frozen FEATHER backbone")

    config_path = Path(str(defaults.get("config", "configs/methods.yaml")))
    if not config_path.is_absolute():
        config_path = source.parents[1] / config_path
    required = {"id", "group", "model", "label", "factor", "value", "overrides"}
    digests: Dict[str, str] = {}
    for variant_id, entry in variants.items():
        if not isinstance(entry, dict) or required.difference(entry):
            raise ValueError(f"ATLAS-v3 variant {variant_id!r} is incomplete")
        if entry["id"] != variant_id or entry["model"] != "atlas_v3":
            raise ValueError(f"ATLAS-v3 identity mismatch for {variant_id}")
        overrides = entry["overrides"]
        expected = {"atlasv3_distribution_mode": EXPECTED_MODES[variant_id]}
        if overrides != expected:
            raise ValueError(
                f"ATLAS-v3 variant {variant_id} must only select mode "
                f"{EXPECTED_MODES[variant_id]!r}"
            )
        resolved = load_experiment_config(
            str(config_path), method="atlas_v3", backbone="feather"
        )
        resolved.update(overrides)
        for output_only in ("exp_desc", "folds", "csv_log", "tensorboard"):
            resolved.pop(output_only, None)
        digest = stable_hash(resolved)
        if digest in digests:
            raise ValueError(
                f"ATLAS-v3 settings {digests[digest]} and {variant_id} resolve identically"
            )
        digests[digest] = variant_id
        entry["resolved_config"] = resolved
        entry["config_hash"] = stable_hash(
            {"id": variant_id, "resolved_config": resolved}
        )
    return {"path": source, "defaults": defaults, "variants": variants}


def select_variants(
    registry: Mapping[str, Any], requested: Iterable[str]
) -> list[Dict[str, Any]]:
    requested = list(requested)
    ids = list(SETTING_IDS) if not requested or requested == ["all"] else requested
    unknown = [value for value in ids if value not in registry["variants"]]
    if unknown:
        raise ValueError("Unknown ATLAS-v3 settings: " + ", ".join(unknown))
    return [dict(registry["variants"][variant_id]) for variant_id in ids]
