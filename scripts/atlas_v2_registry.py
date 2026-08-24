"""Load and scientifically validate the eight-setting ATLAS-v2 registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from configs.experiment_loader import load_experiment_config


SETTING_IDS = (
    "atlasv2_base_frozen",
    "atlasv2_base_lora",
    "atlasv2_lora_replay",
    "atlasv2_lora_replay_proto",
    "atlasv2_lora_replay_proto_realign",
    "atlasv2_frozen_proto",
    "atlasv2_lora_replay_proto_realign_prompt",
    "atlasv2_lora_replay_proto_realign_prompt_nce",
)
MECHANISM_FIELDS = (
    "atlasv2_lora", "atlasv2_replay", "atlasv2_prototype",
    "atlasv2_realign", "atlasv2_prompt", "atlasv2_nce",
)
PAIRWISE = (
    ("atlasv2_base_frozen", "atlasv2_base_lora", "atlasv2_lora"),
    ("atlasv2_base_lora", "atlasv2_lora_replay", "atlasv2_replay"),
    ("atlasv2_lora_replay", "atlasv2_lora_replay_proto", "atlasv2_prototype"),
    (
        "atlasv2_lora_replay_proto", "atlasv2_lora_replay_proto_realign",
        "atlasv2_realign",
    ),
    (
        "atlasv2_lora_replay_proto_realign",
        "atlasv2_lora_replay_proto_realign_prompt", "atlasv2_prompt",
    ),
    (
        "atlasv2_lora_replay_proto_realign_prompt",
        "atlasv2_lora_replay_proto_realign_prompt_nce", "atlasv2_nce",
    ),
)


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_registry(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("ATLAS-v2 registry must be a YAML mapping")
    defaults, variants = raw.get("defaults", {}), raw.get("variants", {})
    if not isinstance(defaults, dict) or not isinstance(variants, dict):
        raise ValueError("ATLAS-v2 defaults and variants must be mappings")
    if tuple(variants) != SETTING_IDS:
        raise ValueError("ATLAS-v2 registry must contain exactly the eight ordered settings")
    if int(defaults.get("expected_variants", -1)) != len(SETTING_IDS):
        raise ValueError("ATLAS-v2 expected_variants must equal 8")

    config_path = Path(str(defaults.get("config", "configs/methods.yaml")))
    if not config_path.is_absolute():
        config_path = source.parents[1] / config_path
    digests: Dict[str, str] = {}
    required = {"id", "group", "model", "label", "factor", "value", "overrides"}
    for variant_id, entry in variants.items():
        if not isinstance(entry, dict) or required.difference(entry):
            raise ValueError(f"ATLAS-v2 variant {variant_id!r} is incomplete")
        if entry["id"] != variant_id or entry["model"] != "atlas_v2":
            raise ValueError(f"ATLAS-v2 identity mismatch for {variant_id}")
        overrides = entry["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(f"ATLAS-v2 overrides must be a mapping for {variant_id}")
        if any(field not in overrides for field in MECHANISM_FIELDS):
            raise ValueError(f"ATLAS-v2 mechanisms must be explicit for {variant_id}")
        replay = bool(overrides["atlasv2_replay"])
        if int(overrides.get("buffer_size", -1)) != (30 if replay else 0):
            raise ValueError(f"ATLAS-v2 buffer budget mismatch for {variant_id}")
        resolved = load_experiment_config(
            str(config_path), method="atlas_v2",
            backbone=str(defaults.get("backbone", "feather")),
        )
        resolved.update(overrides)
        for output_only in ("exp_desc", "folds", "csv_log", "tensorboard"):
            resolved.pop(output_only, None)
        digest = stable_hash(resolved)
        if digest in digests:
            raise ValueError(
                f"ATLAS-v2 settings {digests[digest]} and {variant_id} resolve identically"
            )
        digests[digest] = variant_id
        entry["resolved_config"] = resolved
        entry["config_hash"] = stable_hash({"id": variant_id, "resolved_config": resolved})

    for left_id, right_id, changed in PAIRWISE:
        left = dict(variants[left_id]["overrides"])
        right = dict(variants[right_id]["overrides"])
        # Replay necessarily changes the declared capacity from no-memory 0 to 30.
        allowed = {changed, "buffer_size"} if changed == "atlasv2_replay" else {changed}
        differences = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
        if differences != allowed:
            raise ValueError(
                f"ATLAS-v2 comparison {left_id} -> {right_id} differs in {sorted(differences)}"
            )
    return {"path": source, "defaults": defaults, "variants": variants}


def select_variants(registry: Mapping[str, Any], requested: Iterable[str]) -> list[Dict[str, Any]]:
    requested = list(requested)
    ids = list(SETTING_IDS) if not requested or requested == ["all"] else requested
    unknown = [value for value in ids if value not in registry["variants"]]
    if unknown:
        raise ValueError("Unknown ATLAS-v2 settings: " + ", ".join(unknown))
    return [dict(registry["variants"][variant_id]) for variant_id in ids]
