"""Load and scientifically validate the ATLAS-v2 setting registry."""

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
    "atlasv2_base_lora_svd_orthogonal",
    "atlasv2_replay_proto",
    "atlasv2_lora_proto",
    "atlasv2_base_lora_comel_owlora",
    "atlasv2_frozen_proto_oas_lda",
    "atlasv2_frozen_proto_diag",
    "atlasv2_frozen_proto_diag_shrink",
    "atlasv2_frozen_proto_lowrank",
    "atlasv2_frozen_proto_task_centroid",
    "atlasv2_frozen_proto_task_lme",
    "atlasv2_frozen_proto_multi",
    "atlasv2_frozen_proto_pt_only",
    "atlasv2_frozen_atlas_tf",
    "atlasv2_frozen_atlas_pt",
    "atlasv2_frozen_ranpac",
)
COMEL_SETTING_ID = "atlasv2_base_lora_comel_owlora"
FROZEN_PROTO_LDA_ID = "atlasv2_frozen_proto_oas_lda"
DISTRIBUTION_SETTING_IDS = SETTING_IDS[-10:]
PROTO_FACTORIAL_IDS = (
    "atlasv2_frozen_proto",
    "atlasv2_replay_proto",
    "atlasv2_lora_proto",
    "atlasv2_lora_replay_proto",
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
        raise ValueError("ATLAS-v2 registry does not match the ordered setting schema")
    if int(defaults.get("expected_variants", -1)) != len(SETTING_IDS):
        raise ValueError(f"ATLAS-v2 expected_variants must equal {len(SETTING_IDS)}")

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
    base = variants["atlasv2_base_lora"]["overrides"]
    geometry = variants["atlasv2_base_lora_svd_orthogonal"]["overrides"]
    geometry_differences = {
        key for key in set(base) | set(geometry) if base.get(key) != geometry.get(key)
    }
    if geometry_differences != {"atlasv2_svd_orthogonal", "atlasv2_svd_energy"}:
        raise ValueError(
            "ATLAS-v2 SVD-orthogonal extension must otherwise match base LoRA"
        )
    comel = variants[COMEL_SETTING_ID]["overrides"]
    comel_differences = {
        key for key in set(base) | set(comel) if base.get(key) != comel.get(key)
    }
    expected_comel = {
        "atlasv2_comel_owlora", "atlasv2_comel_svd_energy",
        "atlasv2_comel_orthogonal_weight",
    }
    if comel_differences != expected_comel:
        raise ValueError("ATLAS-v2 CoMEL OWLoRA must otherwise match base LoRA")
    frozen_proto = variants["atlasv2_frozen_proto"]["overrides"]
    frozen_proto_lda = variants[FROZEN_PROTO_LDA_ID]["overrides"]
    lda_differences = {
        key for key in set(frozen_proto) | set(frozen_proto_lda)
        if frozen_proto.get(key) != frozen_proto_lda.get(key)
    }
    if lda_differences != {"atlasv2_prototype_lda"}:
        raise ValueError(
            "ATLAS-v2 OAS-LDA extension must otherwise match frozen prototypes"
        )
    expected_distribution_modes = {
        setting.removeprefix("atlasv2_frozen_proto_"): setting
        for setting in DISTRIBUTION_SETTING_IDS[:7]
    }
    expected_distribution_modes.update({
        "atlas_tf": "atlasv2_frozen_atlas_tf",
        "atlas_pt": "atlasv2_frozen_atlas_pt",
        "ranpac": "atlasv2_frozen_ranpac",
    })
    for mode, variant_id in expected_distribution_modes.items():
        overrides = variants[variant_id]["overrides"]
        if overrides.get("atlasv2_distribution_mode") != mode:
            raise ValueError(f"{variant_id} must select distribution mode {mode}")
        if any(bool(overrides[field]) for field in (
            "atlasv2_lora", "atlasv2_replay", "atlasv2_realign",
            "atlasv2_prompt", "atlasv2_nce", "atlasv2_train_classifier",
        )) or not bool(overrides["atlasv2_prototype"]):
            raise ValueError(f"{variant_id} is not a frozen prototype-only setting")
    expected_cells = {
        "atlasv2_frozen_proto": (False, False, False),
        "atlasv2_replay_proto": (False, True, False),
        "atlasv2_lora_proto": (True, False, True),
        "atlasv2_lora_replay_proto": (True, True, True),
    }
    for variant_id in PROTO_FACTORIAL_IDS:
        overrides = variants[variant_id]["overrides"]
        actual = (
            bool(overrides["atlasv2_lora"]),
            bool(overrides["atlasv2_replay"]),
            bool(overrides["atlasv2_train_classifier"]),
        )
        if actual != expected_cells[variant_id]:
            raise ValueError(
                f"ATLAS-v2 prototype factorial cell {variant_id} is malformed"
            )
        if not bool(overrides["atlasv2_prototype"]) or any(
            bool(overrides[field])
            for field in ("atlasv2_realign", "atlasv2_prompt", "atlasv2_nce")
        ):
            raise ValueError(
                "ATLAS-v2 prototype factorial requires prototype only, without "
                f"realignment/prompt/NCE: {variant_id}"
            )
    return {"path": source, "defaults": defaults, "variants": variants}


def select_variants(registry: Mapping[str, Any], requested: Iterable[str]) -> list[Dict[str, Any]]:
    requested = list(requested)
    ids = list(SETTING_IDS) if not requested or requested == ["all"] else requested
    unknown = [value for value in ids if value not in registry["variants"]]
    if unknown:
        raise ValueError("Unknown ATLAS-v2 settings: " + ", ".join(unknown))
    return [dict(registry["variants"][variant_id]) for variant_id in ids]
