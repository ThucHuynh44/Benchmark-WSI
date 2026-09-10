"""Run a fold-isolated sweep for ACL gated normalized-OAS transport.

The sweep intentionally keeps ACL training and OAS-LDA fixed.  Only the
post-hoc historical-statistics transport is varied.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = "ablations/atlas_v3_gated_transport_sweep"
BASE_OVERRIDES: dict[str, Any] = {
    "atlasv3_acl_mode": "gated_transport_normalized_oas_no_histneg",
    "atlasv3_acl_transport_rank": 8,
    "atlasv3_acl_transport_ridge": 1.0e-3,
    "atlasv3_acl_transport_mean_scale": 1.0,
    "atlasv3_acl_transport_cov_scale": 1.0,
    "atlasv3_acl_coverage_energy": 0.95,
    "atlasv3_acl_bootstrap_samples": 20,
    "atlasv3_acl_uncertainty_beta": 10.0,
}


def _setting(
    setting_id: str,
    group: str | Iterable[str],
    **overrides: Any,
) -> dict[str, Any]:
    values = dict(BASE_OVERRIDES)
    values.update(overrides)
    groups = (group,) if isinstance(group, str) else tuple(group)
    payload = {"id": setting_id, "groups": groups, "overrides": values}
    payload["config_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


SETTINGS = (
    _setting(
        "gated_m050_c000_r8_l1em3",
        "strength",
        atlasv3_acl_transport_mean_scale=0.50,
        atlasv3_acl_transport_cov_scale=0.0,
    ),
    _setting(
        "map_gated_m050_c000_r1_l1em3",
        "map",
        atlasv3_acl_transport_rank=1,
        atlasv3_acl_transport_mean_scale=0.50,
        atlasv3_acl_transport_cov_scale=0.0,
    ),
    _setting(
        "map_gated_m050_c000_r4_l1em3",
        "map",
        atlasv3_acl_transport_rank=4,
        atlasv3_acl_transport_mean_scale=0.50,
        atlasv3_acl_transport_cov_scale=0.0,
    ),
)

def select_settings(groups: Sequence[str]) -> list[dict[str, Any]]:
    requested = set(groups)
    unknown = requested.difference({"strength", "map", "all"})
    if unknown:
        raise ValueError("Unknown sweep groups: " + ", ".join(sorted(unknown)))
    if "all" in requested:
        return list(SETTINGS)
    return [setting for setting in SETTINGS if requested.intersection(setting["groups"])]


def select_setting_ids(value: str) -> list[dict[str, Any]]:
    requested = [
        token.strip()
        for token in value.replace(":", ",").split(",")
        if token.strip()
    ]
    available = {setting["id"]: setting for setting in SETTINGS}
    unknown = [setting_id for setting_id in requested if setting_id not in available]
    if unknown:
        raise ValueError("Unknown sweep settings: " + ", ".join(unknown))
    if not requested:
        raise ValueError("--settings must contain at least one setting ID")
    if len(requested) != len(set(requested)):
        raise ValueError("--settings contains duplicate setting IDs")
    return [available[setting_id] for setting_id in requested]


def experiment_desc(setting: Mapping[str, Any], fold: int) -> str:
    return f"{RESULT_ROOT}/{setting['id']}/fold_{int(fold)}"


def build_command(setting: Mapping[str, Any], fold: int) -> list[str]:
    command = [
        sys.executable,
        "utils/main.py",
        "--config",
        "configs/methods.yaml",
        "--model",
        "atlas_v3_acl",
        "--backbone",
        "feather",
        "--folds",
        str(int(fold)),
        "--exp_desc",
        experiment_desc(setting, fold),
        "--ablation_id",
        str(setting["id"]),
        "--ablation_group",
        "atlas_v3_gated_transport_sweep",
        "--ablation_config_hash",
        str(setting["config_hash"]),
    ]
    for name, value in setting["overrides"].items():
        command.extend((f"--{name}", str(value)))
    return command


def _read_keys(path: Path) -> Counter:
    if not path.is_file():
        return Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        return Counter(
            (int(row["fold"]), int(row["after_task"]), int(row["eval_task"]))
            for row in csv.DictReader(handle)
        )


def inspect_run(setting: Mapping[str, Any], fold: int) -> str:
    run_dir = REPO_ROOT / "results" / experiment_desc(setting, fold)
    checkpoint_dir = REPO_ROOT / "checkpoints" / experiment_desc(setting, fold) / f"fold_{int(fold)}"
    if not run_dir.exists():
        return "incomplete" if checkpoint_dir.exists() else "missing"
    manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
    if not manifest_path.is_file():
        return "incomplete"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("ablation_id") != setting["id"]
        or manifest.get("ablation_config_hash") != setting["config_hash"]
    ):
        return "mismatch"
    num_tasks = int(manifest.get("num_tasks", 10))
    expected = {
        (int(fold), after, evaluated)
        for after in range(num_tasks)
        for evaluated in range(after + 1)
    }
    matrices = (
        _read_keys(run_dir / "evaluation/class_il/eval_matrix.csv"),
        _read_keys(run_dir / "evaluation/task_il/eval_matrix.csv"),
    )
    complete = all(
        set(matrix) == expected and all(count == 1 for count in matrix.values())
        for matrix in matrices
    )
    return "complete" if complete else "incomplete"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "resume", "list"))
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument(
        "--settings",
        default="",
        help="Comma- or colon-separated setting IDs; overrides --groups when provided.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--rerun-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.fold <= 9:
        parser.error("--fold must be in 0..9")
    settings = select_setting_ids(args.settings) if args.settings else select_settings(args.groups)
    if args.action == "list":
        for setting in settings:
            print(setting["id"], json.dumps(setting["overrides"], sort_keys=True))
        return 0
    for setting in settings:
        status = inspect_run(setting, args.fold)
        if status == "complete" and args.action == "resume":
            print(f"[skip] complete setting={setting['id']} fold={args.fold}", flush=True)
            continue
        if status == "mismatch":
            raise RuntimeError(f"Identity mismatch for {setting['id']} fold {args.fold}")
        if status == "incomplete" and not args.rerun_incomplete:
            raise RuntimeError(
                f"Incomplete run {setting['id']} fold {args.fold}; pass --rerun-incomplete"
            )
        if args.action == "run" and status != "missing":
            raise RuntimeError(
                f"Refusing existing run {setting['id']} fold {args.fold}: status={status}"
            )
        print(
            f"[atlas-v3-gated-sweep] start setting={setting['id']} "
            f"fold={args.fold} prior_status={status}",
            flush=True,
        )
        result = subprocess.run(build_command(setting, args.fold), cwd=REPO_ROOT, check=False)
        print(
            f"[atlas-v3-gated-sweep] end setting={setting['id']} exit={result.returncode}",
            flush=True,
        )
        if result.returncode:
            return int(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
