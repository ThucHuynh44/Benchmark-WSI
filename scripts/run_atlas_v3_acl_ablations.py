"""Generate, audit, and run fold-isolated ATLAS-v3 ACL settings."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.atlas_v3_acl_registry import (
    EXPECTED_EPOCHS,
    FULL_RANK_IDS,
    SETTING_IDS,
    load_registry,
    select_variants,
)


OAS_DIAGNOSTIC_SEMANTICS = {
    "normalized_oas_static": "acl_only_normalized_oas_static_no_transport_v1",
    "transport_normalized_oas": "acl_only_normalized_oas_ungated_lowrank_transport_v1",
    "gated_transport_normalized_oas_no_histneg": "acl_only_normalized_oas_gated_lowrank_transport_v1",
}
FULL_RANK_DIAGNOSTIC_SEMANTICS = (
    "acl_only_normalized_oas_gated_full_ridge_transport_v1"
)


def parse_folds(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(10))
    folds = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(part) for part in token.split("-", 1))
            if stop < start:
                raise ValueError(f"Invalid fold range {token!r}")
            folds.update(range(start, stop + 1))
        else:
            folds.add(int(token))
    if not folds or min(folds) < 0 or max(folds) > 9:
        raise ValueError("Folds must be within 0..9")
    return sorted(folds)


def _option_tokens(name: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        return [f"--{name}" if value else f"--no-{name}"]
    return [f"--{name}", str(value)]


def experiment_desc(registry: Dict[str, Any], variant_id: str, fold: int) -> str:
    root = str(registry["defaults"].get("result_root", "ablations/atlas_v3_acl"))
    return f"{root.strip('/')}/{variant_id}/fold_{int(fold)}"


def build_command(registry: Dict[str, Any], variant: Dict[str, Any], fold: int) -> list[str]:
    defaults = registry["defaults"]
    command = [
        sys.executable, "utils/main.py",
        "--config", str(defaults.get("config", "configs/methods.yaml")),
        "--model", "atlas_v3_acl",
        "--backbone", "feather",
        "--folds", str(int(fold)),
        "--exp_desc", experiment_desc(registry, variant["id"], fold),
        "--ablation_id", variant["id"],
        "--ablation_group", variant["group"],
        "--ablation_config_hash", variant["config_hash"],
    ]
    for name, value in variant["overrides"].items():
        command.extend(_option_tokens(str(name), value))
    return command


def validate_command(command: Sequence[str]) -> None:
    from configs.experiment_loader import config_to_argv, load_experiment_config
    from models import validate_model_configuration

    tokens = list(command[2:])
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--model")
    bootstrap.add_argument("--backbone")
    known, _ = bootstrap.parse_known_args(tokens)
    configured = load_experiment_config(known.config, method=known.model, backbone=known.backbone)
    module = importlib.import_module("models." + str(known.model))
    parsed = module.get_parser().parse_args(config_to_argv(configured) + tokens)
    validate_model_configuration(parsed)


def resolved_audit(variant: Dict[str, Any], fold: int) -> str:
    mode = variant["overrides"]["atlasv3_acl_mode"]
    epochs = int(variant["overrides"].get("n_epochs", 1))
    adaptation = "NONE" if epochs == 0 else f"ACL-{epochs}-EPOCH"
    return (
        f"variant={variant['id']} fold={int(fold)} FEATHER={adaptation} "
        f"Mode={mode} Replay=NONE OldData=NONE HistNeg=NONE LoRA=NONE"
    )


def _read_keys(path: Path) -> Counter:
    if not path.is_file():
        return Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        return Counter((int(row["fold"]), int(row["after_task"]), int(row["eval_task"])) for row in csv.DictReader(handle))


def inspect_run(registry: Dict[str, Any], variant: Dict[str, Any], fold: int) -> str:
    description = experiment_desc(registry, variant["id"], fold)
    run_dir = REPO_ROOT / "results" / description
    checkpoint_dir = REPO_ROOT / "checkpoints" / description / f"fold_{int(fold)}"
    if not run_dir.exists():
        return "incomplete" if checkpoint_dir.exists() else "missing"
    manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
    if not manifest_path.is_file():
        return "incomplete"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ablation_id") != variant["id"]:
        return "mismatch"
    saved_mode = manifest.get("atlas_v3_acl_config", {}).get("mode")
    if saved_mode != variant["overrides"]["atlasv3_acl_mode"]:
        return "mismatch"
    expected_epochs = int(EXPECTED_EPOCHS.get(variant["id"], 1))
    saved_epochs = manifest.get("resolved_config", {}).get("n_epochs")
    if saved_epochs is None or int(saved_epochs) != expected_epochs:
        return "mismatch"
    mode = variant["overrides"]["atlasv3_acl_mode"]
    expected_semantics = (
        FULL_RANK_DIAGNOSTIC_SEMANTICS
        if variant["id"] in FULL_RANK_IDS
        else OAS_DIAGNOSTIC_SEMANTICS.get(mode)
    )
    saved_semantics = manifest.get("atlas_v3_acl_config", {}).get(
        "implementation_semantics"
    )
    if expected_semantics is not None and saved_semantics != expected_semantics:
        return "incomplete"
    num_tasks = int(manifest.get("num_tasks", 10))
    expected = {(int(fold), after, evaluated) for after in range(num_tasks) for evaluated in range(after + 1)}
    counts = [_read_keys(run_dir / f"evaluation/{mode}/eval_matrix.csv") for mode in ("class_il", "task_il")]
    complete = all(set(values) == expected and all(count == 1 for count in values.values()) for values in counts)
    return "complete" if complete else "incomplete"


def _run_job(job, gpu: str | None):
    variant, fold, command = job
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[atlas-v3-acl] start variant={variant['id']} fold={fold} gpu={gpu or 'inherited'}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    print(f"[atlas-v3-acl] end variant={variant['id']} fold={fold} exit={result.returncode}", flush=True)
    return variant["id"], int(fold), int(result.returncode)


def _execute(jobs, gpus: Sequence[str]) -> int:
    if not jobs:
        print("No ATLAS-v3 ACL fold-runs need execution.")
        return 0
    workers = list(gpus) or [None]
    queues = [jobs[index::len(workers)] for index in range(len(workers))]

    def run_queue(gpu, queue):
        results = []
        for job in queue:
            result = _run_job(job, gpu)
            results.append(result)
            if result[2]:
                break
        return results

    failures = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [
            executor.submit(run_queue, gpu, queue)
            for gpu, queue in zip(workers, queues)
        ]
        for future in as_completed(futures):
            failures.extend(value for value in future.result() if value[2])
    for variant_id, fold, code in failures:
        print(f"FAILED variant={variant_id} fold={fold} exit={code}", file=sys.stderr)
    return int(bool(failures))


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--folds", default="all")
    parser.add_argument("--gpus", default="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "configs/atlas_v3_acl_ablations.yaml"))
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    for name in ("dry-run", "run", "resume"):
        child = actions.add_parser(name)
        _add_selection_args(child)
        if name == "resume":
            child.add_argument("--rerun-incomplete", action="store_true")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    if args.action == "list":
        for variant in registry["variants"].values():
            print(resolved_audit(variant, 0).replace(" fold=0", ""))
        print(f"variants={len(SETTING_IDS)}")
        return 0
    variants = select_variants(registry, args.variants)
    folds = parse_folds(args.folds)
    jobs = []
    for variant in variants:
        for fold in folds:
            status = inspect_run(registry, variant, fold)
            command = build_command(registry, variant, fold)
            if args.action == "dry-run":
                validate_command(command)
                print(resolved_audit(variant, fold))
                print(f"[{status}] " + subprocess.list2cmdline(command))
                continue
            if args.action == "run" and status != "missing":
                raise RuntimeError(f"Refusing existing run {variant['id']} fold {fold}: status={status}")
            if args.action == "resume":
                if status == "complete":
                    print(f"[skip] complete variant={variant['id']} fold={fold}")
                    continue
                if status == "mismatch":
                    raise RuntimeError(f"Identity mismatch for {variant['id']} fold {fold}")
                if status == "incomplete" and not args.rerun_incomplete:
                    raise RuntimeError("Pass --rerun-incomplete to replace an incomplete run")
            jobs.append((variant, fold, command))
    if args.action == "dry-run":
        print(f"commands={len(variants) * len(folds)}")
        return 0
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    return _execute(jobs, gpus)


if __name__ == "__main__":
    raise SystemExit(main())
