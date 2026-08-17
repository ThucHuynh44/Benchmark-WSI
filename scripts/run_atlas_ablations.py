"""Run fold-isolated ATLAS-MIL ablations from the declarative registry."""

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
from typing import Any, Dict, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.atlas_ablation_registry import load_registry, select_variants


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
    option = "--" + name
    if isinstance(value, bool):
        return [option if value else "--no-" + name]
    return [option, str(value)]


def experiment_desc(registry: Dict[str, Any], variant_id: str, fold: int) -> str:
    root = str(registry["defaults"].get("result_root", "ablations/atlas_mil")).strip("/")
    return f"{root}/{variant_id}/fold_{int(fold)}"


def build_command(registry: Dict[str, Any], variant: Dict[str, Any], fold: int) -> list[str]:
    defaults = registry["defaults"]
    command = [
        sys.executable,
        "utils/main.py",
        "--config", str(defaults.get("config", "configs/methods.yaml")),
        "--model", str(variant["model"]),
        "--backbone", str(defaults.get("backbone", "feather")),
        "--folds", str(int(fold)),
        "--exp_desc", experiment_desc(registry, variant["id"], fold),
        "--ablation_id", str(variant["id"]),
        "--ablation_group", str(variant["group"]),
        "--ablation_config_hash", str(variant["config_hash"]),
    ]
    for name, value in variant["overrides"].items():
        command.extend(_option_tokens(str(name), value))
    return command


def validate_command(command: Sequence[str]) -> None:
    """Parse and validate one rendered command without building a model."""
    from configs.experiment_loader import config_to_argv, load_experiment_config
    from models import validate_model_configuration

    tokens = list(command[2:])
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--model")
    bootstrap.add_argument("--backbone")
    known, _ = bootstrap.parse_known_args(tokens)
    configured = load_experiment_config(
        known.config, method=known.model, backbone=known.backbone
    )
    module = importlib.import_module("models." + str(known.model))
    parsed = module.get_parser().parse_args(config_to_argv(configured) + tokens)
    validate_model_configuration(parsed)


def _read_keys(path: Path) -> Counter:
    if not path.is_file():
        return Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        return Counter(
            (int(row["fold"]), int(row["after_task"]), int(row["eval_task"]))
            for row in csv.DictReader(handle)
        )


def inspect_run(registry: Dict[str, Any], variant: Dict[str, Any], fold: int) -> str:
    description = experiment_desc(registry, variant["id"], fold)
    run_dir = REPO_ROOT / "results" / description
    checkpoint_dir = REPO_ROOT / "checkpoints" / description / f"fold_{int(fold)}"
    if not run_dir.exists():
        return "incomplete" if checkpoint_dir.exists() else "missing"
    manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
    if not manifest_path.is_file():
        return "incomplete"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("ablation_id") != variant["id"]
        or manifest.get("ablation_config_hash") != variant["config_hash"]
        or [int(value) for value in manifest.get("folds", [])] != [int(fold)]
    ):
        return "mismatch"
    expected = {
        (int(fold), after, evaluated)
        for after in range(10)
        for evaluated in range(after + 1)
    }
    class_counts = _read_keys(run_dir / "evaluation/class_il/eval_matrix.csv")
    task_counts = _read_keys(run_dir / "evaluation/task_il/eval_matrix.csv")
    complete = all(
        set(counts) == expected and all(count == 1 for count in counts.values())
        for counts in (class_counts, task_counts)
    )
    return "complete" if complete else "incomplete"


def _run_job(job, gpu: str | None) -> tuple[str, int, int]:
    variant, fold, command = job
    environment = os.environ.copy()
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[ablation] start variant={variant['id']} fold={fold} gpu={gpu or 'inherited'}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    print(f"[ablation] end variant={variant['id']} fold={fold} exit={result.returncode}", flush=True)
    return variant["id"], int(fold), int(result.returncode)


def _execute(jobs, gpus: Sequence[str]) -> int:
    if not jobs:
        print("No fold-runs need execution.")
        return 0
    workers = list(gpus) or [None]
    queues = [jobs[index::len(workers)] for index in range(len(workers))]

    def run_queue(gpu, queue):
        results = []
        for job in queue:
            result = _run_job(job, gpu)
            results.append(result)
            if result[2] != 0:
                break
        return results

    failures = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(run_queue, gpu, queue) for gpu, queue in zip(workers, queues)]
        for future in as_completed(futures):
            failures.extend(result for result in future.result() if result[2] != 0)
    if failures:
        for variant_id, fold, code in failures:
            print(f"FAILED variant={variant_id} fold={fold} exit={code}", file=sys.stderr)
        return 1
    return 0


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--gpus", default="",
        help="Comma-separated physical GPU IDs; one sequential worker is used per GPU.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(REPO_ROOT / "configs/atlas_mil_ablations.yaml"),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list")
    dry = subparsers.add_parser("dry-run")
    _add_selection_args(dry)
    run = subparsers.add_parser("run")
    _add_selection_args(run)
    resume = subparsers.add_parser("resume")
    _add_selection_args(resume)
    resume.add_argument("--rerun-incomplete", action="store_true")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    if args.action == "list":
        for entry in registry["variants"].values():
            print(
                f"{entry['id']:<34} {entry['group']:<24} "
                f"{entry['factor']}={entry['value']}"
            )
        print(f"variants={len(registry['variants'])} folds=10 fold_runs={len(registry['variants']) * 10}")
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
                print(f"[{status}] " + subprocess.list2cmdline(command))
                continue
            if args.action == "run" and status != "missing":
                raise RuntimeError(
                    f"Refusing existing run {variant['id']} fold {fold}: status={status}"
                )
            if args.action == "resume":
                if status == "complete":
                    print(f"[skip] complete variant={variant['id']} fold={fold}")
                    continue
                if status == "mismatch":
                    raise RuntimeError(
                        f"Config hash/identity mismatch for {variant['id']} fold {fold}"
                    )
                if status == "incomplete" and not args.rerun_incomplete:
                    raise RuntimeError(
                        f"Incomplete run {variant['id']} fold {fold}; pass --rerun-incomplete"
                    )
            jobs.append((variant, fold, command))

    if args.action == "dry-run":
        print(f"commands={len(variants) * len(folds)}")
        return 0
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    return _execute(jobs, gpus)


if __name__ == "__main__":
    raise SystemExit(main())
