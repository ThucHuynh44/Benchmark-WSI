"""Summarize ATLAS-v2 without ranking or automatically selecting a winner."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.atlas_v2_registry import (
    COMEL_SETTING_ID, DISTRIBUTION_SETTING_IDS, FROZEN_PROTO_LDA_ID,
    PROTO_FACTORIAL_IDS, SETTING_IDS,
    load_registry,
)
from scripts.build_cl_table import (
    _complete_folds, _expected_keys, _read_eval_matrix, _rows_by_key,
    _sequential_fold_metrics,
)
from scripts.run_atlas_v2_ablations import experiment_desc, valid_calibration_manifest


METRICS = ("mACC", "bACC", "masked_bACC", "BWT", "FGT")
MEMORY_FIELDS = ("retained_wsis", "retained_patch_rows", "replay_memory_mib")
MEMORY_CACHE_VERSION = 1
DISTRIBUTION_DIAGNOSTICS = (
    "cross_task_error_rate", "soft_task_top1_accuracy",
    "masked_bacc_minus_class_il_bacc", "effective_lowrank_rank",
    "prototype_offset_norm", "distribution_memory_mib", "checkpoint_memory_mib",
)


def _finite(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _mean_std(values: Iterable[Any]) -> tuple[float, float]:
    values = _finite(values)
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values)


def _memory_from_manifest(manifest: Mapping[str, Any], variant: Mapping[str, Any]) -> dict:
    """Read lightweight replay accounting already recorded by training.

    Never deserialize full-bag checkpoints here: a completed replay checkpoint
    is hundreds of MiB, while the manifest already contains these three scalar
    values.
    """

    if not variant["overrides"]["atlasv2_replay"]:
        return {"retained_wsis": 0, "retained_patch_rows": 0, "replay_memory_mib": 0.0}
    accounting = manifest.get("replay_memory_accounting")
    if not isinstance(accounting, Mapping):
        return {field: math.nan for field in MEMORY_FIELDS}
    try:
        values = {
            "retained_wsis": int(accounting["retained_wsis"]),
            "retained_patch_rows": int(accounting["retained_patch_rows"]),
            "replay_memory_mib": float(accounting["replay_memory_mib"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return {field: math.nan for field in MEMORY_FIELDS}
    if any(value < 0 or not math.isfinite(float(value)) for value in values.values()):
        return {field: math.nan for field in MEMORY_FIELDS}
    # Historical manifests captured metadata before training and therefore
    # contain an impossible empty buffer even for completed replay runs.
    if values["retained_wsis"] == 0:
        return {field: math.nan for field in MEMORY_FIELDS}
    return values


def _read_memory_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != MEMORY_CACHE_VERSION
        or not isinstance(payload.get("entries"), dict)
    ):
        return {}
    return dict(payload["entries"])


def _write_memory_cache(path: Path, entries: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MEMORY_CACHE_VERSION, "entries": dict(entries)}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _latest_checkpoint(registry, variant, fold: int) -> Path | None:
    description = experiment_desc(registry, variant["id"], fold)
    root = REPO_ROOT / "checkpoints" / description / f"fold_{int(fold)}"
    candidates = []
    for path in root.glob("task*_checkpoint.pt"):
        match = re.fullmatch(r"task(\d+)_checkpoint\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def _memory_from_checkpoint_cache(registry, variant, fold: int, cache: dict) -> dict:
    """Use cached accounting, or mmap one legacy checkpoint without reading storage."""

    path = _latest_checkpoint(registry, variant, fold)
    if path is None:
        return {field: math.nan for field in MEMORY_FIELDS}
    stat = path.stat()
    cache_key = f"{variant['id']}/fold_{int(fold)}"
    fingerprint = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and all(
        cached.get(name) == value for name, value in fingerprint.items()
    ):
        values = cached.get("accounting")
        if isinstance(values, dict) and all(field in values for field in MEMORY_FIELDS):
            return {field: values[field] for field in MEMORY_FIELDS}

    # Lazy import keeps the common/cache-hit path lightweight. mmap=True maps
    # tensor storage and lets shape/numel metadata be inspected without reading
    # the full feature bags into RAM.
    import torch

    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    memory = payload.get("method_state", {}).get("memory", {})
    entries = memory.get("entries", []) if isinstance(memory, dict) else []
    rows = sum(int(entry["features"].shape[0]) for entry in entries)
    byte_count = sum(
        tensor.numel() * tensor.element_size()
        for entry in entries
        for tensor in (
            entry["features"], entry["coords"], entry["patch_size"], entry["label"],
        )
    )
    values = {
        "retained_wsis": len(entries), "retained_patch_rows": rows,
        "replay_memory_mib": byte_count / (1024.0 ** 2),
    }
    cache[cache_key] = {**fingerprint, "checkpoint": str(path), "accounting": values}
    del payload
    return values


def _load_run_once(registry, variant, fold: int):
    """Audit one run and retain its already-read manifest/matrices."""

    description = experiment_desc(registry, variant["id"], fold)
    run_dir = REPO_ROOT / "results" / description
    checkpoint_dir = REPO_ROOT / "checkpoints" / description / f"fold_{int(fold)}"
    if not run_dir.exists():
        status = "incomplete" if checkpoint_dir.exists() else "missing"
        return status, None, None, None
    manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
    if not manifest_path.is_file():
        return "incomplete", None, None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("ablation_id") != variant["id"]
        or manifest.get("ablation_config_hash") != variant["config_hash"]
        or [int(value) for value in manifest.get("folds", [])] != [int(fold)]
    ):
        return "mismatch", manifest, None, None
    try:
        class_rows = _read_eval_matrix(
            run_dir / "evaluation/class_il/eval_matrix.csv"
        )
        task_rows = _read_eval_matrix(
            run_dir / "evaluation/task_il/eval_matrix.csv"
        )
    except FileNotFoundError:
        return "incomplete", manifest, None, None
    expected = _expected_keys(int(manifest.get("num_tasks", 10)), joint=False)
    if not valid_calibration_manifest(
        run_dir / f"evaluation/calibration/fold_{int(fold)}.json",
        variant, fold, int(manifest.get("num_tasks", 10)),
    ):
        return "incomplete", manifest, None, None
    complete = (
        int(fold) in _complete_folds(class_rows, expected)
        and int(fold) in _complete_folds(task_rows, expected)
    )
    return (
        ("complete", manifest, class_rows, task_rows)
        if complete else ("incomplete", manifest, None, None)
    )


def collect(registry, memory_cache: dict | None = None) -> list[dict]:
    memory_cache = {} if memory_cache is None else memory_cache
    rows = []
    for variant in registry["variants"].values():
        for fold in range(10):
            status, manifest, class_rows, task_rows = _load_run_once(
                registry, variant, fold
            )
            row = {"variant_id": variant["id"], "fold": fold, "status": status}
            if status == "complete":
                num_tasks = int(manifest["num_tasks"])
                row.update(_sequential_fold_metrics(
                    _rows_by_key(class_rows, fold), _rows_by_key(task_rows, fold), num_tasks
                ))
                memory = _memory_from_manifest(manifest, variant)
                if variant["overrides"]["atlasv2_replay"] and not all(
                    math.isfinite(float(memory[field])) for field in MEMORY_FIELDS
                ):
                    memory = _memory_from_checkpoint_cache(
                        registry, variant, fold, memory_cache
                    )
                row.update(memory)
                run_dir = REPO_ROOT / "results" / experiment_desc(
                    registry, variant["id"], fold
                )
                diagnostic_path = run_dir / "evaluation/atlas_distribution_eval.csv"
                if diagnostic_path.is_file():
                    with diagnostic_path.open(newline="", encoding="utf-8") as handle:
                        diagnostic_rows = list(csv.DictReader(handle))
                    final_rows = [
                        value for value in diagnostic_rows
                        if int(value.get("fold", -1)) == fold
                        and int(value.get("after_task", -1)) == num_tasks - 1
                    ]
                    if final_rows:
                        for field in DISTRIBUTION_DIAGNOSTICS[:3]:
                            row[field] = float(final_rows[-1][field])
                accounting = manifest.get("distribution_accounting")
                if isinstance(accounting, Mapping):
                    for field in ("effective_lowrank_rank", "prototype_offset_norm"):
                        row[field] = float(accounting.get(field, math.nan))
                    row["distribution_memory_mib"] = float(
                        accounting.get("distribution_memory_bytes", math.nan)
                    ) / (1024.0 ** 2)
                    checkpoint = _latest_checkpoint(registry, variant, fold)
                    row["checkpoint_memory_mib"] = (
                        checkpoint.stat().st_size / (1024.0 ** 2)
                        if checkpoint is not None else math.nan
                    )
                    checkpoint = _latest_checkpoint(registry, variant, fold)
                    if checkpoint is not None:
                        row["checkpoint_memory_mib"] = checkpoint.stat().st_size / (1024.0 ** 2)
                calibration = manifest.get("calibration_manifest")
                if isinstance(calibration, list) and calibration:
                    selected = [
                        {
                            key: value for key, value in stage.items()
                            if key not in {
                                "validation_bacc", "candidate_count",
                                "locked_from_task",
                            }
                        }
                        for stage in calibration[-1].get("stages", [])
                    ]
                    row["selected_hyperparameters"] = json.dumps(
                        selected, sort_keys=True,
                        separators=(",", ":"),
                    )
            rows.append(row)
    return rows


def summarize(registry, rows: Sequence[dict]) -> list[dict]:
    output = []
    for variant in registry["variants"].values():
        complete = [
            row for row in rows
            if row["variant_id"] == variant["id"] and row["status"] == "complete"
        ]
        summary = {
            "variant_id": variant["id"], "completed_folds": len(complete),
            "status": "complete" if len(complete) == 10 else "incomplete",
        }
        for field in (*METRICS, *MEMORY_FIELDS):
            summary[f"{field}_mean"], summary[f"{field}_std"] = _mean_std(
                row.get(field) for row in complete
            )
        for field in DISTRIBUTION_DIAGNOSTICS:
            summary[f"{field}_mean"], summary[f"{field}_std"] = _mean_std(
                row.get(field) for row in complete
            )
        frequencies = Counter(
            row.get("selected_hyperparameters") for row in complete
            if row.get("selected_hyperparameters")
        )
        summary["selected_hyperparameter_frequency"] = json.dumps(
            dict(frequencies), sort_keys=True, separators=(",", ":")
        )
        output.append(summary)
    return output


def _format(row: Mapping[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean", math.nan)
    std = row.get(f"{metric}_std", math.nan)
    return "" if not math.isfinite(float(mean)) else f"{mean:.4f} ± {std:.4f}"


def markdown(summaries: Sequence[dict]) -> str:
    by_id = {row["variant_id"]: row for row in summaries}

    def table(title: str, ids: Sequence[str]) -> list[str]:
        lines = [
            f"# {title}", "",
            "| Setting | Folds | mACC | bACC | Masked bACC | BWT | FGT |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for variant_id in ids:
            row = by_id[variant_id]
            values = [_format(row, metric) for metric in METRICS]
            lines.append(f"| {variant_id} | {row['completed_folds']} | " + " | ".join(values) + " |")
        lines.append("")
        return lines

    lines = [
        "Primary classification metric: **bACC**. Primary stability metrics: **BWT / FGT**.",
        "Supporting metric: mACC. Diagnostic metric: Masked bACC.",
        "No winner is selected automatically.", "",
    ]
    lines.extend(table("ATLAS-v2 Additive Ladder", SETTING_IDS[:5]))
    lines.extend(table("ATLAS-v2 Frozen Baseline", (SETTING_IDS[5],)))
    lines.extend(table("ATLAS-v2 Semantic Extensions", (SETTING_IDS[4], SETTING_IDS[6], SETTING_IDS[7])))
    lines.extend(table("ATLAS-v2 LoRA Geometry Extension", (SETTING_IDS[8],)))
    lines.extend(table("ATLAS-v2 CoMEL LoRA Strategy", (COMEL_SETTING_ID,)))
    lines.extend(table("ATLAS-v2 Frozen Prototype Extension", (FROZEN_PROTO_LDA_ID,)))
    lines.extend(table("ATLAS Distribution Suite", DISTRIBUTION_SETTING_IDS))
    lines.extend([
        "# ATLAS Distribution Diagnostics", "",
        "| Setting | Cross-task error | Soft-task top-1 | Masked−Class bACC | Stats MiB |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant_id in DISTRIBUTION_SETTING_IDS:
        row = by_id[variant_id]
        values = [
            _format(row, field) for field in (
                "cross_task_error_rate", "soft_task_top1_accuracy",
                "masked_bacc_minus_class_il_bacc", "distribution_memory_mib",
            )
        ]
        lines.append(f"| {variant_id} | " + " | ".join(values) + " |")
    lines.append("")
    lines.extend(table("ATLAS-v2 Prototype LoRA × Replay Factorial", PROTO_FACTORIAL_IDS))
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "configs/atlas_v2_ablations.yaml"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results/ablations/atlas_v2/summary"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    output = Path(args.output).expanduser().resolve()
    cache_path = output / "replay_memory_cache.json"
    memory_cache = _read_memory_cache(cache_path)
    rows = collect(registry, memory_cache)
    summaries = summarize(registry, rows)
    per_fold_fields = [
        "variant_id", "fold", "status", *METRICS, *MEMORY_FIELDS,
        *DISTRIBUTION_DIAGNOSTICS, "selected_hyperparameters",
    ]
    summary_fields = [
        "variant_id", "completed_folds", "status",
        *[name for field in (*METRICS, *MEMORY_FIELDS) for name in (f"{field}_mean", f"{field}_std")],
        *[name for field in DISTRIBUTION_DIAGNOSTICS for name in (f"{field}_mean", f"{field}_std")],
        "selected_hyperparameter_frequency",
    ]
    _write_csv(output / "atlas_v2_per_fold.csv", rows, per_fold_fields)
    _write_csv(output / "atlas_v2_summary.csv", summaries, summary_fields)
    (output / "atlas_v2_tables.md").write_text(markdown(summaries), encoding="utf-8")
    _write_memory_cache(cache_path, memory_cache)
    incomplete = [row for row in summaries if row["status"] != "complete"]
    print(
        f"Wrote {len(SETTING_IDS)} ATLAS-v2 settings to {output}; "
        f"incomplete={len(incomplete)}"
    )
    return int(bool(args.strict and incomplete))


if __name__ == "__main__":
    raise SystemExit(main())
