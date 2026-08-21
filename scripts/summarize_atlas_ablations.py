"""Audit and summarize fold-isolated ATLAS-MIL ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.atlas_ablation_registry import load_registry
from scripts.build_cl_table import (
    _read_eval_matrix,
    _rows_by_key,
    _sequential_fold_metrics,
)
from scripts.run_atlas_ablations import experiment_desc, inspect_run


CL_METRICS = ("mACC", "bACC", "masked_bACC", "BWT", "FGT", "auroc")
RESOURCE_METRICS = (
    "training_time", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
    "total_parameters", "trainable_parameters", "parameter_growth",
)
MECHANISM_METRICS = (
    "semantic_rho", "intra_class_distance", "inter_class_separation",
    "embedding_drift", "attention_drift", "old_current_overlap",
    "within_current_overlap", "all_seen_overlap", "mean_effective_rank",
    "memory_count",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


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


def _canonical_fold_row(path: Path, fold: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if str(row.get("fold")) == str(int(fold))]
    if len(selected) != 1:
        raise ValueError(f"Expected one canonical fold row in {path}, got {len(selected)}")
    return selected[0]


def _fold_metrics(run_dir: Path, fold: int, manifest: dict) -> Dict[str, float]:
    num_tasks = int(manifest["num_tasks"])
    class_rows = _read_eval_matrix(run_dir / "evaluation/class_il/eval_matrix.csv")
    task_rows = _read_eval_matrix(run_dir / "evaluation/task_il/eval_matrix.csv")
    metrics = _sequential_fold_metrics(
        _rows_by_key(class_rows, fold), _rows_by_key(task_rows, fold), num_tasks
    )
    canonical = _canonical_fold_row(
        run_dir / "evaluation/class_il/per_fold_summary.csv", fold
    )
    metrics["auroc"] = float(canonical["auroc"])
    metrics["training_time"] = float(canonical["training_time"])
    resources = manifest.get("per_fold_resources", {}).get(str(fold), {})
    for field in RESOURCE_METRICS:
        if field == "training_time":
            continue
        metrics[field] = float(resources.get(field, math.nan))
    return metrics


def collect(registry: Dict[str, Any]) -> tuple[list[dict], list[dict]]:
    fold_rows, mechanism_rows = [], []
    for variant in registry["variants"].values():
        for fold in range(10):
            status = inspect_run(registry, variant, fold)
            base = {
                "variant_id": variant["id"], "group": variant["group"],
                "label": variant["label"], "factor": variant["factor"],
                "value": variant["value"], "model": variant["model"],
                "fold": fold, "status": status,
            }
            if status != "complete":
                fold_rows.append(base)
                continue
            run_dir = REPO_ROOT / "results" / experiment_desc(registry, variant["id"], fold)
            manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            fold_rows.append({**base, **_fold_metrics(run_dir, fold, manifest)})
            diagnostics = run_dir / "evaluation/atlas_task_diagnostics.csv"
            if diagnostics.is_file():
                with diagnostics.open(newline="", encoding="utf-8") as handle:
                    for raw in csv.DictReader(handle):
                        mechanism_rows.append({
                            "variant_id": variant["id"], "group": variant["group"],
                            "label": variant["label"], "fold": fold,
                            **raw,
                        })
    return fold_rows, mechanism_rows


def summarize_folds(registry: Dict[str, Any], fold_rows: Sequence[dict]) -> list[dict]:
    full = {
        int(row["fold"]): row for row in fold_rows
        if row["variant_id"] == "full" and row["status"] == "complete"
    }
    summaries = []
    for variant in registry["variants"].values():
        rows = [row for row in fold_rows if row["variant_id"] == variant["id"]]
        complete = [row for row in rows if row["status"] == "complete"]
        output = {
            "variant_id": variant["id"], "group": variant["group"],
            "label": variant["label"], "factor": variant["factor"],
            "value": variant["value"], "model": variant["model"],
            "completed_folds": len(complete),
            "status": "complete" if len(complete) == 10 else "incomplete",
        }
        for metric in (*CL_METRICS, *RESOURCE_METRICS):
            mean, std = _mean_std(row.get(metric) for row in complete)
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            paired = [
                float(row[metric]) - float(full[int(row["fold"])][metric])
                for row in complete
                if int(row["fold"]) in full
                and metric in row and metric in full[int(row["fold"])]
                and math.isfinite(float(row[metric]))
                and math.isfinite(float(full[int(row["fold"])][metric]))
            ]
            delta_mean, delta_std = _mean_std(paired)
            output[f"{metric}_delta_vs_full_mean"] = delta_mean
            output[f"{metric}_delta_vs_full_std"] = delta_std
        summaries.append(output)
    return summaries


def summarize_mechanisms(rows: Sequence[dict]) -> list[dict]:
    output = []
    variants = sorted({str(row["variant_id"]) for row in rows})
    for variant_id in variants:
        selected = [row for row in rows if row["variant_id"] == variant_id]
        tasks = sorted({int(row["task"]) for row in selected})
        for task in [*tasks, "overall"]:
            task_rows = selected if task == "overall" else [
                row for row in selected if int(row["task"]) == task
            ]
            summary = {"variant_id": variant_id, "task": task}
            for metric in MECHANISM_METRICS:
                if task == "overall":
                    fold_macros = []
                    for fold in sorted({int(row["fold"]) for row in task_rows}):
                        values = _finite(
                            row.get(metric) for row in task_rows if int(row["fold"]) == fold
                        )
                        if values:
                            fold_macros.append(statistics.fmean(values))
                    mean, std = _mean_std(fold_macros)
                else:
                    mean, std = _mean_std(row.get(metric) for row in task_rows)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
            summary["fold_count"] = len({int(row["fold"]) for row in task_rows})
            output.append(summary)
    return output


def _markdown(registry: Dict[str, Any], summaries: Sequence[dict]) -> str:
    by_id = {row["variant_id"]: row for row in summaries}

    def table(title: str, ids: Sequence[str]) -> list[str]:
        lines = [f"## {title}", "", "| Variant | Folds | mACC | bACC | Masked bACC | BWT | FGT |", "|---|---:|---:|---:|---:|---:|---:|"]
        for variant_id in ids:
            row = by_id[variant_id]
            values = []
            for metric in ("mACC", "bACC", "masked_bACC", "BWT", "FGT"):
                mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
                values.append("" if not math.isfinite(float(mean)) else f"{mean:.4f} ± {std:.4f}")
            lines.append(
                f"| {variant_id} | {row['completed_folds']} | " + " | ".join(values) + " |"
            )
        lines.append("")
        return lines

    lines = ["# ATLAS-MIL Ablation Summary", ""]
    for group, title in (
        ("external_reference", "External references"),
        ("additive", "Additive ladder"),
    ):
        ids = [entry["id"] for entry in registry["variants"].values() if entry["group"] == group]
        lines.extend(table(title, ids))
    leave_one_out = [
        "full", "full_no_lora", "wo_replay", "wo_nce", "wo_reconstruction",
        "wo_manifold", "wo_attention", "solm_none", "solm_hard", "prompt_only",
        "centroid_with_prompt_fallback",
    ]
    lines.extend(table("Leave-one-out", leave_one_out))
    lines.extend(table(
        "Attention-weight pilot",
        ["atlas_ce", "att_w025", "att_w05", "add_attention"],
    ))
    for axis, members in registry["axis_members"].items():
        lines.extend(table(f"Sweep: {axis}", [member["variant_id"] for member in members]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "configs/atlas_mil_ablations.yaml"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results/ablations/atlas_mil/summary"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    fold_rows, mechanism_rows = collect(registry)
    summaries = summarize_folds(registry, fold_rows)
    mechanism_summary = summarize_mechanisms(mechanism_rows)
    output = Path(args.output).expanduser().resolve()
    fold_fields = [
        "variant_id", "group", "label", "factor", "value", "model", "fold", "status",
        *CL_METRICS, *RESOURCE_METRICS,
    ]
    summary_fields = [
        "variant_id", "group", "label", "factor", "value", "model",
        "completed_folds", "status",
        *[
            field for metric in (*CL_METRICS, *RESOURCE_METRICS)
            for field in (
                f"{metric}_mean", f"{metric}_std",
                f"{metric}_delta_vs_full_mean", f"{metric}_delta_vs_full_std",
            )
        ],
    ]
    mechanism_fields = [
        "variant_id", "task", "fold_count",
        *[field for metric in MECHANISM_METRICS for field in (f"{metric}_mean", f"{metric}_std")],
    ]
    raw_mechanism_fields = sorted({key for row in mechanism_rows for key in row})
    _write_csv(output / "ablation_per_fold.csv", fold_rows, fold_fields)
    _write_csv(output / "ablation_summary.csv", summaries, summary_fields)
    _write_csv(output / "mechanism_per_task.csv", mechanism_rows, raw_mechanism_fields)
    _write_csv(output / "mechanism_summary.csv", mechanism_summary, mechanism_fields)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_tables.md").write_text(_markdown(registry, summaries), encoding="utf-8")
    incomplete = [row for row in summaries if row["status"] != "complete"]
    print(f"Wrote {len(summaries)} variants to {output}; incomplete={len(incomplete)}")
    return 1 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
