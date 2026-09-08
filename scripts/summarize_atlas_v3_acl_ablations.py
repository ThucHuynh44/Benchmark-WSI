"""Summarize fold-isolated ATLAS-v3 ACL ablation runs."""

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

from scripts.atlas_v3_acl_registry import load_registry
from scripts.build_cl_table import (
    _read_eval_matrix,
    _rows_by_key,
    _sequential_fold_metrics,
)
from scripts.run_atlas_v3_acl_ablations import experiment_desc, inspect_run


CL_METRICS = ("mACC", "bACC", "masked_bACC", "BWT", "FGT", "auroc")
RESOURCE_METRICS = (
    "training_time",
    "peak_gpu_allocated_mib",
    "peak_gpu_reserved_mib",
    "total_parameters",
    "trainable_parameters",
    "parameter_growth",
)
REFERENCE_ID = "atlasv3_acl"


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _canonical_fold_row(path: Path, fold: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        selected = [
            row for row in csv.DictReader(handle)
            if str(row.get("fold", "")).strip() == str(int(fold))
        ]
    if len(selected) != 1:
        raise ValueError(f"Expected one fold={fold} row in {path}, got {len(selected)}")
    return selected[0]


def _fold_metrics(run_dir: Path, fold: int, manifest: Mapping[str, Any]) -> Dict[str, float]:
    num_tasks = int(manifest["num_tasks"])
    class_rows = _read_eval_matrix(run_dir / "evaluation/class_il/eval_matrix.csv")
    task_rows = _read_eval_matrix(run_dir / "evaluation/task_il/eval_matrix.csv")
    metrics = _sequential_fold_metrics(
        _rows_by_key(class_rows, fold),
        _rows_by_key(task_rows, fold),
        num_tasks,
    )
    canonical = _canonical_fold_row(
        run_dir / "evaluation/class_il/per_fold_summary.csv", fold
    )
    metrics["auroc"] = float(canonical["auroc"])
    metrics["training_time"] = float(canonical["training_time"])
    resources = manifest.get("per_fold_resources", {}).get(str(fold), {})
    for field in RESOURCE_METRICS:
        if field != "training_time":
            metrics[field] = float(resources.get(field, math.nan))
    return metrics


def collect(registry: Dict[str, Any]) -> list[dict]:
    rows = []
    for variant in registry["variants"].values():
        for fold in range(10):
            status = inspect_run(registry, variant, fold)
            row = {
                "variant_id": variant["id"],
                "group": variant["group"],
                "label": variant["label"],
                "factor": variant["factor"],
                "value": variant["value"],
                "fold": fold,
                "status": status,
            }
            if status == "complete":
                run_dir = REPO_ROOT / "results" / experiment_desc(
                    registry, variant["id"], fold
                )
                with (run_dir / "evaluation/class_il/run_manifest.json").open(
                    encoding="utf-8"
                ) as handle:
                    manifest = json.load(handle)
                row.update(_fold_metrics(run_dir, fold, manifest))
            rows.append(row)
    return rows


def summarize(registry: Dict[str, Any], fold_rows: Sequence[dict]) -> list[dict]:
    reference = {
        int(row["fold"]): row
        for row in fold_rows
        if row["variant_id"] == REFERENCE_ID and row["status"] == "complete"
    }
    output = []
    for variant in registry["variants"].values():
        rows = [row for row in fold_rows if row["variant_id"] == variant["id"]]
        complete = [row for row in rows if row["status"] == "complete"]
        summary = {
            "variant_id": variant["id"],
            "group": variant["group"],
            "label": variant["label"],
            "factor": variant["factor"],
            "value": variant["value"],
            "completed_folds": len(complete),
            "folds_used": ",".join(str(row["fold"]) for row in complete),
            "status": "complete" if len(complete) == 10 else "incomplete",
        }
        for metric in (*CL_METRICS, *RESOURCE_METRICS):
            mean, std = _mean_std(row.get(metric) for row in complete)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            paired = [
                float(row[metric]) - float(reference[int(row["fold"])][metric])
                for row in complete
                if int(row["fold"]) in reference
                and metric in row
                and metric in reference[int(row["fold"])]
                and math.isfinite(float(row[metric]))
                and math.isfinite(float(reference[int(row["fold"])][metric]))
            ]
            delta_mean, delta_std = _mean_std(paired)
            summary[f"{metric}_delta_vs_acl_mean"] = delta_mean
            summary[f"{metric}_delta_vs_acl_std"] = delta_std
        output.append(summary)
    return output


def _cell(row: Mapping[str, Any], metric: str, scale: float) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    if not math.isfinite(mean):
        return ""
    return f"{mean * scale:.2f} ± {std * scale:.2f}"


def _markdown(rows: Sequence[Mapping[str, Any]], percent: bool) -> str:
    scale = 100.0 if percent else 1.0
    lines = [
        "# ATLAS-v3 ACL Ablation Summary",
        "",
        "| Setting | Folds | mACC | bACC | Masked bACC | BWT | FGT | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            _cell(row, metric, scale)
            for metric in ("mACC", "bACC", "masked_bACC", "BWT", "FGT", "auroc")
        ]
        lines.append(
            f"| {row['variant_id']} | {row['completed_folds']}/10 | "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", default=str(REPO_ROOT / "configs/atlas_v3_acl_ablations.yaml")
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "results/ablations/atlas_v3_acl/summary")
    )
    parser.add_argument("--percent", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    fold_rows = collect(registry)
    summaries = summarize(registry, fold_rows)
    output = Path(args.output).expanduser().resolve()

    fold_fields = [
        "variant_id", "group", "label", "factor", "value", "fold", "status",
        *CL_METRICS, *RESOURCE_METRICS,
    ]
    summary_fields = [
        "variant_id", "group", "label", "factor", "value",
        "completed_folds", "folds_used", "status",
        *[
            field
            for metric in (*CL_METRICS, *RESOURCE_METRICS)
            for field in (
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_delta_vs_acl_mean",
                f"{metric}_delta_vs_acl_std",
            )
        ],
    ]
    _write_csv(output / "ablation_per_fold.csv", fold_rows, fold_fields)
    _write_csv(output / "ablation_summary.csv", summaries, summary_fields)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ablation_table.md").write_text(
        _markdown(summaries, args.percent), encoding="utf-8"
    )

    incomplete = [row for row in summaries if row["status"] != "complete"]
    print(f"Wrote {len(summaries)} settings to {output}; incomplete={len(incomplete)}")
    return 1 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
