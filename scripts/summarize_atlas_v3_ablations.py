"""Summarize selected fold-isolated ATLAS-v3 frozen settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.atlas_v3_registry import load_registry, select_variants
from scripts.run_atlas_v3_ablations import experiment_desc, inspect_run
from scripts.summarize_atlas_v3_acl_ablations import (
    CL_METRICS,
    RESOURCE_METRICS,
    _fold_metrics,
    _mean_std,
    _write_csv,
)


REFERENCE_ID = "atlasv3_frozen_proto"


def collect(registry: Dict[str, Any], variants: Sequence[dict]) -> list[dict]:
    rows = []
    for variant in variants:
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


def summarize(variants: Sequence[dict], fold_rows: Sequence[dict]) -> list[dict]:
    reference = {
        int(row["fold"]): row
        for row in fold_rows
        if row["variant_id"] == REFERENCE_ID and row["status"] == "complete"
    }
    output = []
    for variant in variants:
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
            summary[f"{metric}_delta_vs_frozen_proto_mean"] = delta_mean
            summary[f"{metric}_delta_vs_frozen_proto_std"] = delta_std
        output.append(summary)
    return output


def _metric_cell(row: Mapping[str, Any], metric: str, scale: float) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return "" if not math.isfinite(mean) else f"{mean * scale:.2f} ± {std * scale:.2f}"


def _markdown(rows: Sequence[Mapping[str, Any]], percent: bool) -> str:
    scale = 100.0 if percent else 1.0
    lines = [
        "# ATLAS-v3 Frozen Summary",
        "",
        "| Setting | Folds | mACC | bACC | Masked bACC | BWT | FGT | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            _metric_cell(row, metric, scale)
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
        "--registry", default=str(REPO_ROOT / "configs/atlas_v3_ablations.yaml")
    )
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "results/ablations/atlas_v3/summary")
    )
    parser.add_argument("--percent", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    variants = select_variants(registry, args.variants)
    fold_rows = collect(registry, variants)
    summaries = summarize(variants, fold_rows)
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
                f"{metric}_delta_vs_frozen_proto_mean",
                f"{metric}_delta_vs_frozen_proto_std",
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
