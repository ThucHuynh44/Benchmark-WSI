"""Summarize the fold-isolated gated normalized-OAS transport sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_atlas_v3_acl_ablations import parse_folds
from scripts.run_atlas_v3_gated_transport_sweep import (
    RESULT_ROOT,
    SETTINGS,
    experiment_desc,
    inspect_run,
    select_setting_ids,
    select_settings,
)
from scripts.summarize_atlas_v3_acl_ablations import _fold_metrics


METRICS = ("mACC", "bACC", "masked_bACC", "BWT", "FGT", "auroc")


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
    finite = _finite(values)
    if not finite:
        return math.nan, math.nan
    return statistics.fmean(finite), statistics.pstdev(finite)


def _write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def collect(settings: Sequence[Mapping[str, Any]], folds: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for setting in settings:
        for fold in folds:
            status = inspect_run(setting, fold)
            row: dict[str, Any] = {
                "setting_id": setting["id"],
                "groups": ",".join(setting["groups"]),
                "fold": fold,
                "status": status,
                **setting["overrides"],
            }
            if status == "complete":
                run_dir = REPO_ROOT / "results" / experiment_desc(setting, fold)
                manifest_path = run_dir / "evaluation/class_il/run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                row.update(_fold_metrics(run_dir, fold, manifest))
            rows.append(row)
    return rows


def aggregate(settings: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    static = {
        int(row["fold"]): row
        for row in rows
        if row["setting_id"] == "static_control" and row["status"] == "complete"
    }
    output = []
    for setting in settings:
        selected = [
            row for row in rows
            if row["setting_id"] == setting["id"] and row["status"] == "complete"
        ]
        summary: dict[str, Any] = {
            "setting_id": setting["id"],
            "groups": ",".join(setting["groups"]),
            "completed_folds": len(selected),
            "folds_used": ",".join(str(row["fold"]) for row in selected),
            **setting["overrides"],
        }
        for metric in METRICS:
            mean, std = _mean_std(row.get(metric) for row in selected)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            deltas = [
                float(row[metric]) - float(static[int(row["fold"])][metric])
                for row in selected
                if int(row["fold"]) in static and metric in row
            ]
            summary[f"{metric}_delta_vs_static"] = (
                statistics.fmean(deltas) if deltas else math.nan
            )
        output.append(summary)
    return sorted(
        output,
        key=lambda row: (
            not math.isfinite(float(row["mACC_mean"])),
            -float(row["mACC_mean"]) if math.isfinite(float(row["mACC_mean"])) else 0.0,
        ),
    )


def markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# ATLAS-v3 gated transport sweep",
        "",
        "| Setting | Folds | mACC | Δ static | bACC | Masked bACC | BWT | FGT | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def cell(key: str) -> str:
            value = float(row[key])
            return "" if not math.isfinite(value) else f"{100.0 * value:.2f}"

        lines.append(
            f"| {row['setting_id']} | {row['completed_folds']} | "
            f"{cell('mACC_mean')} | {cell('mACC_delta_vs_static')} | "
            f"{cell('bACC_mean')} | {cell('masked_bACC_mean')} | "
            f"{cell('BWT_mean')} | {cell('FGT_mean')} | {cell('auroc_mean')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument(
        "--settings",
        default="",
        help="Comma- or colon-separated setting IDs; overrides --groups.",
    )
    parser.add_argument("--folds", default="0")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / RESULT_ROOT / "summary"),
    )
    args = parser.parse_args(argv)
    selected = select_setting_ids(args.settings) if args.settings else select_settings(args.groups)
    # Always include the exact static reference when it has already been run.
    static = next(setting for setting in SETTINGS if setting["id"] == "static_control")
    settings = [static, *[setting for setting in selected if setting["id"] != "static_control"]]
    folds = parse_folds(args.folds)
    rows = collect(settings, folds)
    summaries = aggregate(settings, rows)
    output = Path(args.output).expanduser().resolve()
    parameter_fields = list(BASE_PARAMETER_FIELDS)
    per_fold_fields = ["setting_id", "groups", "fold", "status", *parameter_fields, *METRICS]
    summary_fields = [
        "setting_id", "groups", "completed_folds", "folds_used", *parameter_fields,
        *[
            field
            for metric in METRICS
            for field in (f"{metric}_mean", f"{metric}_std", f"{metric}_delta_vs_static")
        ],
    ]
    _write(output / "sweep_per_fold.csv", rows, per_fold_fields)
    _write(output / "sweep_summary.csv", summaries, summary_fields)
    output.mkdir(parents=True, exist_ok=True)
    table = markdown(summaries)
    (output / "sweep_table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote sweep summary to {output}")
    return 0


BASE_PARAMETER_FIELDS = (
    "atlasv3_acl_mode",
    "atlasv3_acl_transport_rank",
    "atlasv3_acl_transport_ridge",
    "atlasv3_acl_transport_mean_scale",
    "atlasv3_acl_transport_cov_scale",
    "atlasv3_acl_coverage_energy",
    "atlasv3_acl_bootstrap_samples",
    "atlasv3_acl_uncertainty_beta",
)


if __name__ == "__main__":
    raise SystemExit(main())
