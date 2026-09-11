"""Summarize paired ATLAS-v3 oracle transport-correction diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "results/diagnostics/atlas_v3_transport_correction"
METHODS = ("static", "ungated", "gated")
BASE_METRICS = (
    "prototype_cosine_distance",
    "prototype_l2",
    "mean_relative_error",
    "covariance_relative_frobenius",
    "covariance_bures",
)
TRANSPORT_METRICS = (
    "prototype_gain",
    "recovery_rate",
    "overcorrected",
    "direction_alignment",
    "step_magnitude_ratio",
    "covariance_gain",
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


def _mean(values: Iterable[Any]) -> float:
    values = _finite(values)
    return statistics.fmean(values) if values else math.nan


def _mean_std(values: Iterable[Any]) -> tuple[float, float]:
    values = _finite(values)
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values)


def _read_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in sorted(root.glob("fold_*/task_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for name in ("fold", "after_task", "eval_task", "class_id", "class_age"):
                    row[name] = int(row[name])
                key = tuple(row[name] for name in ("fold", "after_task", "eval_task", "class_id"))
                if key in seen:
                    raise ValueError(f"Duplicate diagnostic row {key}")
                seen.add(key)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No fold_*/task_*.csv diagnostics under {root}")
    return rows


def _fold_means(rows: Sequence[Mapping[str, Any]], field: str) -> dict[int, float]:
    return {
        fold: _mean(row.get(field) for row in rows if int(row["fold"]) == fold)
        for fold in sorted({int(row["fold"]) for row in rows})
    }


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        summary: dict[str, Any] = {
            "method": method,
            "folds": len({int(row["fold"]) for row in rows}),
            "class_time_rows": len(rows),
        }
        metrics = list(BASE_METRICS)
        if method != "static":
            metrics.extend(TRANSPORT_METRICS)
        for metric in metrics:
            field = f"{method}_{metric}"
            per_fold = _fold_means(rows, field)
            mean, std = _mean_std(per_fold.values())
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        output.append(summary)
    return output


def _grouped(rows: Sequence[Mapping[str, Any]], group: str) -> list[dict[str, Any]]:
    output = []
    for value in sorted({int(row[group]) for row in rows}):
        selected = [row for row in rows if int(row[group]) == value]
        for method in METHODS:
            row: dict[str, Any] = {
                group: value,
                "method": method,
                "folds": len({int(item["fold"]) for item in selected}),
                "class_time_rows": len(selected),
            }
            metrics = list(BASE_METRICS)
            if method != "static":
                metrics.extend(TRANSPORT_METRICS)
            for metric in metrics:
                field = f"{method}_{metric}"
                per_fold = _fold_means(selected, field)
                row[f"{metric}_mean"] = _mean(per_fold.values())
            output.append(row)
    return output


def _paired_pvalues(values: Sequence[float]) -> tuple[float, float]:
    values = _finite(values)
    if len(values) < 2:
        return math.nan, math.nan
    try:
        from scipy.stats import ttest_1samp, wilcoxon

        t_p = float(ttest_1samp(values, popmean=0.0).pvalue)
        if all(abs(value) <= 1.0e-15 for value in values):
            w_p = 1.0
        else:
            w_p = float(wilcoxon(values).pvalue)
        return t_p, w_p
    except (ImportError, ValueError):
        return math.nan, math.nan


def _paired_tests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        ("ungated_vs_static", "static", "ungated"),
        ("gated_vs_static", "static", "gated"),
        ("gated_vs_ungated", "ungated", "gated"),
    )
    output = []
    for name, baseline, candidate in comparisons:
        for metric in ("prototype_cosine_distance", "covariance_relative_frobenius"):
            base = _fold_means(rows, f"{baseline}_{metric}")
            proposed = _fold_means(rows, f"{candidate}_{metric}")
            folds = sorted(set(base) & set(proposed))
            # Both metrics are errors, so positive improvement means lower is better.
            differences = [base[fold] - proposed[fold] for fold in folds]
            t_p, w_p = _paired_pvalues(differences)
            mean, std = _mean_std(differences)
            output.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "folds": len(folds),
                    "improvement_mean": mean,
                    "improvement_std": std,
                    "wins": sum(value > 0.0 for value in differences),
                    "ties": sum(value == 0.0 for value in differences),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                }
            )
    return output


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _cell(row: Mapping[str, Any], metric: str) -> str:
    mean = float(row.get(f"{metric}_mean", math.nan))
    std = float(row.get(f"{metric}_std", math.nan))
    return "" if not math.isfinite(mean) else f"{mean:.4f} ± {std:.4f}"


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# ATLAS-v3 Transport-Correction Oracle Diagnostic",
        "",
        "Only old classes (`class_age > 0`) are included. Values are mean ± SD across fold means.",
        "",
        "| Method | Folds | Prototype residual ↓ | Mean relative error ↓ | Covariance residual ↓ | Recovery ↑ | Direction ↑ | Over-correction ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['folds']} | "
            f"{_cell(row, 'prototype_cosine_distance')} | "
            f"{_cell(row, 'mean_relative_error')} | "
            f"{_cell(row, 'covariance_relative_frobenius')} | "
            f"{_cell(row, 'recovery_rate')} | "
            f"{_cell(row, 'direction_alignment')} | "
            f"{_cell(row, 'overcorrected')} |"
        )
    lines.extend(
        [
            "",
            "Positive recovery/gain means transport moved stored statistics closer to the oracle; over-correction is the fraction moved farther away.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.input).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source / "summary"
    )
    rows = _read_rows(source)
    if any(int(row["class_age"]) <= 0 for row in rows):
        raise ValueError("Diagnostic input unexpectedly contains non-historical classes")
    summaries = _summaries(rows)
    by_age = _grouped(rows, "class_age")
    by_task = _grouped(rows, "after_task")
    tests = _paired_tests(rows)
    _write(output / "transport_correction_per_class.csv", rows)
    _write(output / "transport_correction_summary.csv", summaries)
    _write(output / "transport_correction_by_age.csv", by_age)
    _write(output / "transport_correction_by_task.csv", by_task)
    _write(output / "transport_correction_paired_tests.csv", tests)
    output.mkdir(parents=True, exist_ok=True)
    (output / "transport_correction_table.md").write_text(
        _markdown(summaries), encoding="utf-8"
    )
    observed_folds = {int(row["fold"]) for row in rows}
    observed_pairs = {(int(row["fold"]), int(row["after_task"])) for row in rows}
    expected_pairs = {(fold, task) for fold in range(10) for task in range(1, 10)}
    complete = observed_folds == set(range(10)) and observed_pairs == expected_pairs
    print(
        f"Wrote paired oracle diagnostic to {output}; "
        f"folds={len(observed_folds)} task-checkpoints={len(observed_pairs)}/90"
    )
    return 1 if args.strict and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
