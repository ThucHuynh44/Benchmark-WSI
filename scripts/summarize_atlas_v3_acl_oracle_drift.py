"""Export per-class oracle drift diagnostics and compact task-level figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "results/ablations/atlas_v3_acl/atlasv3_acl_oas_oracle"
)

METRICS = (
    "mean_drift_cosine",
    "mean_residual_ungated_cosine",
    "mean_residual_gated_cosine",
    "mean_error_ungated_l2",
    "mean_error_gated_l2",
    "mean_error_ungated_relative_l2",
    "mean_error_gated_relative_l2",
    "covariance_drift_frobenius",
    "covariance_residual_ungated_frobenius",
    "covariance_residual_gated_frobenius",
    "covariance_drift_relative_frobenius",
    "covariance_residual_ungated_relative_frobenius",
    "covariance_residual_gated_relative_frobenius",
    "mean_gain_ungated",
    "mean_gain_gated",
    "covariance_gain_ungated",
    "covariance_gain_gated",
)
CLASS_FIELDS = (
    "fold",
    "after_task",
    "class_id",
    "origin_task",
    "class_age",
    "sample_count",
    "coverage",
    "step_gate",
    "bootstrap_uncertainty",
    *METRICS,
)


def _fold_from_path(path: Path) -> int:
    for parent in path.parents:
        if parent.name.startswith("fold_"):
            return int(parent.name.split("_", 1)[1])
    raise ValueError(f"Cannot infer fold from {path}")


def collect(run_root: Path) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    complete_folds = []
    manifests = sorted(run_root.glob("fold_*/evaluation/class_il/run_manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("ablation_id") != "atlasv3_acl_oas_oracle":
            raise ValueError(f"Unexpected ablation in {manifest_path}")
        fold = _fold_from_path(manifest_path)
        accounting = manifest.get("transport_accounting", {})
        history = accounting.get("history", [])
        num_tasks = int(manifest.get("num_tasks", 0))
        if len(history) != num_tasks:
            continue
        expected_diagnostic_tasks = max(num_tasks - 1, 0)
        actual_diagnostic_tasks = sum(
            bool(task_row.get("oracle_class_diagnostics")) for task_row in history
        )
        if actual_diagnostic_tasks != expected_diagnostic_tasks:
            continue
        complete_folds.append(fold)
        for task_row in history:
            for class_row in task_row.get("oracle_class_diagnostics", []):
                rows.append({
                    field: fold if field == "fold" else class_row.get(field, "")
                    for field in CLASS_FIELDS
                })
    return rows, sorted(set(complete_folds))


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


def aggregate(rows: Sequence[dict]) -> list[dict]:
    output = []
    tasks = sorted({int(row["after_task"]) for row in rows})
    for task in tasks:
        selected = [row for row in rows if int(row["after_task"]) == task]
        item: Dict[str, Any] = {
            "after_task": task,
            "task_number": task + 1,
            "classes_across_folds": len(selected),
            "folds": len({int(row["fold"]) for row in selected}),
        }
        for metric in METRICS:
            values = _finite(row.get(metric) for row in selected)
            item[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
        output.append(item)
    return output


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[dict], folds: Sequence[int]) -> None:
    lines = [
        "# ATLAS-v3 ACL Oracle Drift",
        "",
        f"Folds: {', '.join(map(str, folds))}",
        "",
        "The probe measures one-step transport starting from oracle-aligned statistics at the previous task.",
        "",
        "Positive gain means transport reduced error relative to leaving historical statistics unchanged.",
        "",
        "| After task | Mean drift | Ungated residual | Gated residual | Ungated gain | Gated gain | Cov drift (rel.) | Cov ungated residual | Cov gated residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def show(row: dict, key: str) -> str:
        value = row.get(key, "")
        return "" if value == "" else f"{float(value):.6f}"

    for row in rows:
        lines.append(
            "| {task} | {drift} | {ungated} | {gated} | {ugain} | {ggain} | {covdrift} | {covungated} | {covgated} |".format(
                task=row["task_number"],
                drift=show(row, "mean_drift_cosine_mean"),
                ungated=show(row, "mean_residual_ungated_cosine_mean"),
                gated=show(row, "mean_residual_gated_cosine_mean"),
                ugain=show(row, "mean_gain_ungated_mean"),
                ggain=show(row, "mean_gain_gated_mean"),
                covdrift=show(row, "covariance_drift_relative_frobenius_mean"),
                covungated=show(row, "covariance_residual_ungated_relative_frobenius_mean"),
                covgated=show(row, "covariance_residual_gated_relative_frobenius_mean"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(output: Path, rows: Sequence[dict]) -> None:
    matplotlib_cache = output / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV and Markdown outputs were still written")
        return
    tasks = [int(row["task_number"]) for row in rows]
    panels = (
        (
            "Mean drift and transport residual",
            "Cosine distance",
            (
                ("No correction", "mean_drift_cosine_mean"),
                ("Ungated transport", "mean_residual_ungated_cosine_mean"),
                ("Gated transport", "mean_residual_gated_cosine_mean"),
            ),
        ),
        (
            "Covariance drift and transport residual",
            "Relative Frobenius error",
            (
                ("No correction", "covariance_drift_relative_frobenius_mean"),
                ("Ungated transport", "covariance_residual_ungated_relative_frobenius_mean"),
                ("Gated transport", "covariance_residual_gated_relative_frobenius_mean"),
            ),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, (title, ylabel, series) in zip(axes, panels):
        for label, key in series:
            axis.plot(tasks, [float(row[key]) for row in rows], marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel("Task")
        axis.set_ylabel(ylabel)
        axis.set_xticks(tasks)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output / "oracle_drift_transport_residual.png", dpi=200)
    figure.savefig(output / "oracle_drift_transport_residual.pdf")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RUN_ROOT.parent / "summary_oas_oracle_drift"),
    )
    parser.add_argument("--strict", action="store_true", help="Require all ten folds")
    args = parser.parse_args(argv)
    run_root = Path(args.run_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    class_rows, folds = collect(run_root)
    if args.strict and folds != list(range(10)):
        raise RuntimeError(f"Expected folds 0..9, found {folds}")
    if not class_rows:
        raise RuntimeError(f"No oracle class diagnostics found under {run_root}")
    task_rows = aggregate(class_rows)
    task_fields = list(task_rows[0])
    _write_csv(output / "oracle_drift_per_class.csv", class_rows, CLASS_FIELDS)
    _write_csv(output / "oracle_drift_by_task.csv", task_rows, task_fields)
    _write_markdown(output / "oracle_drift_table.md", task_rows, folds)
    _plot(output, task_rows)
    print(f"Wrote {len(class_rows)} class rows across {len(folds)} folds to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
