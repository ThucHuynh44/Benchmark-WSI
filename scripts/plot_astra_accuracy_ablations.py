"""Plot ASTRA ACL-epoch and transport-rank ablations using accuracy only.

For each fold and training stage t, the script computes micro accuracy over
all tasks seen at that stage from the ``acc`` and ``n`` columns in the
Class-IL evaluation matrix.  Mean Accuracy (mACC) is the arithmetic mean of
these stage accuracies.  The plotted points are the mean and population
standard deviation of mACC over folds.

Edit only the USER CONFIGURATION block to change variants or plot styling.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchmarkwsi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/benchmarkwsi-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER CONFIGURATION
# =============================================================================
REPO_ROOT = Path(
    "/datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/version_moi/Benchmark-WSI"
)
RUN_ROOT = REPO_ROOT / "results/ablations/atlas_v3_acl"
OUTPUT_DIR = REPO_ROOT / "results/figures/astra_accuracy_ablations"

BASE_VARIANT = "atlasv3_acl_gated_transport_normalized_oas_no_histneg"

# Epoch 1 and rank 8 use BASE_VARIANT; they do not have an explicit suffix.
EPOCH_VARIANTS: dict[int, str] = {
    1: BASE_VARIANT,
    2: f"{BASE_VARIANT}_e2",
    3: f"{BASE_VARIANT}_e3",
    4: f"{BASE_VARIANT}_e4",
}

RANK_VARIANTS: dict[int, str] = {
    2: f"{BASE_VARIANT}_r2",
    4: f"{BASE_VARIANT}_r4",
    8: BASE_VARIANT,
    16: f"{BASE_VARIANT}_r16",
}

EXPECTED_FOLDS = tuple(range(10))
EXPECTED_TASKS = 10
STRICT_COMPLETE = True
SCALE_TO_PERCENT = True

DPI = 300
FIGSIZE_SINGLE = (6.0, 4.6)
FIGSIZE_COMBINED = (11.5, 4.6)
LINE_WIDTH = 1.8
BAR_WIDTH = 0.64
BAR_ALPHA = 0.88
ERROR_CAP_SIZE = 4.0
Y_PADDING = 0.35  # percentage points when SCALE_TO_PERCENT=True

EPOCH_COLOR = "#CC79A7"
RANK_COLOR = "#0072B2"
# =============================================================================
# END USER CONFIGURATION
# =============================================================================


REQUIRED_COLUMNS = {"fold", "after_task", "eval_task", "acc", "n"}


def _evaluation_path(variant: str, fold: int) -> Path:
    return (
        RUN_ROOT
        / variant
        / f"fold_{fold}"
        / "evaluation/class_il/eval_matrix.csv"
    )


def _read_fold_macc(path: Path, expected_fold: int) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for raw in reader:
            fold = int(raw["fold"])
            if fold != expected_fold:
                raise ValueError(
                    f"{path} contains fold={fold}, expected fold={expected_fold}"
                )
            rows.append(
                {
                    "after_task": int(raw["after_task"]),
                    "eval_task": int(raw["eval_task"]),
                    "acc": float(raw["acc"]),
                    "n": int(raw["n"]),
                }
            )

    grouped: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in rows:
        after_task = int(row["after_task"])
        eval_task = int(row["eval_task"])
        if eval_task <= after_task:
            grouped[after_task].append(row)

    expected_stages = set(range(EXPECTED_TASKS))
    if set(grouped) != expected_stages:
        raise ValueError(
            f"{path} has stages {sorted(grouped)}, expected {sorted(expected_stages)}"
        )

    stage_accuracies = []
    for after_task in range(EXPECTED_TASKS):
        stage_rows = grouped[after_task]
        observed_eval_tasks = {int(row["eval_task"]) for row in stage_rows}
        expected_eval_tasks = set(range(after_task + 1))
        if observed_eval_tasks != expected_eval_tasks:
            raise ValueError(
                f"{path}: after_task={after_task} has eval tasks "
                f"{sorted(observed_eval_tasks)}, expected {sorted(expected_eval_tasks)}"
            )
        total_n = sum(int(row["n"]) for row in stage_rows)
        if total_n <= 0:
            raise ValueError(f"{path}: after_task={after_task} has no samples")
        accuracy = sum(
            float(row["acc"]) * int(row["n"]) for row in stage_rows
        ) / total_n
        stage_accuracies.append(accuracy)

    return float(np.mean(stage_accuracies))


def _collect(
    group: str, variants: Mapping[int, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_fold: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for value, variant in variants.items():
        observed = []
        missing = []
        for fold in EXPECTED_FOLDS:
            path = _evaluation_path(variant, fold)
            if not path.is_file():
                missing.append(fold)
                continue
            macc = _read_fold_macc(path, fold)
            observed.append(macc)
            per_fold.append(
                {
                    "group": group,
                    "value": value,
                    "variant": variant,
                    "fold": fold,
                    "mACC": macc,
                    "mACC_percent": 100.0 * macc,
                }
            )

        if missing and STRICT_COMPLETE:
            raise FileNotFoundError(
                f"{group}={value} ({variant}) is missing folds: {missing}"
            )
        if not observed:
            raise RuntimeError(f"No completed folds for {group}={value}: {variant}")

        values = np.asarray(observed, dtype=float)
        sample_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * sample_std / math.sqrt(len(values))
        summary.append(
            {
                "group": group,
                "value": value,
                "variant": variant,
                "folds": len(values),
                "mACC_mean": float(values.mean()),
                "mACC_std": float(values.std(ddof=0)),
                "mACC_mean_percent": float(100.0 * values.mean()),
                "mACC_std_percent": float(100.0 * values.std(ddof=0)),
                "mACC_ci95": ci95,
                "mACC_ci95_percent": 100.0 * ci95,
            }
        )

    return per_fold, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _axis_limits(means: np.ndarray, bands: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(means - bands))
    upper = float(np.max(means + bands))
    padding = Y_PADDING if SCALE_TO_PERCENT else Y_PADDING / 100.0
    if math.isclose(lower, upper):
        padding = max(padding, 0.01)
    return lower - padding, upper + padding


def _draw(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_label: str,
    color: str,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["value"]))
    tick_labels = [int(row["value"]) for row in ordered]
    x = np.arange(len(ordered), dtype=float)
    mean_key = "mACC_mean_percent" if SCALE_TO_PERCENT else "mACC_mean"
    ci_key = "mACC_ci95_percent" if SCALE_TO_PERCENT else "mACC_ci95"
    means = np.asarray([float(row[mean_key]) for row in ordered])
    ci95 = np.asarray([float(row[ci_key]) for row in ordered])

    ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        color=color,
        alpha=BAR_ALPHA,
        edgecolor=color,
        linewidth=0.9,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=ci95,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.25,
        capsize=ERROR_CAP_SIZE,
        capthick=1.25,
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(
        "Mean accuracy (mACC, %) ↑" if SCALE_TO_PERCENT else "Mean accuracy (mACC) ↑"
    )
    ax.set_xticks(x, tick_labels)
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_ylim(*_axis_limits(means, ci95))
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_single(
    rows: list[dict[str, Any]],
    *,
    stem: str,
    title: str,
    x_label: str,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _draw(ax, rows, title=title, x_label=x_label, color=color)
    fig.text(
        0.5,
        0.01,
        "Error bars denote the 95% confidence interval over folds.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epoch_fold, epoch_summary = _collect("acl_epochs", EPOCH_VARIANTS)
    rank_fold, rank_summary = _collect("transport_rank", RANK_VARIANTS)
    per_fold = epoch_fold + rank_fold
    summaries = epoch_summary + rank_summary

    _write_csv(OUTPUT_DIR / "accuracy_ablation_per_fold.csv", per_fold)
    _write_csv(OUTPUT_DIR / "accuracy_ablation_summary.csv", summaries)

    _save_single(
        epoch_summary,
        stem="acl_epochs_accuracy",
        title="Effect of ACL Adaptation Epochs",
        x_label="Number of ACL epochs",
        color=EPOCH_COLOR,
    )
    _save_single(
        rank_summary,
        stem="transport_rank_accuracy",
        title="Effect of Transport Rank",
        x_label="Transport rank",
        color=RANK_COLOR,
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_COMBINED)
    _draw(
        axes[0],
        epoch_summary,
        title="(a) ACL Adaptation Epochs",
        x_label="Number of ACL epochs",
        color=EPOCH_COLOR,
    )
    _draw(
        axes[1],
        rank_summary,
        title="(b) Transport Rank",
        x_label="Transport rank",
        color=RANK_COLOR,
    )
    fig.suptitle(
        "ASTRA: accuracy sensitivity to adaptation epochs and transport rank",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Mean accuracy is computed exclusively from Class-IL accuracy; error bars show 95% confidence intervals.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"acl_epochs_and_transport_rank_accuracy.{suffix}",
            dpi=DPI,
            bbox_inches="tight",
        )
    plt.close(fig)

    for row in summaries:
        print(
            f"{row['group']}={row['value']}: "
            f"mACC={row['mACC_mean_percent']:.2f} ± "
            f"{row['mACC_std_percent']:.2f} ({row['folds']} folds)"
        )
    print(f"Wrote figures and CSV files to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
