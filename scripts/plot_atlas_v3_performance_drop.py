"""Plot per-task Class-IL accuracy trajectories (performance drop).

Each panel fixes one evaluation task.  Its curves start when that task is
learned and continue until the end of the continual-learning sequence.  This
makes forgetting visible instead of collapsing the whole sequence into one
average or last-accuracy number.

Edit only the USER CONFIGURATION block to add/remove methods or change the
appearance of the figure.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchmarkwsi-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/benchmarkwsi-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# USER CONFIGURATION: edit values in this block only.
# =============================================================================
REPO_ROOT = Path(
    "/datastore/uittogether/LuuTru/Thuchd/benchmarkWSI/version_moi/Benchmark-WSI"
)
OUTPUT_DIR = REPO_ROOT / "results/figures/atlas_v3_performance_drop"

EXPECTED_FOLDS = tuple(range(10))
EXPECTED_TASKS = 10
ACCURACY_FIELD = "acc"  # Use raw classification accuracy from eval_matrix.csv.
STRICT_COMPLETE = True
SCALE_TO_PERCENT = True
UNCERTAINTY = "std"  # "std", "ci95", or "none"

TASKS_PER_ROW = 5
FIRST_TASKS_TO_PLOT = 6
FIRST_TASKS_PER_ROW = 3
FIG_WIDTH_PER_TASK = 3.05
FIG_HEIGHT_PER_ROW = 2.85
DPI = 300
TITLE = "Per-task Performance Across the Continual Learning Sequence"
X_LABEL = "Task learned"
Y_LABEL = "Accuracy (%)" if SCALE_TO_PERCENT else "Accuracy"
Y_LIMITS = (0.0, 100.0) if SCALE_TO_PERCENT else (0.0, 1.0)
GRID_ALPHA = 0.28
BAND_ALPHA = 0.10
LINE_WIDTH = 1.45
PROPOSED_LINE_WIDTH = 2.55
MARKER_SIZE = 3.8
LEGEND_COLUMNS = 5
ANNOTATE_OURS_DROP = False  # True: print initial-minus-final drop in each panel.

# ``source`` is a path or glob relative to REPO_ROOT. Toggle ``enabled`` to
# show/hide a curve. Add another dictionary to add a new method.
METHODS: list[dict[str, Any]] = [
    {
        "enabled": True,
        "label": "SGD",
        "source": "results/forward/sgd_feather_nobuffer_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#9E9E9E",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "enabled": True,
        "label": "EWC",
        "source": "results/forward/ewc_on_feather_nobuffer_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#8C6D31",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "enabled": True,
        "label": "LwF",
        "source": "results/forward/lwf_feather_nobuffer_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#9467BD",
        "marker": "^",
        "linestyle": "--",
    },
    {
        "enabled": True,
        "label": "A-GEM (buffer 30)",
        "source": "results/forward/agem_feather_buffer30_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#17BECF",
        "marker": "D",
        "linestyle": "-.",
    },
    {
        "enabled": True,
        "label": "DER++ (buffer 30)",
        "source": "results/forward/derpp_feather_buffer30_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#1F77B4",
        "marker": "v",
        "linestyle": "-.",
    },
    {
        "enabled": True,
        "label": "ER-ACE (buffer 30)",
        "source": "results/forward/er_ace_feather_buffer30_10tasks_forward/evaluation/class_il/eval_matrix.csv",
        "color": "#2CA02C",
        "marker": "P",
        "linestyle": "-.",
    },
    {
        "enabled": True,
        "label": "LWSR (buffer 30)",
        "source": "results/forward/lwsr_feather_buffer30_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#E377C2",
        "marker": "X",
        "linestyle": "-.",
    },
    {
        "enabled": True,
        "label": "MICIL (buffer 30)",
        "source": "results/forward/micil_feather_buffer30_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#FF7F0E",
        "marker": "h",
        "linestyle": "-.",
    },
    {
        # Disabled: folds 1..9 in this file use a reverse task order, so their
        # per-task trajectories are not comparable with the fixed order here.
        "enabled": False,
        "label": "AMIL (buffer 30)",
        "source": "results/forward/amil_feather_buffer30_10tasks_forward/evaluation/class_il/eval_matrix.csv",
        "color": "#BCBD22",
        "marker": "<",
        "linestyle": ":",
    },
    {
        "enabled": True,
        "label": "ATLAS-MIL (buffer 30)",
        "source": "results/forward/atlas_mil_feather_buffer30_10tasks_forward/evaluation/class_il/eval_matrix.csv",
        "color": "#7F7F7F",
        "marker": ">",
        "linestyle": ":",
    },
    {
        "enabled": True,
        "label": "ATLAS-v3 (ours)",
        "source": (
            "results/ablations/atlas_v3_acl/"
            "atlasv3_acl_gated_transport_normalized_oas_no_histneg/"
            "fold_*/evaluation/class_il/eval_matrix.csv"
        ),
        "color": "#D62728",
        "marker": "*",
        "linestyle": "-",
        "proposed": True,
        "zorder": 20,
    },
    # Ready-to-enable examples:
    {
        "enabled": False,
        "label": "OWLoRA",
        "source": "results/forward/owlora_feather_nobuffer_10tasks_forward/evaluation/class_il/eval_matrix.csv",
        "color": "#AEC7E8",
        "marker": "d",
        "linestyle": ":",
    },
    {
        "enabled": False,
        "label": "DER++ (buffer 10)",
        "source": "results/forward/derpp_feather_buffer10_10tasks/evaluation/class_il/eval_matrix.csv",
        "color": "#6BAED6",
        "marker": "v",
        "linestyle": ":",
    },
]
# =============================================================================
# END USER CONFIGURATION
# =============================================================================


REQUIRED_COLUMNS = {
    "fold",
    "after_task",
    "eval_task",
    "task_name",
    ACCURACY_FIELD,
}


def _source_paths(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_absolute():
        anchor = Path(path.anchor)
        relative = str(path)[len(path.anchor) :]
        matches = sorted(anchor.glob(relative))
    else:
        matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No evaluation matrix matches: {pattern}")
    return matches


def _read_rows(paths: Iterable[Path], label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for raw in reader:
                fold = int(raw["fold"])
                after_task = int(raw["after_task"])
                eval_task = int(raw["eval_task"])
                key = (fold, after_task, eval_task)
                if key in seen:
                    raise ValueError(f"Duplicate evaluation row for {label}: {key}")
                seen.add(key)
                if eval_task > after_task:
                    continue
                output.append(
                    {
                        "method": label,
                        "fold": fold,
                        "after_task": after_task,
                        "task_number": after_task + 1,
                        "eval_task": eval_task,
                        "task_name": str(raw["task_name"]),
                        "accuracy": float(raw[ACCURACY_FIELD]),
                    }
                )
    return output


def _validate(rows: list[dict[str, Any]], label: str) -> None:
    if not STRICT_COMPLETE:
        return
    folds = {int(row["fold"]) for row in rows}
    if folds != set(EXPECTED_FOLDS):
        raise ValueError(f"{label}: expected folds {list(EXPECTED_FOLDS)}, got {sorted(folds)}")
    observed = {
        (int(row["fold"]), int(row["after_task"]), int(row["eval_task"]))
        for row in rows
    }
    expected = {
        (fold, after_task, eval_task)
        for fold in EXPECTED_FOLDS
        for after_task in range(EXPECTED_TASKS)
        for eval_task in range(after_task + 1)
    }
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if missing or extra:
        raise ValueError(
            f"{label}: incomplete evaluation triangle; "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )


def _task_names(rows: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        names[int(row["eval_task"])].add(str(row["task_name"]))
    conflicts = {task: values for task, values in names.items() if len(values) != 1}
    if conflicts:
        raise ValueError(f"Methods disagree on task names/order: {conflicts}")
    result = {task: next(iter(values)) for task, values in names.items()}
    if set(result) != set(range(EXPECTED_TASKS)):
        raise ValueError(f"Expected task indices 0..{EXPECTED_TASKS - 1}, got {sorted(result)}")
    return result


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            int(row["eval_task"]),
            str(row["task_name"]),
            int(row["after_task"]),
        )
        grouped[key].append(float(row["accuracy"]))

    output = []
    for (method, eval_task, task_name, after_task), values in grouped.items():
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        std = float(array.std(ddof=0))
        ci_half = 1.96 * std / math.sqrt(len(array))
        output.append(
            {
                "method": method,
                "eval_task": eval_task,
                "task_name": task_name,
                "after_task": after_task,
                "task_number": after_task + 1,
                "folds": len(array),
                "accuracy_mean": mean,
                "accuracy_std": std,
                "ci95_low": mean - ci_half,
                "ci95_high": mean + ci_half,
            }
        )
    return sorted(output, key=lambda row: (row["eval_task"], row["method"], row["after_task"]))


def _endpoints(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        grouped[(str(row["method"]), int(row["eval_task"]), str(row["task_name"]))].append(row)

    output = []
    for (method, eval_task, task_name), values in grouped.items():
        ordered = sorted(values, key=lambda row: int(row["after_task"]))
        initial = float(ordered[0]["accuracy_mean"])
        final = float(ordered[-1]["accuracy_mean"])
        output.append(
            {
                "method": method,
                "eval_task": eval_task,
                "task_name": task_name,
                "first_evaluated_after_task": int(ordered[0]["after_task"]),
                "initial_accuracy": initial,
                "final_accuracy": final,
                "performance_drop": initial - final,
            }
        )
    return sorted(output, key=lambda row: (row["eval_task"], row["method"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _band(row: dict[str, Any], scale: float) -> tuple[float, float]:
    mean = float(row["accuracy_mean"])
    if UNCERTAINTY == "std":
        low, high = mean - float(row["accuracy_std"]), mean + float(row["accuracy_std"])
    elif UNCERTAINTY == "ci95":
        low, high = float(row["ci95_low"]), float(row["ci95_high"])
    elif UNCERTAINTY == "none":
        low = high = mean
    else:
        raise ValueError(f"Unknown UNCERTAINTY={UNCERTAINTY!r}")
    return low * scale, high * scale


def _plot(
    methods: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    task_names: dict[int, str],
    task_indices: list[int],
    tasks_per_row: int,
    output_stem: str,
) -> None:
    scale = 100.0 if SCALE_TO_PERCENT else 1.0
    if not task_indices:
        raise ValueError("Plot must contain at least one task")
    ncols = min(max(1, tasks_per_row), len(task_indices))
    nrows = int(math.ceil(len(task_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH_PER_TASK * ncols, FIG_HEIGHT_PER_ROW * nrows),
        sharey=True,
        squeeze=False,
    )

    legend_handles = None
    legend_labels = None
    for panel_index, (eval_task, axis) in enumerate(
        zip(task_indices, axes.ravel()[: len(task_indices)])
    ):
        for spec in methods:
            values = sorted(
                (
                    row
                    for row in summary
                    if row["method"] == spec["label"] and row["eval_task"] == eval_task
                ),
                key=lambda row: row["after_task"],
            )
            if not values:
                continue
            x = np.asarray([int(row["task_number"]) for row in values])
            y = np.asarray([float(row["accuracy_mean"]) * scale for row in values])
            proposed = bool(spec.get("proposed", False))
            line = axis.plot(
                x,
                y,
                label=str(spec["label"]),
                color=str(spec["color"]),
                marker=str(spec["marker"]),
                linestyle=str(spec["linestyle"]),
                linewidth=PROPOSED_LINE_WIDTH if proposed else LINE_WIDTH,
                markersize=MARKER_SIZE + (1.8 if proposed else 0.0),
                zorder=int(spec.get("zorder", 5)),
            )[0]
            if UNCERTAINTY != "none":
                band = np.asarray([_band(row, scale) for row in values])
                axis.fill_between(
                    x,
                    band[:, 0],
                    band[:, 1],
                    color=line.get_color(),
                    alpha=BAND_ALPHA,
                    linewidth=0,
                    zorder=int(spec.get("zorder", 5)) - 1,
                )
            if ANNOTATE_OURS_DROP and proposed and len(values) > 1:
                drop = (float(values[0]["accuracy_mean"]) - float(values[-1]["accuracy_mean"])) * scale
                axis.text(
                    0.97,
                    0.05,
                    f"Ours drop: {drop:+.1f}",
                    color=str(spec["color"]),
                    fontsize=7,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                    transform=axis.transAxes,
                )

        axis.set_title(f"T{eval_task + 1}: {task_names[eval_task]}")
        axis.set_xlim(eval_task + 0.7, EXPECTED_TASKS + 0.3)
        axis.set_ylim(*Y_LIMITS)
        axis.set_xticks(range(eval_task + 1, EXPECTED_TASKS + 1))
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=GRID_ALPHA)
        axis.set_xlabel(X_LABEL)
        if panel_index % ncols == 0:
            axis.set_ylabel(Y_LABEL)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    for axis in axes.ravel()[len(task_indices):]:
        axis.set_axis_off()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=min(LEGEND_COLUMNS, len(legend_labels)),
            frameon=False,
        )
    fig.suptitle(TITLE, y=0.995, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.97])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{output_stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    enabled = [spec for spec in METHODS if bool(spec.get("enabled", True))]
    if not enabled:
        raise ValueError("Enable at least one method in METHODS")

    all_rows: list[dict[str, Any]] = []
    for spec in enabled:
        paths = _source_paths(str(spec["source"]))
        rows = _read_rows(paths, str(spec["label"]))
        _validate(rows, str(spec["label"]))
        all_rows.extend(rows)
        print(
            f"[curve] {spec['label']}: files={len(paths)} "
            f"folds={len({row['fold'] for row in rows})}"
        )

    names = _task_names(all_rows)
    summary = _summarize(all_rows)
    endpoints = _endpoints(summary)
    _write_csv(OUTPUT_DIR / "performance_drop_per_fold.csv", all_rows)
    _write_csv(OUTPUT_DIR / "performance_drop_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "performance_drop_endpoints.csv", endpoints)
    _plot(
        enabled,
        summary,
        names,
        task_indices=list(range(EXPECTED_TASKS)),
        tasks_per_row=TASKS_PER_ROW,
        output_stem="performance_drop_by_task",
    )
    first_task_count = min(FIRST_TASKS_TO_PLOT, EXPECTED_TASKS)
    _plot(
        enabled,
        summary,
        names,
        task_indices=list(range(first_task_count)),
        tasks_per_row=FIRST_TASKS_PER_ROW,
        output_stem="performance_drop_first_6_tasks_2x3",
    )
    print(
        f"Wrote {EXPECTED_TASKS}-task and {first_task_count}-task figures for "
        f"{len(enabled)} methods to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
