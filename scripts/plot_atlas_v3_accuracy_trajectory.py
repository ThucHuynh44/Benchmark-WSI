"""Plot Class-IL accuracy after every task for forward runs and ATLAS-v3.

Edit only the USER CONFIGURATION block below to add/remove methods or change
the figure appearance.  Each curve is the 10-fold mean of micro accuracy over
all test samples from tasks seen at that training stage.
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
OUTPUT_DIR = REPO_ROOT / "results/figures/atlas_v3_accuracy_trajectory"

EXPECTED_FOLDS = tuple(range(10))
EXPECTED_TASKS = 10
ACCURACY_FIELD = "acc"
STAGE_AGGREGATION = "micro_by_n"  # "micro_by_n" or "macro_over_tasks"
UNCERTAINTY = "std"  # "std", "ci95", or "none"
SCALE_TO_PERCENT = True
STRICT_COMPLETE = True

FIGSIZE = (13.0, 7.2)
DPI = 300
TITLE = "Class-IL Accuracy Across the Continual Learning Sequence"
X_LABEL = "Task learned"
Y_LABEL = "Accuracy (%)" if SCALE_TO_PERCENT else "Accuracy"
Y_LIMITS = (0.0, 100.0) if SCALE_TO_PERCENT else (0.0, 1.0)
GRID_ALPHA = 0.22
BAND_ALPHA = 0.10
LINE_WIDTH = 1.8
PROPOSED_LINE_WIDTH = 3.2
MARKER_SIZE = 5.5
LEGEND_COLUMNS = 1

# ``source`` is a path or glob relative to REPO_ROOT. Toggle ``enabled`` to
# control which curves appear. Add a new dictionary to add another method.
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
        # Disabled because folds 1..9 in this file use the reverse task order,
        # so its stage indices are not comparable to the fixed forward order.
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
    # Examples kept disabled to make additions explicit and easy:
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
    "n",
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
                key = (
                    int(raw["fold"]),
                    int(raw["after_task"]),
                    int(raw["eval_task"]),
                )
                if key in seen:
                    raise ValueError(f"Duplicate evaluation row for {label}: {key}")
                seen.add(key)
                output.append(
                    {
                        "fold": key[0],
                        "after_task": key[1],
                        "eval_task": key[2],
                        "task_name": raw["task_name"],
                        "value": float(raw[ACCURACY_FIELD]),
                        "n": int(raw["n"]),
                    }
                )
    return output


def _stage_accuracy(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["eval_task"] <= row["after_task"]:
            grouped[(row["fold"], row["after_task"])].append(row)

    output = []
    for (fold, after_task), values in sorted(grouped.items()):
        expected_eval = set(range(after_task + 1))
        observed_eval = {int(row["eval_task"]) for row in values}
        if observed_eval != expected_eval:
            raise ValueError(
                f"{label} fold={fold} after_task={after_task}: expected eval tasks "
                f"{sorted(expected_eval)}, got {sorted(observed_eval)}"
            )
        if STAGE_AGGREGATION == "micro_by_n":
            denominator = sum(int(row["n"]) for row in values)
            if denominator <= 0:
                raise ValueError(f"{label} has no samples at task {after_task + 1}")
            accuracy = sum(row["value"] * row["n"] for row in values) / denominator
        elif STAGE_AGGREGATION == "macro_over_tasks":
            accuracy = float(np.mean([row["value"] for row in values]))
        else:
            raise ValueError(f"Unknown STAGE_AGGREGATION={STAGE_AGGREGATION!r}")
        output.append(
            {"method": label, "fold": fold, "after_task": after_task, "accuracy": accuracy}
        )

    observed_folds = {int(row["fold"]) for row in output}
    observed_tasks = {int(row["after_task"]) for row in output}
    if STRICT_COMPLETE and observed_folds != set(EXPECTED_FOLDS):
        raise ValueError(
            f"{label}: expected folds {list(EXPECTED_FOLDS)}, got {sorted(observed_folds)}"
        )
    if STRICT_COMPLETE and observed_tasks != set(range(EXPECTED_TASKS)):
        raise ValueError(
            f"{label}: expected {EXPECTED_TASKS} task stages, got {sorted(observed_tasks)}"
        )
    return output


def _task_names(all_rows: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, set[str]] = defaultdict(set)
    for row in all_rows:
        names[int(row["eval_task"])].add(str(row["task_name"]))
    conflicts = {task: values for task, values in names.items() if len(values) != 1}
    if conflicts:
        raise ValueError(f"Methods disagree on task names/order: {conflicts}")
    return {task: next(iter(values)) for task, values in names.items()}


def _summarize(stage_rows: list[dict[str, Any]], task_names: dict[int, str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in stage_rows:
        grouped[(row["method"], row["after_task"])].append(float(row["accuracy"]))
    output = []
    for (method, after_task), values in grouped.items():
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        std = float(array.std(ddof=0))
        ci_half = 1.96 * std / math.sqrt(len(array))
        output.append(
            {
                "method": method,
                "after_task": after_task,
                "task_number": after_task + 1,
                "task_name": task_names[after_task],
                "folds": len(array),
                "accuracy_mean": mean,
                "accuracy_std": std,
                "ci95_low": mean - ci_half,
                "ci95_high": mean + ci_half,
            }
        )
    return sorted(output, key=lambda row: (row["method"], row["after_task"]))


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


def _plot(methods: list[dict[str, Any]], summary: list[dict[str, Any]], task_names: dict[int, str]) -> None:
    scale = 100.0 if SCALE_TO_PERCENT else 1.0
    fig, axis = plt.subplots(figsize=FIGSIZE)
    for spec in methods:
        values = sorted(
            (row for row in summary if row["method"] == spec["label"]),
            key=lambda row: row["after_task"],
        )
        x = np.asarray([row["task_number"] for row in values])
        y = np.asarray([row["accuracy_mean"] * scale for row in values])
        width = PROPOSED_LINE_WIDTH if spec.get("proposed", False) else LINE_WIDTH
        axis.plot(
            x,
            y,
            label=spec["label"],
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=width,
            markersize=MARKER_SIZE + (2.0 if spec.get("proposed", False) else 0.0),
            zorder=int(spec.get("zorder", 5)),
        )
        if UNCERTAINTY != "none":
            bounds = np.asarray([_band(row, scale) for row in values])
            axis.fill_between(
                x,
                bounds[:, 0],
                bounds[:, 1],
                color=spec["color"],
                alpha=BAND_ALPHA,
                linewidth=0,
                zorder=int(spec.get("zorder", 5)) - 1,
            )

    labels = [f"T{task + 1}\n{task_names[task]}" for task in range(EXPECTED_TASKS)]
    axis.set_xticks(range(1, EXPECTED_TASKS + 1), labels, rotation=25, ha="right")
    axis.set_xlabel(X_LABEL)
    axis.set_ylabel(Y_LABEL)
    axis.set_title(TITLE)
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(0.7, EXPECTED_TASKS + 0.3)
    axis.grid(True, axis="both", alpha=GRID_ALPHA, linestyle="--")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=LEGEND_COLUMNS,
    )
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "accuracy_by_task.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "accuracy_by_task.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    enabled = [spec for spec in METHODS if bool(spec.get("enabled", True))]
    if not enabled:
        raise ValueError("Enable at least one method in METHODS")

    all_eval_rows: list[dict[str, Any]] = []
    all_stage_rows: list[dict[str, Any]] = []
    for spec in enabled:
        paths = _source_paths(str(spec["source"]))
        rows = _read_rows(paths, str(spec["label"]))
        all_eval_rows.extend(rows)
        stages = _stage_accuracy(rows, str(spec["label"]))
        all_stage_rows.extend(stages)
        print(
            f"[curve] {spec['label']}: files={len(paths)} "
            f"folds={len({row['fold'] for row in stages})}"
        )

    names = _task_names(all_eval_rows)
    if set(names) != set(range(EXPECTED_TASKS)):
        raise ValueError(f"Expected task names 0..{EXPECTED_TASKS - 1}, got {sorted(names)}")
    summary = _summarize(all_stage_rows, names)
    _write_csv(OUTPUT_DIR / "accuracy_by_task_per_fold.csv", all_stage_rows)
    _write_csv(OUTPUT_DIR / "accuracy_by_task_summary.csv", summary)
    _plot(enabled, summary, names)
    print(f"Wrote {len(enabled)} curves to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
