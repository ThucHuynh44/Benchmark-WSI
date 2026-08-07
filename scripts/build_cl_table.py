"""Build one continual-learning comparison table from many result folders.

The reported metrics follow the definitions in the accompanying evaluation
note, rather than reusing the ambiguously named canonical ``mACC`` column:

* mACC: mean, over training stages, of Class-IL micro accuracy.
* bACC: final Class-IL balanced accuracy, macro-averaged over tasks.
* Masked bACC: final Task-IL balanced accuracy, macro-averaged over tasks.
* BWT and FGT: Class-IL backward transfer and forgetting over old tasks.

All metrics are recomputed from ``evaluation/*/eval_matrix.csv``.  This also
lets the script reject incomplete triangular evaluation matrices instead of
silently treating a partially written fold as a completed experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


METRICS = ("mACC", "bACC", "masked_bACC", "BWT", "FGT")
OUTPUT_FIELDS = [
    "experiment",
    "method",
    "backbone",
    "buffer_size",
    "num_tasks",
    "expected_folds",
    "completed_folds",
    "folds_used",
    "status",
    "notes",
    "scale",
    *[field for metric in METRICS for field in (metric, f"{metric}_mean", f"{metric}_std")],
    "run_dir",
]


def _manifest_paths(input_path: Path) -> Iterable[Path]:
    """Yield one Class-IL manifest per experiment below ``input_path``."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.name != "run_manifest.json" or input_path.parent.name != "class_il":
            raise ValueError(
                "A manifest input must be evaluation/class_il/run_manifest.json: "
                f"{input_path}"
            )
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    yield from sorted(input_path.rglob("evaluation/class_il/run_manifest.json"))


def _read_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def _read_eval_matrix(path: Path) -> List[dict]:
    required = {"fold", "after_task", "eval_task", "acc", "bacc", "n"}
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            try:
                rows.append({
                    "fold": int(raw["fold"]),
                    "after_task": int(raw["after_task"]),
                    "eval_task": int(raw["eval_task"]),
                    "acc": float(raw["acc"]),
                    "bacc": float(raw["bacc"]),
                    "n": int(float(raw["n"])),
                })
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value at {path}:{line_number}: {error}") from error
    return rows


def _prediction_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _expected_keys(num_tasks: int, joint: bool) -> set[Tuple[int, int]]:
    if joint:
        return {(num_tasks - 1, task) for task in range(num_tasks)}
    return {
        (after_task, eval_task)
        for after_task in range(num_tasks)
        for eval_task in range(after_task + 1)
    }


def _fold_key_counts(rows: Sequence[dict]) -> Dict[int, Counter]:
    output: Dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        output[int(row["fold"])][(int(row["after_task"]), int(row["eval_task"]))] += 1
    return output


def _complete_folds(rows: Sequence[dict], expected: set[Tuple[int, int]]) -> set[int]:
    return {
        fold
        for fold, counts in _fold_key_counts(rows).items()
        if set(counts) == expected and all(count == 1 for count in counts.values())
    }


def _rows_by_key(rows: Sequence[dict], fold: int) -> Dict[Tuple[int, int], dict]:
    selected = [row for row in rows if int(row["fold"]) == int(fold)]
    return {
        (int(row["after_task"]), int(row["eval_task"])): row
        for row in selected
    }


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def _std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    center = _mean(finite)
    # Match the repository summaries: population standard deviation (ddof=0).
    return math.sqrt(sum((value - center) ** 2 for value in finite) / len(finite))


def _sequential_fold_metrics(
    class_rows: Mapping[Tuple[int, int], dict],
    task_rows: Mapping[Tuple[int, int], dict],
    num_tasks: int,
) -> Dict[str, float]:
    stage_micro = []
    for after_task in range(num_tasks):
        current = [class_rows[(after_task, task)] for task in range(after_task + 1)]
        denominator = sum(int(row["n"]) for row in current)
        if denominator <= 0:
            raise ValueError(f"after_task={after_task} has no test samples")
        stage_micro.append(
            sum(float(row["acc"]) * int(row["n"]) for row in current) / denominator
        )

    final = num_tasks - 1
    final_class = [class_rows[(final, task)] for task in range(num_tasks)]
    final_task = [task_rows[(final, task)] for task in range(num_tasks)]
    bwt = [
        float(class_rows[(final, task)]["acc"])
        - float(class_rows[(task, task)]["acc"])
        for task in range(num_tasks - 1)
    ]
    forgetting = []
    for task in range(num_tasks - 1):
        trajectory = [
            float(class_rows[(after_task, task)]["acc"])
            for after_task in range(task, num_tasks)
        ]
        forgetting.append(max(trajectory) - float(class_rows[(final, task)]["acc"]))

    return {
        "mACC": _mean(stage_micro),
        "bACC": _mean([float(row["bacc"]) for row in final_class]),
        "masked_bACC": _mean([float(row["bacc"]) for row in final_task]),
        "BWT": _mean(bwt),
        "FGT": _mean(forgetting),
    }


def _joint_fold_metrics(
    class_rows: Mapping[Tuple[int, int], dict],
    task_rows: Mapping[Tuple[int, int], dict],
    num_tasks: int,
) -> Dict[str, float]:
    final = num_tasks - 1
    return {
        # Joint has no per-stage trajectory, so these quantities are undefined.
        "mACC": math.nan,
        "bACC": _mean([float(class_rows[(final, task)]["bacc"]) for task in range(num_tasks)]),
        "masked_bACC": _mean([float(task_rows[(final, task)]["bacc"]) for task in range(num_tasks)]),
        "BWT": math.nan,
        "FGT": math.nan,
    }


def _buffer_size(experiment_dir: Path) -> str:
    match = re.search(r"(?:^|_)buffer(\d+)(?:_|$)", experiment_dir.name)
    if match:
        return match.group(1)
    for path in sorted(experiment_dir.glob("fold_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and str(rows[-1].get("buffer_size", "")).strip():
            return str(rows[-1]["buffer_size"]).strip()
        break
    return ""


def _format_number(value: float, precision: int) -> str:
    return "" if not math.isfinite(value) else f"{value:.{precision}f}"


def _summarize_experiment(
    manifest_path: Path,
    expected_tasks: int,
    percent: bool,
    precision: int,
    check_predictions: bool,
) -> dict:
    manifest = _read_manifest(manifest_path)
    experiment_dir = manifest_path.parents[2]
    method = str(manifest.get("method", ""))
    backbone = str(manifest.get("backbone", ""))
    num_tasks = int(manifest.get("num_tasks", 0))
    expected_folds = sorted(int(fold) for fold in manifest.get("folds", []))
    notes: List[str] = []
    severity = 0  # 0 complete, 1 incomplete, 2 corrupt, 3 invalid

    if num_tasks != int(expected_tasks):
        severity = 3
        notes.append(f"expected {expected_tasks} tasks but manifest has {num_tasks}")

    class_path = experiment_dir / "evaluation/class_il/eval_matrix.csv"
    task_path = experiment_dir / "evaluation/task_il/eval_matrix.csv"
    try:
        class_rows = _read_eval_matrix(class_path)
        task_rows = _read_eval_matrix(task_path)
    except (FileNotFoundError, ValueError) as error:
        severity = 3
        notes.append(str(error))
        class_rows, task_rows = [], []

    joint = method == "joint"
    expected_keys = _expected_keys(num_tasks, joint) if num_tasks > 0 else set()
    class_complete = _complete_folds(class_rows, expected_keys)
    task_complete = _complete_folds(task_rows, expected_keys)
    complete = sorted(class_complete & task_complete & set(expected_folds))
    missing = sorted(set(expected_folds).difference(complete))
    if missing:
        severity = max(severity, 1)
        notes.append(f"incomplete folds: {missing}")

    for mode, rows in (("class_il", class_rows), ("task_il", task_rows)):
        counts = _fold_key_counts(rows)
        duplicate_folds = sorted(
            fold for fold, fold_counts in counts.items() if any(count > 1 for count in fold_counts.values())
        )
        if duplicate_folds:
            severity = max(severity, 2)
            notes.append(f"{mode} duplicate eval keys in folds {duplicate_folds}")
        unexpected_folds = sorted(set(counts).difference(expected_folds))
        if unexpected_folds:
            severity = max(severity, 2)
            notes.append(f"{mode} unexpected folds: {unexpected_folds}")

        if check_predictions:
            prediction_path = experiment_dir / f"evaluation/{mode}/per_slide_predictions.csv"
            actual_predictions = _prediction_row_count(prediction_path)
            expected_predictions = sum(int(row["n"]) for row in rows)
            if actual_predictions is None:
                severity = max(severity, 1)
                notes.append(f"missing {mode} predictions")
            elif actual_predictions != expected_predictions:
                severity = max(severity, 2)
                notes.append(
                    f"{mode} prediction rows={actual_predictions}, expected={expected_predictions}"
                )

    fold_metrics: List[Dict[str, float]] = []
    for fold in complete:
        class_by_key = _rows_by_key(class_rows, fold)
        task_by_key = _rows_by_key(task_rows, fold)
        fold_metrics.append(
            _joint_fold_metrics(class_by_key, task_by_key, num_tasks)
            if joint
            else _sequential_fold_metrics(class_by_key, task_by_key, num_tasks)
        )

    if joint:
        notes.append("Joint has no stage trajectory: mACC/BWT/FGT are N/A")

    scale = 100.0 if percent else 1.0
    output = {
        "experiment": experiment_dir.name,
        "method": method,
        "backbone": backbone,
        "buffer_size": _buffer_size(experiment_dir),
        "num_tasks": num_tasks,
        "expected_folds": len(expected_folds),
        "completed_folds": len(complete),
        "folds_used": ",".join(map(str, complete)),
        "status": ("complete", "incomplete", "corrupt", "invalid")[severity],
        "notes": "; ".join(notes),
        "scale": "percent" if percent else "fraction",
        "run_dir": str(experiment_dir),
    }
    for metric in METRICS:
        values = [row[metric] for row in fold_metrics]
        mean = _mean(values) * scale
        std = _std(values) * scale
        output[metric] = (
            ""
            if not (math.isfinite(mean) and math.isfinite(std))
            else f"{mean:.{precision}f} ± {std:.{precision}f}"
        )
        output[f"{metric}_mean"] = _format_number(mean, precision)
        output[f"{metric}_std"] = _format_number(std, precision)
    return output


def build_table(
    input_path: str,
    output_path: str,
    expected_tasks: int = 10,
    percent: bool = False,
    precision: int = 4,
    check_predictions: bool = False,
    strict: bool = False,
) -> List[dict]:
    manifests = list(_manifest_paths(Path(input_path)))
    if not manifests:
        raise FileNotFoundError(
            f"No evaluation/class_il/run_manifest.json found below {Path(input_path).expanduser()}"
        )
    rows = [
        _summarize_experiment(
            path,
            expected_tasks=expected_tasks,
            percent=percent,
            precision=precision,
            check_predictions=check_predictions,
        )
        for path in manifests
    ]
    rows.sort(key=lambda row: (str(row["method"]), str(row["backbone"]), str(row["experiment"])))

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    bad = [row for row in rows if row["status"] != "complete"]
    if strict and bad:
        details = ", ".join(f"{row['experiment']}={row['status']}" for row in bad)
        raise RuntimeError(f"Result audit failed: {details}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parent directory containing experiment folders")
    parser.add_argument("--output", required=True, help="Output comparison CSV")
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=10,
        help="Expected number of continual tasks (default: 10)",
    )
    parser.add_argument("--percent", action="store_true", help="Write metrics on a 0-100 scale")
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument(
        "--check-predictions",
        action="store_true",
        help=(
            "Also scan the large per-slide CSVs and compare their row counts "
            "with eval_matrix (slower on network storage)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return an error if any experiment is incomplete, corrupt, or invalid",
    )
    args = parser.parse_args()
    rows = build_table(
        input_path=args.input,
        output_path=args.output,
        expected_tasks=args.expected_tasks,
        percent=args.percent,
        precision=args.precision,
        check_predictions=args.check_predictions,
        strict=args.strict,
    )
    counts = Counter(str(row["status"]) for row in rows)
    print(f"Wrote {len(rows)} experiments to {Path(args.output).expanduser().resolve()}")
    print("Status: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
