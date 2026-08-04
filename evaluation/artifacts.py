"""Evaluation artifacts compatible with the sibling Benchmark repository."""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


EVAL_MATRIX_FIELDS = [
    "method", "fold", "mode", "after_task", "eval_task", "task_name",
    "acc", "bacc", "macro_f1", "weighted_f1", "auroc", "kappa",
    "loss", "n", "eval_time_sec", "confusion_matrix_path",
]

PER_FOLD_FIELDS = [
    "method", "fold", "mode", "final_acc", "final_bacc", "macro_f1",
    "weighted_f1", "auroc", "mACC", "BWT", "FGT", "training_time",
    "total_eval_time", "inference_time_per_task", "total_parameters",
    "trainable_parameters", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib",
]

PER_TASK_FIELDS = [
    "method", "task", "task_name", "mean_acc", "std_acc", "mean_bacc",
    "std_bacc", "mean_macro_f1", "std_macro_f1", "mean_auroc",
    "std_auroc", "n_test",
]


def prediction_fields(total_classes: int) -> List[str]:
    fields = [
        "method", "fold", "mode", "after_task", "eval_task", "slide_id",
        "patient_id", "task_name", "split", "feature_path", "y_true_local",
        "y_true_global", "y_pred_global", "y_pred_local", "correct",
    ]
    fields += [f"prob_{class_id}" for class_id in range(int(total_classes))]
    fields += [f"logit_{class_id}" for class_id in range(int(total_classes))]
    fields += ["num_patches_total", "num_patches_used", "k", "seed"]
    return fields


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerows(rows)


def initialize_artifacts(save_dir: str, total_classes: int) -> Dict[str, object]:
    paths: Dict[str, object] = {
        "save_dir": save_dir,
        "eval_matrix": os.path.join(save_dir, "eval_matrix.csv"),
        "predictions": os.path.join(save_dir, "per_slide_predictions.csv"),
        "confusion_dir": os.path.join(save_dir, "confusion_matrices"),
        "prediction_fields": prediction_fields(total_classes),
        "rows": [],
    }
    write_csv(str(paths["eval_matrix"]), [], EVAL_MATRIX_FIELDS)
    write_csv(str(paths["predictions"]), [], paths["prediction_fields"])
    os.makedirs(str(paths["confusion_dir"]), exist_ok=True)
    return paths


def append_evaluation(
    artifacts: Dict[str, object], row: dict, prediction_rows: Sequence[dict]
) -> None:
    append_csv(str(artifacts["eval_matrix"]), [row], EVAL_MATRIX_FIELDS)
    append_csv(
        str(artifacts["predictions"]),
        prediction_rows,
        artifacts["prediction_fields"],
    )
    artifacts["rows"].append(dict(row))


def _feature_path(dataset, slide_id: str, row=None) -> str:
    if row is not None:
        for column in ("feature_path", "features_path", "h5_path"):
            if column in row and pd.notna(row[column]) and str(row[column]).strip():
                return str(row[column])
    feature_root = str(
        getattr(dataset, "feature_root", getattr(dataset, "data_dir", ""))
    )
    if not feature_root:
        return ""
    stem = str(slide_id).strip()
    if stem.lower().endswith((".svs", ".h5")):
        stem = stem[:-4]
    candidates = [
        os.path.join(feature_root, f"{stem}.h5"),
        os.path.join(feature_root, "h5_files", f"{stem}.h5"),
        os.path.join(feature_root, "features_conch_v15", f"{stem}.h5"),
    ]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def slide_metadata(dataset, sample_index: int) -> Dict[str, str]:
    if hasattr(dataset, "slide_data"):
        row = dataset.slide_data.iloc[int(sample_index)]
        slide_id = str(row.get("slide_id", ""))
        patient_id = row.get("case_id", row.get("patient_id", ""))
        patient_id = "" if pd.isna(patient_id) else str(patient_id)
        return {
            "slide_id": slide_id,
            "patient_id": patient_id,
            "feature_path": _feature_path(dataset, slide_id, row=row),
        }
    return {"slide_id": "", "patient_id": "", "feature_path": ""}


def safe_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    present_classes = np.unique(targets).astype(int)
    if len(present_classes) < 2:
        return float("nan")
    if probabilities.ndim != 2 or int(present_classes.max()) >= probabilities.shape[1]:
        raise ValueError(
            "probabilities must contain a column for every target class: "
            f"targets={present_classes.tolist()}, shape={probabilities.shape}"
        )
    selected = probabilities[:, present_classes]
    normalizer = selected.sum(axis=1, keepdims=True)
    selected = np.divide(
        selected,
        normalizer,
        out=np.zeros_like(selected),
        where=normalizer > 0,
    )
    try:
        if len(present_classes) == 2:
            return float(
                roc_auc_score(
                    (targets == present_classes[1]).astype(np.int64),
                    selected[:, 1],
                )
            )
        return float(
            roc_auc_score(
                targets,
                selected,
                labels=present_classes,
                multi_class="ovo",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def evaluation_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        bacc = float(balanced_accuracy_score(targets, predictions))
    return {
        "loss": float(loss),
        "acc": float(accuracy_score(targets, predictions)),
        "bacc": bacc,
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "auroc": safe_auroc(targets, probabilities),
        "kappa": float(cohen_kappa_score(targets, predictions)),
        "n": int(len(targets)),
    }


def write_confusion_matrix(
    output_dir: str,
    fold_id: int,
    after_task: int,
    eval_task: int,
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: Sequence[int],
    mode: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir,
        f"fold_{int(fold_id)}_after_{int(after_task)}_eval_{int(eval_task)}_{mode}.csv",
    )
    labels_array = np.asarray(labels, dtype=int)
    matrix = confusion_matrix(targets, predictions, labels=labels_array)
    pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels_array],
        columns=[f"pred_{label}" for label in labels_array],
    ).to_csv(path)
    return path


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.std(finite)) if finite else float("nan")


def _continual_scores(final_rows: Sequence[dict], rows: Sequence[dict]) -> Tuple[float, float]:
    if len(final_rows) <= 1:
        return float("nan"), float("nan")
    final_after_task = int(final_rows[0]["after_task"])
    final_by_task = {int(row["eval_task"]): float(row["acc"]) for row in final_rows}
    diagonal = {
        int(row["eval_task"]): float(row["acc"])
        for row in rows
        if int(row["after_task"]) == int(row["eval_task"])
    }
    bwt = [
        final_by_task[task] - diagonal[task]
        for task in range(final_after_task)
        if task in final_by_task and task in diagonal
    ]
    forgetting = []
    for task in range(final_after_task):
        trajectory = [
            float(row["acc"])
            for row in rows
            if int(row["eval_task"]) == task and int(row["after_task"]) >= task
        ]
        if trajectory and task in final_by_task:
            forgetting.append(max(trajectory) - final_by_task[task])
    return _finite_mean(bwt), _finite_mean(forgetting)


def build_fold_summary(
    method: str,
    rows: Sequence[dict],
    training_times: Dict[int, float],
    resource_usage: Dict[int, dict] | None = None,
) -> List[dict]:
    resource_usage = resource_usage or {}
    summaries = []
    for fold_id in sorted({int(row["fold"]) for row in rows}):
        fold_rows = [row for row in rows if int(row["fold"]) == fold_id]
        final_after = max(int(row["after_task"]) for row in fold_rows)
        final_rows = [row for row in fold_rows if int(row["after_task"]) == final_after]
        bwt, forgetting = _continual_scores(final_rows, fold_rows)
        total_eval_time = sum(float(row["eval_time_sec"]) for row in fold_rows)
        resources = resource_usage.get(fold_id, {})
        peak_allocated = resources.get("peak_gpu_allocated_mib")
        peak_reserved = resources.get("peak_gpu_reserved_mib")
        summaries.append({
            "method": method,
            "fold": fold_id,
            "mode": final_rows[0]["mode"],
            "final_acc": _finite_mean([row["acc"] for row in final_rows]),
            "final_bacc": _finite_mean([row["bacc"] for row in final_rows]),
            "macro_f1": _finite_mean([row["macro_f1"] for row in final_rows]),
            "weighted_f1": _finite_mean([row["weighted_f1"] for row in final_rows]),
            "auroc": _finite_mean([row["auroc"] for row in final_rows]),
            "mACC": _finite_mean([row["acc"] for row in final_rows]),
            "BWT": bwt,
            "FGT": forgetting,
            "training_time": float(training_times.get(fold_id, 0.0)),
            "total_eval_time": float(total_eval_time),
            "inference_time_per_task": float(total_eval_time / max(len(fold_rows), 1)),
            "total_parameters": int(resources.get("total_parameters", 0)),
            "trainable_parameters": int(resources.get("trainable_parameters", 0)),
            "peak_gpu_allocated_mib": (
                float(peak_allocated) if peak_allocated is not None else float("nan")
            ),
            "peak_gpu_reserved_mib": (
                float(peak_reserved) if peak_reserved is not None else float("nan")
            ),
        })
    return summaries


def with_fold_aggregates(rows: Sequence[dict]) -> List[dict]:
    rows = list(rows)
    if not rows:
        return []
    numeric = [field for field in PER_FOLD_FIELDS if field not in ("method", "fold", "mode")]
    output = list(rows)
    for label, reducer in (("mean", _finite_mean), ("std", _finite_std)):
        output.append({
            "method": rows[0]["method"],
            "fold": label,
            "mode": rows[0]["mode"],
            **{field: reducer([row[field] for row in rows]) for field in numeric},
        })
    return output


def build_task_summary(method: str, rows: Sequence[dict]) -> List[dict]:
    summaries = []
    for task_id in sorted({int(row["eval_task"]) for row in rows}):
        task_rows = [row for row in rows if int(row["eval_task"]) == task_id]
        final_rows = [
            row
            for row in task_rows
            if int(row["after_task"]) == max(
                int(candidate["after_task"])
                for candidate in rows
                if int(candidate["fold"]) == int(row["fold"])
            )
        ]
        if not final_rows:
            continue
        summaries.append({
            "method": method,
            "task": task_id,
            "task_name": final_rows[0]["task_name"],
            "mean_acc": _finite_mean([row["acc"] for row in final_rows]),
            "std_acc": _finite_std([row["acc"] for row in final_rows]),
            "mean_bacc": _finite_mean([row["bacc"] for row in final_rows]),
            "std_bacc": _finite_std([row["bacc"] for row in final_rows]),
            "mean_macro_f1": _finite_mean([row["macro_f1"] for row in final_rows]),
            "std_macro_f1": _finite_std([row["macro_f1"] for row in final_rows]),
            "mean_auroc": _finite_mean([row["auroc"] for row in final_rows]),
            "std_auroc": _finite_std([row["auroc"] for row in final_rows]),
            "n_test": int(sum(int(row["n"]) for row in final_rows)),
        })
    return summaries


def finalize_artifacts(
    artifacts: Dict[str, object],
    method: str,
    training_times: Dict[int, float],
    resource_usage: Dict[int, dict] | None = None,
) -> None:
    rows = list(artifacts["rows"])
    if not rows:
        return
    fold_rows = with_fold_aggregates(
        build_fold_summary(method, rows, training_times, resource_usage)
    )
    task_rows = build_task_summary(method, rows)
    write_csv(str(artifacts["eval_matrix"]), rows, EVAL_MATRIX_FIELDS)
    write_csv(
        os.path.join(str(artifacts["save_dir"]), "per_fold_summary.csv"),
        fold_rows,
        PER_FOLD_FIELDS,
    )
    write_csv(
        os.path.join(str(artifacts["save_dir"]), "per_task_summary.csv"),
        task_rows,
        PER_TASK_FIELDS,
    )


def repo_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(save_dir: str, manifest: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    payload = dict(manifest)
    payload.setdefault("artifact_schema_version", 2)
    payload.setdefault("command", "python " + shlex.join([sys.argv[0], *sys.argv[1:]]))
    with open(os.path.join(save_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
