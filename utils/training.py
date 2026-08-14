"""Training and evaluation for variable-class continual WSI streams."""

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from evaluation.artifacts import (
    append_evaluation,
    evaluation_metrics,
    finalize_artifacts,
    initialize_artifacts,
    repo_commit,
    slide_metadata,
    write_manifest,
    write_confusion_matrix,
)
from models.utils.continual_model import ContinualModel
from utils.loggers import CsvLogger
from utils.tb_logger import TensorboardLogger


@dataclass
class EvaluationResult:
    class_il_accuracy: List[float]
    class_il_micro_accuracy: float
    task_il_accuracy: List[float]
    task_il_micro_accuracy: float
    task_auc: List[float]
    global_seen_auc: float
    class_il_metrics: List[Dict[str, float]]
    task_il_metrics: List[Dict[str, float]]
    global_seen_metrics: Dict[str, float]

    @property
    def class_il_macro_accuracy(self):
        return float(np.nanmean(self.class_il_accuracy))

    @property
    def task_il_macro_accuracy(self):
        return float(np.nanmean(self.task_il_accuracy))

    @property
    def task_macro_auc(self):
        return float(np.nanmean(self.task_auc))

    @staticmethod
    def _values(metrics, field):
        return [float(metric[field]) for metric in metrics]

    @property
    def class_il_balanced_accuracy(self):
        return self._values(self.class_il_metrics, "bacc")

    @property
    def task_il_balanced_accuracy(self):
        return self._values(self.task_il_metrics, "bacc")

    @property
    def class_il_macro_f1(self):
        return self._values(self.class_il_metrics, "macro_f1")

    @property
    def task_il_macro_f1(self):
        return self._values(self.task_il_metrics, "macro_f1")

    @property
    def class_il_weighted_f1(self):
        return self._values(self.class_il_metrics, "weighted_f1")

    @property
    def task_il_weighted_f1(self):
        return self._values(self.task_il_metrics, "weighted_f1")

    @property
    def class_il_kappa(self):
        return self._values(self.class_il_metrics, "kappa")

    @property
    def task_il_kappa(self):
        return self._values(self.task_il_metrics, "kappa")

    @property
    def class_il_auc(self):
        return self._values(self.class_il_metrics, "auroc")

    @property
    def class_il_loss(self):
        return self._values(self.class_il_metrics, "loss")

    @property
    def task_il_loss(self):
        return self._values(self.task_il_metrics, "loss")


def _loss_value(result) -> float:
    """Normalize scalar- or dictionary-valued observe results."""
    if isinstance(result, dict):
        if "loss" not in result:
            raise ValueError("observe() dictionaries must contain a 'loss' key")
        result = result["loss"]
    if torch.is_tensor(result):
        if result.numel() != 1:
            raise ValueError(
                f"observe() loss tensors must be scalar, got {tuple(result.shape)}"
            )
        result = result.detach().item()
    value = float(result)
    if not np.isfinite(value):
        raise FloatingPointError(f"observe() returned a non-finite loss: {value}")
    return value


def _iter_logical_batches(loader, group_size: int):
    """Group physical one-WSI batches and always flush the final remainder."""
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"logical group_size must be positive, got {group_size}")
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == group_size:
            yield group
            group = []
    if group:
        yield group


def _observe_logical_batch(model, raw_group, task_id: int, ssl: bool = False):
    prepared = []
    for batch in raw_group:
        features, coords, patch_size = model.prepare_inputs(
            batch.features,
            batch.coords,
            batch.patch_size_level0,
            training=True,
        )
        labels = batch.labels.to(model.device)
        prepared.append((features, coords, patch_size, labels))
    return model.observe_many(prepared, task=task_id, ssl=ssl)


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, task_id: int) -> None:
    """In-place Task-IL mask using a dynamic task class slice."""
    task_slice = dataset.task_slice(task_id)
    outputs[:, :task_slice.start] = -float("inf")
    outputs[:, task_slice.stop:] = -float("inf")


def mask_unseen_classes(outputs: torch.Tensor, dataset: ContinualDataset, last_task: int) -> None:
    """In-place Class-IL mask for classes not introduced yet."""
    outputs[:, dataset.seen_class_count(last_task):] = -float("inf")


def _full_probabilities(probabilities, start: int, total_classes: int):
    full = np.zeros((probabilities.shape[0], int(total_classes)), dtype=np.float32)
    full[:, int(start):int(start) + probabilities.shape[1]] = probabilities
    return full


def _prediction_row(
    *,
    context,
    mode,
    loader,
    sample_index,
    task_id,
    task_slice,
    target,
    prediction,
    probabilities,
    logits,
    num_patches_total,
    num_patches_used,
):
    row = {
        "method": context["method"],
        "fold": context["fold"],
        "mode": mode,
        "after_task": context["after_task"],
        "eval_task": task_id,
        **slide_metadata(loader.dataset, sample_index),
        "task_name": context["task_order"][task_id],
        "split": "test",
        "y_true_local": int(target) - task_slice.start,
        "y_true_global": int(target),
        "y_pred_global": int(prediction),
        "y_pred_local": (
            int(prediction) - task_slice.start
            if task_slice.start <= int(prediction) < task_slice.stop
            else ""
        ),
        "correct": int(int(prediction) == int(target)),
        "num_patches_total": int(num_patches_total),
        "num_patches_used": int(num_patches_used),
        "k": int(context["k"]),
        "seed": context["seed"],
    }
    for class_id in range(int(context["total_classes"])):
        row[f"prob_{class_id}"] = float(probabilities[class_id])
        row[f"logit_{class_id}"] = float(logits[class_id])
    return row


def _append_metric_artifact(
    context,
    mode_key,
    task_id,
    metrics,
    targets,
    predictions,
    labels,
    prediction_rows,
    eval_time,
):
    if context is None:
        return
    artifacts = context[mode_key]
    mode = "class-il-seen" if mode_key == "class_il" else "task-il"
    row = {
        "method": context["method"],
        "fold": context["fold"],
        "mode": mode,
        "after_task": context["after_task"],
        "eval_task": task_id,
        "task_name": context["task_order"][task_id],
        **metrics,
        "eval_time_sec": float(eval_time),
    }
    row["confusion_matrix_path"] = write_confusion_matrix(
        str(artifacts["confusion_dir"]),
        context["fold"],
        context["after_task"],
        task_id,
        targets,
        predictions,
        labels,
        mode,
    )
    append_evaluation(artifacts, row, prediction_rows)


def evaluate(
    model: ContinualModel,
    dataset: ContinualDataset,
    last: bool = False,
    artifact_context=None,
) -> EvaluationResult:
    status = model.net.training
    model.net.eval()
    if not dataset.test_loaders:
        raise ValueError("No test loaders are available for evaluation")
    seen_task = len(dataset.test_loaders) - 1
    selected = [(seen_task, dataset.test_loaders[-1])] if last else list(enumerate(dataset.test_loaders))

    class_metrics_all, task_metrics_all = [], []
    class_correct_total = task_correct_total = sample_total = 0
    global_targets, global_predictions, global_probabilities = [], [], []
    seen_count = dataset.seen_class_count(seen_task)
    total_classes = int(
        getattr(dataset, "total_num_classes", sum(dataset.task_num_classes))
    )

    with torch.no_grad():
        for task_id, loader in selected:
            evaluation_start = time.perf_counter()
            task_slice = dataset.task_slice(task_id)
            class_targets, class_predictions_all, class_probabilities_all = [], [], []
            task_predictions_all, task_probabilities_all = [], []
            class_loss_sum = task_loss_sum = 0.0
            class_prediction_rows, task_prediction_rows = [], []
            class_correct = task_correct = total = 0
            eval_bar = tqdm(
                loader,
                total=len(loader),
                desc=f"eval task {task_id + 1}",
                leave=False,
                disable=bool(
                    getattr(getattr(model, "args", None), "non_verbose", False)
                ),
            )
            for sample_index, (features, coords, patch_size, labels) in enumerate(eval_bar):
                num_patches_total = int(features.shape[-2])
                features, coords, patch_size = model.prepare_inputs(
                    features, coords, patch_size, training=False
                )
                num_patches_used = int(features.shape[-2])
                labels = labels.to(model.device)
                logits = model([features, coords, patch_size])[0]

                class_logits = logits.clone()
                mask_unseen_classes(class_logits, dataset, seen_task)
                class_probabilities = F.softmax(class_logits[:, :seen_count], dim=1)
                class_predictions = class_logits.argmax(dim=1)

                task_logits = logits[:, task_slice]
                task_probabilities = F.softmax(task_logits, dim=1)
                task_predictions = task_logits.argmax(dim=1) + task_slice.start

                batch_size = int(labels.numel())
                class_loss_sum += float(
                    F.cross_entropy(class_logits[:, :seen_count], labels).item()
                ) * batch_size
                task_loss_sum += float(
                    F.cross_entropy(task_logits, labels - task_slice.start).item()
                ) * batch_size

                class_correct += (class_predictions == labels).sum().item()
                task_correct += (task_predictions == labels).sum().item()
                total += labels.numel()
                target_values = labels.cpu().numpy()
                class_prediction_values = class_predictions.cpu().numpy()
                task_prediction_values = task_predictions.cpu().numpy()
                class_probability_values = _full_probabilities(
                    class_probabilities.cpu().numpy(), 0, total_classes
                )
                task_probability_values = _full_probabilities(
                    task_probabilities.cpu().numpy(),
                    task_slice.start,
                    total_classes,
                )
                logits_values = logits.cpu().numpy()
                class_targets.append(target_values)
                class_predictions_all.append(class_prediction_values)
                task_predictions_all.append(task_prediction_values)
                class_probabilities_all.append(class_probability_values)
                task_probabilities_all.append(task_probability_values)

                if artifact_context is not None:
                    class_prediction_rows.append(_prediction_row(
                        context=artifact_context,
                        mode="class-il-seen",
                        loader=loader,
                        sample_index=sample_index,
                        task_id=task_id,
                        task_slice=task_slice,
                        target=target_values[0],
                        prediction=class_prediction_values[0],
                        probabilities=class_probability_values[0],
                        logits=logits_values[0],
                        num_patches_total=num_patches_total,
                        num_patches_used=num_patches_used,
                    ))
                    task_prediction_rows.append(_prediction_row(
                        context=artifact_context,
                        mode="task-il",
                        loader=loader,
                        sample_index=sample_index,
                        task_id=task_id,
                        task_slice=task_slice,
                        target=target_values[0],
                        prediction=task_prediction_values[0],
                        probabilities=task_probability_values[0],
                        logits=logits_values[0],
                        num_patches_total=num_patches_total,
                        num_patches_used=num_patches_used,
                    ))

            targets_array = np.concatenate(class_targets)
            class_predictions_array = np.concatenate(class_predictions_all)
            task_predictions_array = np.concatenate(task_predictions_all)
            class_probabilities_array = np.concatenate(class_probabilities_all)
            task_probabilities_array = np.concatenate(task_probabilities_all)
            class_metrics = evaluation_metrics(
                targets_array,
                class_predictions_array,
                class_probabilities_array,
                class_loss_sum / total,
            )
            task_metrics = evaluation_metrics(
                targets_array,
                task_predictions_array,
                task_probabilities_array,
                task_loss_sum / total,
            )
            class_metrics_all.append(class_metrics)
            task_metrics_all.append(task_metrics)
            elapsed = time.perf_counter() - evaluation_start
            _append_metric_artifact(
                artifact_context,
                "class_il",
                task_id,
                class_metrics,
                targets_array,
                class_predictions_array,
                np.arange(seen_count),
                class_prediction_rows,
                elapsed,
            )
            _append_metric_artifact(
                artifact_context,
                "task_il",
                task_id,
                task_metrics,
                targets_array,
                task_predictions_array,
                np.arange(task_slice.start, task_slice.stop),
                task_prediction_rows,
                elapsed,
            )
            class_correct_total += class_correct
            task_correct_total += task_correct
            sample_total += total
            global_targets.append(targets_array)
            global_predictions.append(class_predictions_array)
            global_probabilities.append(class_probabilities_array)

    global_metrics = evaluation_metrics(
        np.concatenate(global_targets),
        np.concatenate(global_predictions),
        np.concatenate(global_probabilities),
        np.average(
            [metric["loss"] for metric in class_metrics_all],
            weights=[metric["n"] for metric in class_metrics_all],
        ),
    )

    model.net.train(status)
    return EvaluationResult(
        class_il_accuracy=[metric["acc"] for metric in class_metrics_all],
        class_il_micro_accuracy=class_correct_total / sample_total,
        task_il_accuracy=[metric["acc"] for metric in task_metrics_all],
        task_il_micro_accuracy=task_correct_total / sample_total,
        task_auc=[metric["auroc"] for metric in task_metrics_all],
        global_seen_auc=global_metrics["auroc"],
        class_il_metrics=class_metrics_all,
        task_il_metrics=task_metrics_all,
        global_seen_metrics=global_metrics,
    )


def checkpoint_payload(model, dataset, fold: int):
    """Build a versioned checkpoint including non-module continual state."""
    payload = {
        "format_version": 2,
        "model_name": getattr(model, "NAME", None),
        "metadata": dataset.metadata(fold),
    }
    if bool(getattr(model, "CHECKPOINT_USES_STATE_DICT", True)):
        payload["state_dict"] = model.state_dict()
    if (
        bool(getattr(model, "CHECKPOINT_INCLUDE_OPTIMIZER", True))
        and getattr(model, "opt", None) is not None
    ):
        payload["optimizer_state"] = model.opt.state_dict()
    method_state = model.get_checkpoint_state()
    if method_state is not None:
        payload["method_state"] = method_state
    return payload


def load_checkpoint(model, path, dataset, fold: int):
    try:
        payload = torch.load(path, map_location=model.device, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location=model.device)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(f"Legacy/incomplete checkpoint is not compatible: {path}")
    saved_model = payload.get("model_name")
    if saved_model is not None and saved_model != getattr(model, "NAME", None):
        raise ValueError(
            f"Checkpoint method mismatch: saved={saved_model!r}, "
            f"expected={getattr(model, 'NAME', None)!r}"
        )
    expected = dataset.metadata(fold)
    actual = payload["metadata"]
    for key in (
        "fold",
        "task_order",
        "task_num_classes",
        "class_offsets",
        "total_num_classes",
        "optimizer_config",
        "backbone_config",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: saved={actual.get(key)!r}, expected={expected.get(key)!r}"
            )
    if bool(getattr(model, "CHECKPOINT_USES_STATE_DICT", True)):
        if "state_dict" not in payload:
            raise ValueError(f"Checkpoint is missing model state_dict: {path}")
        model.load_state_dict(payload["state_dict"])
    if "optimizer_state" in payload and getattr(model, "opt", None) is not None:
        model.opt.load_state_dict(payload["optimizer_state"])
    if "method_state" in payload:
        model.load_checkpoint_state(payload["method_state"], strict=True)
    elif type(model).get_checkpoint_state is not ContinualModel.get_checkpoint_state:
        raise ValueError(f"Checkpoint is missing method_state: {path}")


class EarlyStopping:
    """Save the best validation checkpoint and optionally stop on a plateau.

    ``min_epoch`` is expressed as a one-based count of completed epochs.  Best
    checkpoint restoration remains active even when ``enabled`` is false.
    """

    def __init__(
        self,
        patience=20,
        min_epoch=1,
        min_delta=0.0,
        verbose=False,
        enabled=True,
    ):
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)
        self.enabled = bool(enabled)
        if self.patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.min_epoch < 0:
            raise ValueError("early_stopping_min_epoch must be non-negative")
        if self.min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, checkpoint_path, dataset, fold):
        score = -float(val_loss)
        improved = (
            self.best_score is None
            or float(val_loss) < self.val_loss_min - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                tqdm.write(
                    "Validation loss decreased "
                    f"({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
                )
            torch.save(checkpoint_payload(model, dataset, fold), checkpoint_path)
            self.val_loss_min = float(val_loss)
        else:
            self.counter += 1
            if self.verbose and self.enabled:
                tqdm.write(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            completed_epochs = int(epoch) + 1
            if (
                self.enabled
                and self.counter >= self.patience
                and completed_epochs >= self.min_epoch
            ):
                self.early_stop = True


def early_stopping_from_args(args) -> EarlyStopping:
    """Build the shared early-stopping policy used by regular and Joint runs."""
    return EarlyStopping(
        patience=getattr(args, "early_stopping_patience", 10),
        min_epoch=getattr(args, "early_stopping_min_epoch", 1),
        min_delta=getattr(args, "early_stopping_min_delta", 0.0),
        verbose=getattr(args, "early_stopping_verbose", True),
        enabled=getattr(args, "early_stopping", True),
    )


def evaluate_val(model, dataset, task_id, epoch, checkpoint_path, fold, early_stopping=None):
    status = model.net.training
    model.net.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        val_bar = tqdm(
            dataset.val_loader,
            total=len(dataset.val_loader),
            desc=f"val task {task_id + 1}",
            leave=False,
            disable=bool(
                getattr(getattr(model, "args", None), "non_verbose", False)
            ),
        )
        for features, coords, patch_size, labels in val_bar:
            features, coords, patch_size = model.prepare_inputs(
                features, coords, patch_size, training=False
            )
            labels = labels.to(model.device)
            logits = model([features, coords, patch_size])[0]
            logits = logits[:, :dataset.seen_class_count(task_id)]
            total_loss += F.cross_entropy(logits, labels).item()
            batches += 1
    if batches == 0:
        raise ValueError(f"Validation split is empty for task {task_id}")
    val_loss = total_loss / batches
    model.net.train(status)
    if early_stopping is not None:
        early_stopping(epoch, val_loss, model, checkpoint_path, dataset, fold)
        if early_stopping.early_stop:
            tqdm.write(
                f"Early stopping after epoch {int(epoch) + 1}; "
                f"best validation loss={early_stopping.val_loss_min:.6f}"
            )
            return True
    return False


def _print_result(result: EvaluationResult):
    print(f"class-il task accuracy: {result.class_il_accuracy}")
    print(f"class-il macro/micro:   {result.class_il_macro_accuracy:.4f} / {result.class_il_micro_accuracy:.4f}")
    print(f"class-il task BAcc:     {result.class_il_balanced_accuracy}")
    print(f"class-il task macro F1: {result.class_il_macro_f1}")
    print(f"class-il weighted F1:   {result.class_il_weighted_f1}")
    print(f"class-il task AUROC:    {result.class_il_auc}")
    print(f"class-il task kappa:    {result.class_il_kappa}")
    print(f"class-il task loss:     {result.class_il_loss}")
    print(f"task-il task accuracy:  {result.task_il_accuracy}")
    print(f"task-il macro/micro:    {result.task_il_macro_accuracy:.4f} / {result.task_il_micro_accuracy:.4f}")
    print(f"task-il task BAcc:      {result.task_il_balanced_accuracy}")
    print(f"task-il task macro F1:  {result.task_il_macro_f1}")
    print(f"task-il weighted F1:    {result.task_il_weighted_f1}")
    print(f"task-il task AUROC:     {result.task_auc}")
    print(f"task-il task kappa:     {result.task_il_kappa}")
    print(f"task-il task loss:      {result.task_il_loss}")
    print(
        "global seen ACC/BAcc/macro-F1/weighted-F1/AUROC/kappa: "
        f"{result.global_seen_metrics['acc']:.4f} / "
        f"{result.global_seen_metrics['bacc']:.4f} / "
        f"{result.global_seen_metrics['macro_f1']:.4f} / "
        f"{result.global_seen_metrics['weighted_f1']:.4f} / "
        f"{result.global_seen_metrics['auroc']:.4f} / "
        f"{result.global_seen_metrics['kappa']:.4f}"
    )


def parameter_statistics(module: torch.nn.Module) -> Dict[str, int]:
    """Return architecture size using PyTorch's de-duplicated parameters."""
    parameters = list(module.parameters())
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
    }


def _cuda_device(model):
    device = torch.device(model.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return device


def _start_cuda_peak_tracking(model) -> None:
    device = _cuda_device(model)
    if device is not None:
        torch.cuda.reset_peak_memory_stats(device)


def _resource_statistics(
    model,
    parameters: Dict[str, int],
    *,
    initial_parameters: Dict[str, int] | None = None,
) -> Dict[str, object]:
    """Collect fold-level parameter and peak CUDA-memory statistics."""
    resources: Dict[str, object] = dict(parameters)
    initial = parameters if initial_parameters is None else initial_parameters
    resources.update({
        "initial_total_parameters": int(initial["total_parameters"]),
        "initial_trainable_parameters": int(initial["trainable_parameters"]),
        "final_total_parameters": int(parameters["total_parameters"]),
        "final_trainable_parameters": int(parameters["trainable_parameters"]),
        "parameter_growth": int(
            parameters["total_parameters"] - initial["total_parameters"]
        ),
    })
    device = _cuda_device(model)
    if device is None:
        resources.update({
            "device": str(model.device),
            "cuda_available": False,
            "peak_gpu_allocated_bytes": None,
            "peak_gpu_reserved_bytes": None,
            "peak_gpu_allocated_mib": None,
            "peak_gpu_reserved_mib": None,
        })
        return resources

    torch.cuda.synchronize(device)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    bytes_per_mib = 1024 ** 2
    resources.update({
        "device": str(device),
        "cuda_available": True,
        "peak_gpu_allocated_bytes": allocated,
        "peak_gpu_reserved_bytes": reserved,
        "peak_gpu_allocated_mib": float(allocated / bytes_per_mib),
        "peak_gpu_reserved_mib": float(reserved / bytes_per_mib),
    })
    return resources


def _write_resource_manifests(state) -> None:
    common = {
        **state["manifest_common"],
        "per_fold_resources": state["resource_usage"],
    }
    write_manifest(
        str(state["class_il"]["save_dir"]),
        {**common, "mode": "class-il-seen"},
    )
    write_manifest(
        str(state["task_il"]["save_dir"]),
        {**common, "mode": "task-il"},
    )


def _initialize_evaluation_run(args, dataset, model, fold):
    """Create one canonical artifact tree shared by all selected folds."""
    state = getattr(args, "_evaluation_artifact_state", None)
    if state is not None:
        return state
    root = Path("results") / args.exp_desc / "evaluation"
    total_classes = int(dataset.total_num_classes)
    state = {
        "class_il": initialize_artifacts(str(root / "class_il"), total_classes),
        "task_il": initialize_artifacts(str(root / "task_il"), total_classes),
        "training_times": {},
        "resource_usage": {},
    }
    selected_folds = list(getattr(args, "selected_folds", [fold]))
    common_manifest = {
        "backbone": getattr(args, "backbone", ""),
        "method": model.NAME,
        "model_name": getattr(args, "backbone_model_id", None),
        "backbone_revision": getattr(args, "backbone_revision", None),
        "num_folds": len(selected_folds),
        "folds": selected_folds,
        "num_tasks": dataset.N_TASKS,
        "task_order": list(dataset.task_order),
        "num_classes_per_task": list(dataset.task_num_classes),
        "total_classes": total_classes,
        "k": int(getattr(args, "backbone_max_patches", 0) or 0),
        "patch_size_fallback": int(
            getattr(args, "patch_size_level0_fallback", 1024)
        ),
        "seed": getattr(args, "seed", None),
        **parameter_statistics(model.net),
        "early_stopping": bool(getattr(args, "early_stopping", True)),
        "early_stopping_patience": int(
            getattr(args, "early_stopping_patience", 10)
        ),
        "early_stopping_min_epoch": int(
            getattr(args, "early_stopping_min_epoch", 1)
        ),
        "early_stopping_min_delta": float(
            getattr(args, "early_stopping_min_delta", 0.0)
        ),
        "early_stopping_verbose": bool(
            getattr(args, "early_stopping_verbose", True)
        ),
        "evaluate_fwt": bool(getattr(args, "evaluate_fwt", False)),
        "repo_commit": repo_commit(Path(__file__).resolve().parents[1]),
    }
    state["manifest_common"] = common_manifest
    _write_resource_manifests(state)
    args._evaluation_artifact_state = state
    return state


def _artifact_context(state, args, dataset, model, fold, after_task):
    return {
        "class_il": state["class_il"],
        "task_il": state["task_il"],
        "method": model.NAME,
        "fold": int(fold),
        "after_task": int(after_task),
        "task_order": list(dataset.task_order),
        "total_classes": int(dataset.total_num_classes),
        "k": int(getattr(args, "backbone_max_patches", 0) or 0),
        "seed": getattr(args, "seed", None),
    }


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace, fold: int) -> None:
    model.to(model.device)
    initial_parameters = parameter_statistics(model.net)
    print(
        "[resources] model parameters: "
        f"total={initial_parameters['total_parameters']:,}, "
        f"trainable={initial_parameters['trainable_parameters']:,}"
    )
    print(
        "[early-stopping] "
        f"enabled={bool(getattr(args, 'early_stopping', True))}, "
        f"patience={int(getattr(args, 'early_stopping_patience', 10))}, "
        f"min_epoch={int(getattr(args, 'early_stopping_min_epoch', 1))}, "
        f"min_delta={float(getattr(args, 'early_stopping_min_delta', 0.0))}"
    )
    _start_cuda_peak_tracking(model)
    class_results, task_results, auc_results = [], [], []
    csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME, fold, args.exp_desc) if args.csv_log else None
    tb_logger = TensorboardLogger(args, dataset.SETTING) if args.tensorboard else None

    evaluate_fwt = bool(getattr(args, "evaluate_fwt", False))
    random_result = None
    if evaluate_fwt:
        random_dataset = get_dataset(args)
        for task_id in range(random_dataset.N_TASKS):
            random_dataset.get_data_loaders(fold, task_id)
        random_result = evaluate(model, random_dataset)
        print(f"Random global AUC = {random_result.global_seen_auc}")

    artifact_state = _initialize_evaluation_run(args, dataset, model, fold)
    fold_training_time = 0.0

    results_dir = Path("checkpoints") / args.exp_desc / f"fold_{fold}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if model.NAME == "joint":
        training_start = time.perf_counter()
        dataset.get_joint_data_loaders(fold)
        model.end_task(dataset, fold)
        fold_training_time += time.perf_counter() - training_start
        result = evaluate(
            model,
            dataset,
            artifact_context=_artifact_context(
                artifact_state, args, dataset, model, fold, dataset.N_TASKS - 1
            ),
        )
        class_results.append(result.class_il_accuracy)
        task_results.append(result.task_il_accuracy)
        auc_results.append(result.task_auc)
        _print_result(result)
        if csv_logger:
            csv_logger.log_result(result)
    else:
        for task_id in range(dataset.N_TASKS):
            train_loader, _, _ = dataset.get_data_loaders(fold, task_id)
            if evaluate_fwt and task_id > 0 and class_results:
                before = evaluate(model, dataset, last=True)
                class_results[-1].append(before.class_il_accuracy[-1])
                task_results[-1].append(before.task_il_accuracy[-1])
                auc_results[-1].append(before.task_auc[-1])

            training_start = time.perf_counter()
            if hasattr(model, "begin_task"):
                model.begin_task(dataset)
            scheduler = dataset.get_scheduler(model, args)
            early_stopping = early_stopping_from_args(args)
            checkpoint_path = results_dir / f".task{task_id}_best.pt"
            final_checkpoint_path = results_dir / f"task{task_id}_checkpoint.pt"

            if getattr(model.net, "supports_ssl", False) and task_id == 0:
                ssl_epochs = 10
                for ssl_epoch in tqdm(
                    range(ssl_epochs),
                    desc=f"fold {fold} task {task_id + 1} SSL",
                    leave=False,
                    disable=bool(getattr(args, "non_verbose", False)),
                ):
                    ssl_loss = 0.0
                    ssl_updates = 0
                    ssl_batches = tqdm(
                        _iter_logical_batches(train_loader, 1),
                        total=len(train_loader),
                        desc=f"SSL epoch {ssl_epoch + 1}/{ssl_epochs}",
                        leave=False,
                        disable=bool(getattr(args, "non_verbose", False)),
                    )
                    for raw_group in ssl_batches:
                        ssl_loss += _loss_value(
                            _observe_logical_batch(
                                model, raw_group, task_id, ssl=True
                            )
                        )
                        ssl_updates += 1
                        ssl_batches.set_postfix(
                            loss=f"{ssl_loss / ssl_updates:.4f}", refresh=False
                        )

            epoch_bar = tqdm(
                range(model.args.n_epochs),
                desc=f"fold {fold} task {task_id + 1}/{dataset.N_TASKS}",
                leave=False,
                disable=bool(getattr(args, "non_verbose", False)),
            )
            for epoch in epoch_bar:
                model.net.train()
                if hasattr(model, "begin_epoch"):
                    model.begin_epoch(task_id, epoch)
                bags_per_update = int(getattr(args, "bags_per_update", 1) or 1)
                logical_total = max(1, math.ceil(len(train_loader) / bags_per_update))
                epoch_loss = 0.0
                epoch_updates = 0
                batch_bar = tqdm(
                    _iter_logical_batches(train_loader, bags_per_update),
                    total=logical_total,
                    desc=f"epoch {epoch + 1}/{model.args.n_epochs}",
                    leave=False,
                    disable=bool(getattr(args, "non_verbose", False)),
                )
                for batch_index, raw_group in enumerate(batch_bar):
                    loss = _loss_value(
                        _observe_logical_batch(model, raw_group, task_id, ssl=False)
                    )
                    epoch_loss += loss
                    epoch_updates += 1
                    batch_bar.set_postfix(loss=f"{loss:.4f}", refresh=False)
                    if tb_logger:
                        tb_logger.log_loss(loss, args, epoch, task_id, batch_index)
                average_loss = epoch_loss / max(epoch_updates, 1)
                epoch_bar.set_postfix(loss=f"{average_loss:.4f}", refresh=False)
                if not bool(getattr(args, "non_verbose", False)):
                    tqdm.write(
                        f"[train] fold={fold} task={task_id + 1}/{dataset.N_TASKS} "
                        f"epoch={epoch + 1}/{model.args.n_epochs} "
                        f"avg_loss={average_loss:.4f} updates={epoch_updates}"
                    )
                if hasattr(model, "end_epoch"):
                    model.end_epoch(task_id, epoch)
                if evaluate_val(model, dataset, task_id, epoch, checkpoint_path, fold, early_stopping):
                    break
                if scheduler is not None:
                    scheduler.step()

            load_checkpoint(model, checkpoint_path, dataset, fold)
            if hasattr(model, "save_buffer") and bool(
                getattr(model, "replay_enabled", True)
            ):
                for features, coords, patch_size, labels in train_loader:
                    features, coords, patch_size = model.prepare_inputs(
                        features, coords, patch_size, training=False
                    )
                    model.save_buffer(
                        features,
                        coords,
                        patch_size,
                        labels.to(model.device),
                        task_id,
                    )
            if hasattr(model, "end_task"):
                model.end_task(dataset)

            # The early-stopping file is deliberately temporary: canonical
            # task checkpoints are written only after replay/teacher/key state
            # has been finalized for the completed task.
            torch.save(
                checkpoint_payload(model, dataset, fold),
                final_checkpoint_path,
            )
            checkpoint_path.unlink(missing_ok=True)
            fold_training_time += time.perf_counter() - training_start

            result = evaluate(
                model,
                dataset,
                artifact_context=_artifact_context(
                    artifact_state, args, dataset, model, fold, task_id
                ),
            )
            class_results.append(list(result.class_il_accuracy))
            task_results.append(list(result.task_il_accuracy))
            auc_results.append(list(result.task_auc))
            _print_result(result)
            if csv_logger:
                csv_logger.log_result(result)

    artifact_state["training_times"][int(fold)] = fold_training_time
    final_parameters = parameter_statistics(model.net)
    resources = _resource_statistics(
        model, final_parameters, initial_parameters=initial_parameters
    )
    artifact_state["resource_usage"][int(fold)] = resources
    if resources["parameter_growth"]:
        print(
            f"[resources] fold {fold} dynamic parameter growth: "
            f"{resources['initial_total_parameters']:,} -> "
            f"{resources['final_total_parameters']:,} "
            f"({resources['parameter_growth']:+,})"
        )
    if resources["cuda_available"]:
        print(
            f"[resources] fold {fold} peak CUDA memory: "
            f"allocated={resources['peak_gpu_allocated_mib']:.2f} MiB, "
            f"reserved={resources['peak_gpu_reserved_mib']:.2f} MiB"
        )
    else:
        print(f"[resources] fold {fold}: CUDA unavailable; peak GPU memory is N/A")
    _write_resource_manifests(artifact_state)
    finalize_artifacts(
        artifact_state["class_il"], model.NAME, artifact_state["training_times"],
        artifact_state["resource_usage"],
    )
    finalize_artifacts(
        artifact_state["task_il"], model.NAME, artifact_state["training_times"],
        artifact_state["resource_usage"],
    )

    if csv_logger:
        csv_logger.add_bwt(class_results, task_results, auc_results)
        csv_logger.add_forgetting(class_results, task_results, auc_results)
        if evaluate_fwt and model.NAME != "joint":
            csv_logger.add_fwt(
                class_results, random_result.class_il_accuracy,
                task_results, random_result.task_il_accuracy,
                auc_results, random_result.task_auc,
            )
        csv_logger.write(dict(vars(args)))
    if tb_logger:
        tb_logger.close()
