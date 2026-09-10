"""Offline oracle audit for ATLAS-v3 ACL historical class statistics.

This script deliberately reads old train splits only after training has
finished.  Its CSV is diagnostic-only and is never consumed by model fitting,
registry selection, or task-boundary hooks.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.experiment_loader import config_to_argv, load_experiment_config
from models import get_model
from utils.main import _prepare_fold
from utils.training import load_checkpoint


def _load_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix.float() + matrix.float().t())
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(0.0).sqrt()) @ vectors.t()


def _bures_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_root = _psd_sqrt(left)
    middle_root = _psd_sqrt(left_root @ right.float() @ left_root)
    squared = left.trace() + right.trace() - 2.0 * middle_root.trace()
    return float(squared.clamp_min(0.0).sqrt())


@torch.no_grad()
def _encode(model, source_loader):
    loader = DataLoader(
        source_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=source_loader.collate_fn,
    )
    values, labels = [], []
    model.net.eval()
    for batch in loader:
        features, coords, patch_size = model.prepare_inputs(
            batch.features,
            batch.coords,
            batch.patch_size_level0,
            training=False,
        )
        values.append(model.net.encode(features, coords, patch_size).cpu())
        labels.append(int(batch.labels.reshape(-1)[0]))
    return torch.cat(values), torch.as_tensor(labels, dtype=torch.long)


def _rows(model, dataset, fold: int, after_task: int):
    output = []
    for eval_task in range(after_task + 1):
        train_loader, _, _ = dataset.get_data_loaders(fold, eval_task)
        raw, labels = _encode(model, train_loader)
        normalized = F.normalize(raw.float(), dim=1, eps=1.0e-8)
        start = dataset.class_offsets[eval_task]
        stop = start + dataset.task_num_classes[eval_task]
        for label in range(start, stop):
            mask = labels == label
            class_norm = normalized[mask]
            oracle_proto = F.normalize(class_norm.mean(0), dim=0, eps=1.0e-8)
            stored_proto = F.normalize(model.net.prototype_bank[label].cpu(), dim=0, eps=1.0e-8)
            cosine = float((oracle_proto * stored_proto).sum().clamp(-1.0, 1.0))
            row = {
                "audit_only": True,
                "fold": int(fold),
                "after_task": int(after_task),
                "eval_task": int(eval_task),
                "class_id": int(label),
                "class_age": int(after_task - eval_task),
                "sample_count": int(mask.sum()),
                "prototype_cosine": cosine,
                "prototype_angle_rad": math.acos(cosine),
                "prototype_l2": float(torch.linalg.vector_norm(oracle_proto - stored_proto)),
                "coverage": float(model.net.last_coverage[label].cpu()),
                "step_gate": float(model.net.last_step_gate[label].cpu()),
                "raw_mean_relative_error": "",
                "covariance_relative_frobenius": "",
                "covariance_bures": "",
            }
            if model.mode in {
                "normalized_oas_static", "transport_normalized_oas",
                "gated_transport_normalized_oas_no_histneg",
            }:
                class_statistics = class_norm
                oracle_mean = class_statistics.mean(0)
                centered = class_statistics - oracle_mean
                oracle_covariance = centered.t() @ centered / float(max(class_statistics.shape[0] - 1, 1))
                stored_mean = model.net.raw_mean[label].cpu().float()
                stored_covariance = model.net.raw_scatter[label].cpu().float() / float(max(int(model.net.raw_count[label]) - 1, 1))
                row["raw_mean_relative_error"] = float(
                    torch.linalg.vector_norm(stored_mean - oracle_mean)
                    / torch.linalg.vector_norm(oracle_mean).clamp_min(1.0e-8)
                )
                row["covariance_relative_frobenius"] = float(
                    torch.linalg.matrix_norm(stored_covariance - oracle_covariance)
                    / torch.linalg.matrix_norm(oracle_covariance).clamp_min(1.0e-8)
                )
                row["covariance_bures"] = _bures_distance(
                    stored_covariance, oracle_covariance
                )
            output.append(row)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/methods.yaml")
    parser.add_argument("--exp-desc", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(10))
    parser.add_argument("--after-task", type=int, required=True)
    parser.add_argument("--output", default=None)
    cli = parser.parse_args(argv)
    checkpoint = REPO_ROOT / "checkpoints" / cli.exp_desc / f"fold_{cli.fold}" / f"task{cli.after_task}_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = _load_payload(checkpoint)
    method_state = payload.get("method_state", {})
    saved_config = method_state.get("config", {})
    configured = load_experiment_config(cli.config, method="atlas_v3_acl", backbone="feather")
    module = __import__("models.atlas_v3_acl", fromlist=["get_parser"])
    arguments = config_to_argv(configured)
    arguments.extend(["--model", "atlas_v3_acl", "--backbone", "feather"])
    args = module.get_parser().parse_args(arguments)
    args.atlasv3_acl_mode = saved_config.get("mode", args.atlasv3_acl_mode)
    for name, value in saved_config.get("hyperparameters", {}).items():
        setattr(args, name, value)
    ablation = payload.get("ablation", {})
    args.ablation_id = ablation.get("id")
    args.ablation_group = ablation.get("group")
    args.ablation_config_hash = ablation.get("config_hash")
    dataset = _prepare_fold(args, cli.fold)
    if not 0 <= cli.after_task < dataset.N_TASKS:
        raise ValueError("after-task is outside the dataset task sequence")
    model = get_model(args, None, dataset.get_loss(), dataset.get_transform())
    model.to(model.device)
    load_checkpoint(model, checkpoint, dataset, cli.fold)
    rows = _rows(model, dataset, cli.fold, cli.after_task)
    output = Path(cli.output) if cli.output else REPO_ROOT / "results" / cli.exp_desc / "evaluation" / "oracle_audit" / f"fold_{cli.fold}_task_{cli.after_task}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} audit rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
