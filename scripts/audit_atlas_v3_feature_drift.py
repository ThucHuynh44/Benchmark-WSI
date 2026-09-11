"""Measure replay-free FEATHER feature drift against the frozen encoder.

This is an offline diagnostic. It forwards fixed old-task test WSI through the
frozen FEATHER encoder and the ACL encoder at each later checkpoint. No model
or stored statistics are updated.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import get_model
from scripts.audit_atlas_v3_transport_correction import (
    SETTINGS,
    _arguments,
    _encoder_difference,
    _load_payload,
    _parse_indices,
    _restore_for_audit,
)


FROZEN_SETTING = "atlasv3_frozen_proto_oas_lda"
DEFAULT_OUTPUT = REPO_ROOT / "results/diagnostics/atlas_v3_drift_forgetting/feature_drift"


def _acl_checkpoint(setting: str, fold: int, after_task: int) -> Path:
    return (
        REPO_ROOT
        / "checkpoints/ablations/atlas_v3_acl"
        / setting
        / f"fold_{fold}/fold_{fold}"
        / f"task{after_task}_checkpoint.pt"
    )


def _frozen_checkpoint(fold: int, after_task: int) -> Path:
    return (
        REPO_ROOT
        / "checkpoints/ablations/atlas_v3"
        / FROZEN_SETTING
        / f"fold_{fold}/fold_{fold}"
        / f"task{after_task}_checkpoint.pt"
    )


def _backbone_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    prefix = "net.backbone."
    state = {
        key[len(prefix) :]: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError("Checkpoint contains no FEATHER backbone state")
    return state


@torch.no_grad()
def _encode(model, source_loader, max_wsis: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        source_loader.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=source_loader.collate_fn,
    )
    values, labels = [], []
    model.net.eval()
    for index, batch in enumerate(loader):
        if max_wsis > 0 and index >= max_wsis:
            break
        features, coords, patch_size = model.prepare_inputs(
            batch.features,
            batch.coords,
            batch.patch_size_level0,
            training=False,
        )
        embedding = model.net.encode(features, coords, patch_size)
        values.append(F.normalize(embedding.float(), dim=1, eps=1.0e-8).cpu())
        labels.append(int(batch.labels.reshape(-1)[0]))
    if not values:
        raise RuntimeError("Feature-drift audit encountered an empty test split")
    return torch.cat(values), torch.as_tensor(labels, dtype=torch.long)


def _summary_row(
    reference: torch.Tensor,
    current: torch.Tensor,
    labels: torch.Tensor,
    *,
    fold: int,
    after_task: int,
    eval_task: int,
    task_name: str,
) -> dict[str, Any]:
    if reference.shape != current.shape:
        raise RuntimeError(
            f"Embedding shape mismatch: frozen={reference.shape} ACL={current.shape}"
        )
    distance = (1.0 - (reference * current).sum(1)).clamp_min(0.0).numpy()
    label_values = labels.numpy()
    class_means = [
        float(distance[label_values == label].mean())
        for label in sorted(set(label_values.tolist()))
    ]
    return {
        "audit_only": True,
        "fold": fold,
        "after_task": after_task,
        "eval_task": eval_task,
        "task_name": task_name,
        "task_age": after_task - eval_task,
        "n_wsis": int(distance.size),
        "n_classes": len(class_means),
        "frozen_feature_drift": 0.0,
        "acl_feature_drift_micro": float(distance.mean()),
        "acl_feature_drift_macro": float(np.mean(class_means)),
        "acl_feature_drift_std": float(distance.std()),
        "acl_feature_drift_median": float(np.median(distance)),
        "acl_feature_drift_p95": float(np.percentile(distance, 95.0)),
        # Static and gated transport share this encoder exactly. Transport only
        # changes historical statistics after adaptation.
        "static_feature_drift_macro": float(np.mean(class_means)),
        "gated_feature_drift_macro": float(np.mean(class_means)),
        "static_gated_encoder_exact": 1,
    }


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("Feature-drift audit produced no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_fold(
    fold: int,
    after_tasks: list[int],
    *,
    config: str,
    output_root: Path,
    max_wsis_per_task: int,
    overwrite: bool,
) -> None:
    first = _load_payload(_acl_checkpoint(SETTINGS["static"], fold, after_tasks[0]))
    args, dataset = _arguments(first, config, fold)
    model = get_model(args, None, dataset.get_loss(), dataset.get_transform())
    model.to(model.device)

    frozen_payload = _load_payload(_frozen_checkpoint(fold, max(after_tasks)))
    frozen_state = _backbone_state(frozen_payload)
    frozen_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for after_task in after_tasks:
        output = output_root / f"fold_{fold}" / f"task_{after_task}.csv"
        if output.is_file() and not overwrite:
            print(f"[skip] {output}")
            continue

        static_payload = _load_payload(
            _acl_checkpoint(SETTINGS["static"], fold, after_task)
        )
        gated_payload = _load_payload(
            _acl_checkpoint(SETTINGS["gated"], fold, after_task)
        )
        exact, maximum = _encoder_difference(
            static_payload["state_dict"], gated_payload["state_dict"]
        )
        if not exact:
            raise RuntimeError(
                f"Static/gated encoder mismatch fold={fold} task={after_task}; "
                f"max_abs_difference={maximum}"
            )

        _restore_for_audit(model, static_payload)
        current_state = _backbone_state(static_payload)
        rows = []
        for eval_task in range(after_task):
            _, _, test_loader = dataset.get_data_loaders(fold, eval_task)
            if eval_task not in frozen_cache:
                model.net.backbone.load_state_dict(frozen_state, strict=True)
                frozen_cache[eval_task] = _encode(
                    model, test_loader, max_wsis_per_task
                )
            frozen_embeddings, frozen_labels = frozen_cache[eval_task]

            model.net.backbone.load_state_dict(current_state, strict=True)
            current_embeddings, current_labels = _encode(
                model, test_loader, max_wsis_per_task
            )
            if not torch.equal(frozen_labels, current_labels):
                raise RuntimeError("Frozen and ACL passes used different WSI ordering")
            rows.append(
                _summary_row(
                    frozen_embeddings,
                    current_embeddings,
                    current_labels,
                    fold=fold,
                    after_task=after_task,
                    eval_task=eval_task,
                    task_name=str(dataset.task_order[eval_task]),
                )
            )
        _write(output, rows)
        print(
            f"[feature-drift] fold={fold} after_task={after_task} "
            f"old_tasks={len(rows)} output={output}"
        )
        del static_payload, gated_payload, rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/methods.yaml")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--after-tasks", default="1-9")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--max-wsis-per-task",
        type=int,
        default=0,
        help="0 uses the complete test split; positive values are for smoke tests only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_wsis_per_task < 0:
        raise ValueError("max-wsis-per-task must be non-negative")
    folds = _parse_indices(args.folds, minimum=0, maximum=9)
    after_tasks = _parse_indices(args.after_tasks, minimum=1, maximum=9)
    output_root = Path(args.output_root).expanduser().resolve()
    for fold in folds:
        run_fold(
            fold,
            after_tasks,
            config=args.config,
            output_root=output_root,
            max_wsis_per_task=args.max_wsis_per_task,
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
