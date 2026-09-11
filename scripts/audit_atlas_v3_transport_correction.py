"""Paired offline oracle audit of ATLAS-v3 historical-statistics transport.

The audit compares static, ungated, and adaptively gated stored statistics
against old-train statistics recomputed with the current encoder.  Old WSI are
opened only by this post-training diagnostic and never update a model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.experiment_loader import config_to_argv, load_experiment_config
from models import get_model
from utils.main import _prepare_fold


SETTINGS = {
    "static": "atlasv3_acl_normalized_oas_static",
    "ungated": "atlasv3_acl_transport_normalized_oas",
    "gated": "atlasv3_acl_gated_transport_normalized_oas_no_histneg",
}
EXPECTED_MODES = {
    "static": "normalized_oas_static",
    "ungated": "transport_normalized_oas",
    "gated": "gated_transport_normalized_oas_no_histneg",
}
DEFAULT_OUTPUT = REPO_ROOT / "results/diagnostics/atlas_v3_transport_correction"
LEGACY_STATE_KEYS = {
    "net.class_task",
    "net.hist_reliability",
    "net.uncertainty_ema",
}


def _parse_indices(value: str, *, minimum: int, maximum: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(minimum, maximum + 1))
    output = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(part) for part in token.split("-", 1))
            if stop < start:
                raise ValueError(f"Invalid range {token!r}")
            output.update(range(start, stop + 1))
        else:
            output.add(int(token))
    if not output or min(output) < minimum or max(output) > maximum:
        raise ValueError(f"Indices must be within {minimum}..{maximum}")
    return sorted(output)


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint(setting: str, fold: int, after_task: int) -> Path:
    description = f"ablations/atlas_v3_acl/{setting}/fold_{fold}"
    return (
        REPO_ROOT
        / "checkpoints"
        / description
        / f"fold_{fold}"
        / f"task{after_task}_checkpoint.pt"
    )


def _arguments(payload: Mapping[str, Any], config: str, fold: int):
    configured = load_experiment_config(
        config, method="atlas_v3_acl", backbone="feather"
    )
    module = __import__("models.atlas_v3_acl", fromlist=["get_parser"])
    args = module.get_parser().parse_args(
        config_to_argv(configured)
        + ["--model", "atlas_v3_acl", "--backbone", "feather"]
    )
    saved = payload.get("method_state", {}).get("config", {})
    args.atlasv3_acl_mode = saved.get("mode", args.atlasv3_acl_mode)
    for name, value in saved.get("hyperparameters", {}).items():
        setattr(args, name, value)
    ablation = payload.get("ablation", {})
    args.ablation_id = ablation.get("id")
    args.ablation_group = ablation.get("group")
    args.ablation_config_hash = ablation.get("config_hash")
    dataset = _prepare_fold(args, fold)
    return args, dataset


def _validate_payload(payload: Mapping[str, Any], role: str, after_task: int) -> None:
    mode = payload.get("method_state", {}).get("config", {}).get("mode")
    if mode != EXPECTED_MODES[role]:
        raise ValueError(f"{role} checkpoint mode is {mode!r}, expected {EXPECTED_MODES[role]!r}")
    completed = int(payload.get("method_state", {}).get("completed_tasks", -1))
    if completed != after_task + 1:
        raise ValueError(
            f"{role} checkpoint completed_tasks={completed}, expected {after_task + 1}"
        )


def _restore_for_audit(model, payload: Mapping[str, Any]) -> None:
    """Restore an old checkpoint while allowing only intentionally pruned buffers."""

    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing or not unexpected.issubset(LEGACY_STATE_KEYS):
        raise RuntimeError(
            "Audit checkpoint state mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    method_state = payload.get("method_state")
    if not isinstance(method_state, Mapping):
        raise RuntimeError("Audit checkpoint is missing method_state")
    # The saved config may list hyperparameters removed during the ATLAS-v3
    # cleanup. Identity/mode/task completion are validated above, while the
    # method hook restores the task counters needed for inference.
    model.load_checkpoint_state(method_state, strict=False)


def _encoder_difference(
    reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> tuple[bool, float]:
    keys = sorted(key for key in reference if key.startswith("net.backbone."))
    if not keys or keys != sorted(
        key for key in candidate if key.startswith("net.backbone.")
    ):
        return False, math.inf
    maximum = 0.0
    exact = True
    for key in keys:
        left, right = reference[key], candidate[key]
        if left.shape != right.shape:
            return False, math.inf
        if not torch.equal(left, right):
            exact = False
            maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return exact, maximum


@torch.no_grad()
def _encode(model, source_loader) -> tuple[torch.Tensor, torch.Tensor]:
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
    if not values:
        raise RuntimeError("Oracle audit encountered an empty old-train split")
    return torch.cat(values).float(), torch.as_tensor(labels, dtype=torch.long)


def _psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix.float() + matrix.float().t())
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(0.0).sqrt()) @ vectors.t()


def _bures(left: torch.Tensor, right: torch.Tensor) -> float:
    left_root = _psd_sqrt(left)
    middle = _psd_sqrt(left_root @ right.float() @ left_root)
    squared = left.trace() + right.trace() - 2.0 * middle.trace()
    return float(squared.clamp_min(0.0).sqrt())


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1.0e-12 else math.nan


def _direction(estimate: torch.Tensor, target: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(estimate) * torch.linalg.vector_norm(target))
    if denominator <= 1.0e-12:
        return math.nan
    return float(torch.dot(estimate, target) / denominator)


def _stored(payload: Mapping[str, Any], label: int) -> dict[str, Any]:
    state = payload["state_dict"]
    count = int(state["net.raw_count"][label])
    if count <= 0:
        raise RuntimeError(f"Stored statistics are missing class {label}")
    return {
        "prototype": F.normalize(
            state["net.prototype_bank"][label].float(), dim=0, eps=1.0e-8
        ),
        "mean": state["net.raw_mean"][label].float(),
        "covariance": state["net.raw_scatter"][label].float()
        / float(max(count - 1, 1)),
        "count": count,
    }


def _method_metrics(
    stored: Mapping[str, Any], oracle_mean: torch.Tensor, oracle_covariance: torch.Tensor
) -> dict[str, float]:
    oracle_proto = F.normalize(oracle_mean.float(), dim=0, eps=1.0e-8)
    prototype = stored["prototype"]
    cosine = float(torch.dot(prototype, oracle_proto).clamp(-1.0, 1.0))
    return {
        "prototype_cosine": cosine,
        "prototype_cosine_distance": 1.0 - cosine,
        "prototype_l2": float(torch.linalg.vector_norm(prototype - oracle_proto)),
        "mean_relative_error": float(
            torch.linalg.vector_norm(stored["mean"] - oracle_mean)
            / torch.linalg.vector_norm(oracle_mean).clamp_min(1.0e-8)
        ),
        "covariance_relative_frobenius": float(
            torch.linalg.matrix_norm(stored["covariance"] - oracle_covariance)
            / torch.linalg.matrix_norm(oracle_covariance).clamp_min(1.0e-8)
        ),
        "covariance_bures": _bures(stored["covariance"], oracle_covariance),
    }


def _rows_for_task(model, dataset, payloads, fold: int, after_task: int) -> list[dict]:
    state_gated = payloads["gated"]["state_dict"]
    output = []
    for eval_task in range(after_task):
        train_loader, _, _ = dataset.get_data_loaders(fold, eval_task)
        raw, labels = _encode(model, train_loader)
        normalized = F.normalize(raw, dim=1, eps=1.0e-8)
        start = int(dataset.class_offsets[eval_task])
        stop = start + int(dataset.task_num_classes[eval_task])
        for label in range(start, stop):
            members = normalized[labels == label]
            if members.shape[0] == 0:
                raise RuntimeError(f"Oracle split has no samples for class {label}")
            oracle_mean = members.mean(0)
            centered = members - oracle_mean
            oracle_covariance = centered.t() @ centered / float(max(members.shape[0] - 1, 1))
            stored = {role: _stored(payload, label) for role, payload in payloads.items()}
            metrics = {
                role: _method_metrics(values, oracle_mean, oracle_covariance)
                for role, values in stored.items()
            }
            row: dict[str, Any] = {
                "audit_only": True,
                "fold": int(fold),
                "after_task": int(after_task),
                "eval_task": int(eval_task),
                "class_id": int(label),
                "class_age": int(after_task - eval_task),
                "sample_count": int(members.shape[0]),
                "stored_count": int(stored["static"]["count"]),
                "coverage": float(state_gated["net.last_coverage"][label]),
                "step_gate": float(state_gated["net.last_step_gate"][label]),
            }
            for role, values in metrics.items():
                row.update({f"{role}_{name}": value for name, value in values.items()})
            static_distance = metrics["static"]["prototype_cosine_distance"]
            static_covariance = metrics["static"]["covariance_relative_frobenius"]
            oracle_proto = F.normalize(oracle_mean, dim=0, eps=1.0e-8)
            true_step = oracle_proto - stored["static"]["prototype"]
            for role in ("ungated", "gated"):
                distance = metrics[role]["prototype_cosine_distance"]
                gain = static_distance - distance
                estimated_step = stored[role]["prototype"] - stored["static"]["prototype"]
                row[f"{role}_prototype_gain"] = gain
                row[f"{role}_recovery_rate"] = _safe_ratio(gain, static_distance)
                row[f"{role}_overcorrected"] = int(distance > static_distance + 1.0e-12)
                row[f"{role}_direction_alignment"] = _direction(estimated_step, true_step)
                row[f"{role}_step_magnitude_ratio"] = _safe_ratio(
                    float(torch.linalg.vector_norm(estimated_step)),
                    float(torch.linalg.vector_norm(true_step)),
                )
                row[f"{role}_covariance_gain"] = (
                    static_covariance
                    - metrics[role]["covariance_relative_frobenius"]
                )
            # This verifies that static really is the pre-transport bank under
            # the same encoder, rather than a separately adapted representation.
            row["static_prototype_matches_raw_mean"] = int(
                torch.allclose(
                    stored["static"]["prototype"],
                    F.normalize(stored["static"]["mean"], dim=0, eps=1.0e-8),
                    atol=1.0e-6,
                    rtol=1.0e-5,
                )
            )
            output.append(row)
    return output


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("Transport audit produced no old-class rows")
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
    overwrite: bool,
) -> None:
    first_checkpoint = _checkpoint(SETTINGS["static"], fold, after_tasks[0])
    first_payload = _load_payload(first_checkpoint)
    args, dataset = _arguments(first_payload, config, fold)
    model = get_model(args, None, dataset.get_loss(), dataset.get_transform())
    model.to(model.device)
    for after_task in after_tasks:
        output = output_root / f"fold_{fold}" / f"task_{after_task}.csv"
        if output.is_file() and not overwrite:
            print(f"[skip] {output}")
            continue
        payloads = {
            role: _load_payload(_checkpoint(setting, fold, after_task))
            for role, setting in SETTINGS.items()
        }
        for role, payload in payloads.items():
            _validate_payload(payload, role, after_task)
        reference_state = payloads["static"]["state_dict"]
        for role in ("ungated", "gated"):
            exact, maximum = _encoder_difference(
                reference_state, payloads[role]["state_dict"]
            )
            if not exact:
                raise RuntimeError(
                    f"Encoder mismatch fold={fold} task={after_task} role={role}; "
                    f"max_abs_difference={maximum}"
                )
        # Restore the static checkpoint solely to obtain the shared current
        # encoder used for oracle forward passes.
        _restore_for_audit(model, payloads["static"])
        rows = _rows_for_task(model, dataset, payloads, fold, after_task)
        _write(output, rows)
        print(
            f"[audit] fold={fold} after_task={after_task} old_classes={len(rows)} "
            f"output={output}"
        )
        del payloads, rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/methods.yaml")
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--after-tasks",
        default="1-9",
        help="Zero-based checkpoints; task 0 is excluded because it has no old classes.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    folds = _parse_indices(args.folds, minimum=0, maximum=9)
    after_tasks = _parse_indices(args.after_tasks, minimum=1, maximum=9)
    output_root = Path(args.output_root).expanduser().resolve()
    for fold in folds:
        run_fold(
            fold,
            after_tasks,
            config=args.config,
            output_root=output_root,
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
