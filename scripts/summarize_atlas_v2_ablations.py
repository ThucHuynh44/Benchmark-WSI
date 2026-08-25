"""Summarize ATLAS-v2 without ranking or automatically selecting a winner."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.atlas_v2_registry import (
    COMEL_SETTING_ID, PROTO_FACTORIAL_IDS, SETTING_IDS, load_registry,
)
from scripts.build_cl_table import _read_eval_matrix, _rows_by_key, _sequential_fold_metrics
from scripts.run_atlas_v2_ablations import experiment_desc, inspect_run


METRICS = ("mACC", "bACC", "masked_bACC", "BWT", "FGT")
MEMORY_FIELDS = ("retained_wsis", "retained_patch_rows", "replay_memory_mib")


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


def _mean_std(values: Iterable[Any]) -> tuple[float, float]:
    values = _finite(values)
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values)


def _memory_from_checkpoint(registry, variant, fold: int) -> dict:
    if not variant["overrides"]["atlasv2_replay"]:
        return {"retained_wsis": 0, "retained_patch_rows": 0, "replay_memory_mib": 0.0}
    root = REPO_ROOT / "checkpoints" / experiment_desc(registry, variant["id"], fold) / f"fold_{fold}"
    paths = sorted(root.glob("task*_checkpoint.pt"))
    if not paths:
        return {field: math.nan for field in MEMORY_FIELDS}
    try:
        payload = torch.load(paths[-1], map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(paths[-1], map_location="cpu")
    memory = payload.get("method_state", {}).get("memory", {})
    entries = memory.get("entries", []) if isinstance(memory, dict) else []
    rows = sum(int(entry["features"].shape[0]) for entry in entries)
    byte_count = sum(
        tensor.numel() * tensor.element_size()
        for entry in entries
        for tensor in (entry["features"], entry["coords"], entry["patch_size"], entry["label"])
    )
    return {
        "retained_wsis": len(entries), "retained_patch_rows": rows,
        "replay_memory_mib": byte_count / (1024.0 ** 2),
    }


def collect(registry) -> list[dict]:
    rows = []
    for variant in registry["variants"].values():
        for fold in range(10):
            status = inspect_run(registry, variant, fold)
            row = {"variant_id": variant["id"], "fold": fold, "status": status}
            if status == "complete":
                run_dir = REPO_ROOT / "results" / experiment_desc(registry, variant["id"], fold)
                manifest = json.loads(
                    (run_dir / "evaluation/class_il/run_manifest.json").read_text(encoding="utf-8")
                )
                num_tasks = int(manifest["num_tasks"])
                class_rows = _read_eval_matrix(run_dir / "evaluation/class_il/eval_matrix.csv")
                task_rows = _read_eval_matrix(run_dir / "evaluation/task_il/eval_matrix.csv")
                row.update(_sequential_fold_metrics(
                    _rows_by_key(class_rows, fold), _rows_by_key(task_rows, fold), num_tasks
                ))
                row.update(_memory_from_checkpoint(registry, variant, fold))
            rows.append(row)
    return rows


def summarize(registry, rows: Sequence[dict]) -> list[dict]:
    output = []
    for variant in registry["variants"].values():
        complete = [
            row for row in rows
            if row["variant_id"] == variant["id"] and row["status"] == "complete"
        ]
        summary = {
            "variant_id": variant["id"], "completed_folds": len(complete),
            "status": "complete" if len(complete) == 10 else "incomplete",
        }
        for field in (*METRICS, *MEMORY_FIELDS):
            summary[f"{field}_mean"], summary[f"{field}_std"] = _mean_std(
                row.get(field) for row in complete
            )
        output.append(summary)
    return output


def _format(row: Mapping[str, Any], metric: str) -> str:
    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    return "" if not math.isfinite(float(mean)) else f"{mean:.4f} ± {std:.4f}"


def markdown(summaries: Sequence[dict]) -> str:
    by_id = {row["variant_id"]: row for row in summaries}

    def table(title: str, ids: Sequence[str]) -> list[str]:
        lines = [
            f"# {title}", "",
            "| Setting | Folds | mACC | bACC | Masked bACC | BWT | FGT |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for variant_id in ids:
            row = by_id[variant_id]
            values = [_format(row, metric) for metric in METRICS]
            lines.append(f"| {variant_id} | {row['completed_folds']} | " + " | ".join(values) + " |")
        lines.append("")
        return lines

    lines = [
        "Primary classification metric: **bACC**. Primary stability metrics: **BWT / FGT**.",
        "Supporting metric: mACC. Diagnostic metric: Masked bACC.",
        "No winner is selected automatically.", "",
    ]
    lines.extend(table("ATLAS-v2 Additive Ladder", SETTING_IDS[:5]))
    lines.extend(table("ATLAS-v2 Frozen Baseline", (SETTING_IDS[5],)))
    lines.extend(table("ATLAS-v2 Semantic Extensions", (SETTING_IDS[4], SETTING_IDS[6], SETTING_IDS[7])))
    lines.extend(table("ATLAS-v2 LoRA Geometry Extension", (SETTING_IDS[8],)))
    lines.extend(table("ATLAS-v2 CoMEL LoRA Strategy", (COMEL_SETTING_ID,)))
    lines.extend(table("ATLAS-v2 Prototype LoRA × Replay Factorial", PROTO_FACTORIAL_IDS))
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(REPO_ROOT / "configs/atlas_v2_ablations.yaml"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results/ablations/atlas_v2/summary"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    rows = collect(registry)
    summaries = summarize(registry, rows)
    output = Path(args.output).expanduser().resolve()
    per_fold_fields = ["variant_id", "fold", "status", *METRICS, *MEMORY_FIELDS]
    summary_fields = [
        "variant_id", "completed_folds", "status",
        *[name for field in (*METRICS, *MEMORY_FIELDS) for name in (f"{field}_mean", f"{field}_std")],
    ]
    _write_csv(output / "atlas_v2_per_fold.csv", rows, per_fold_fields)
    _write_csv(output / "atlas_v2_summary.csv", summaries, summary_fields)
    (output / "atlas_v2_tables.md").write_text(markdown(summaries), encoding="utf-8")
    incomplete = [row for row in summaries if row["status"] != "complete"]
    print(
        f"Wrote {len(SETTING_IDS)} ATLAS-v2 settings to {output}; "
        f"incomplete={len(incomplete)}"
    )
    return int(bool(args.strict and incomplete))


if __name__ == "__main__":
    raise SystemExit(main())
