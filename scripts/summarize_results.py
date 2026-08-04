"""Aggregate canonical evaluation artifacts from multiple experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.artifacts import PER_FOLD_FIELDS, PER_TASK_FIELDS


RUN_INDEX_FIELDS = [
    "artifact_schema_version", "backbone", "method", "model_name", "mode",
    "run_dir", "num_folds", "num_tasks", "task_order",
    "num_classes_per_task", "k", "seed", "total_parameters",
    "trainable_parameters", "early_stopping", "early_stopping_patience",
    "early_stopping_min_epoch", "early_stopping_min_delta",
    "early_stopping_verbose", "repo_commit", "command",
]


def _manifest_paths(inputs: Sequence[str]) -> Iterable[Path]:
    seen = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        candidates = [path] if path.name == "run_manifest.json" else path.rglob("run_manifest.json")
        for candidate in candidates:
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                yield candidate


def _read_csv(path: Path, required: Sequence[str]) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(required).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def _json_cell(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_index_row(manifest: dict, run_dir: Path) -> dict:
    return {
        field: str(run_dir) if field == "run_dir" else _json_cell(manifest.get(field))
        for field in RUN_INDEX_FIELDS
    }


def _add_identity(rows: Sequence[dict], manifest: dict, run_dir: Path) -> List[dict]:
    output = []
    for row in rows:
        normalized = dict(row)
        normalized["method"] = manifest.get("method", row.get("method", ""))
        output.append({
            "backbone": manifest.get("backbone", ""),
            "run_dir": str(run_dir),
            **normalized,
        })
    return output


def aggregate(inputs: Sequence[str], output_dir: str, strict: bool = False) -> Dict[str, int]:
    run_rows, fold_rows, task_rows, skipped_rows = [], [], [], []
    for manifest_path in sorted(_manifest_paths(inputs)):
        run_dir = manifest_path.parent
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        try:
            current_fold_rows = _read_csv(run_dir / "per_fold_summary.csv", PER_FOLD_FIELDS)
            current_task_rows = _read_csv(run_dir / "per_task_summary.csv", PER_TASK_FIELDS)
        except (FileNotFoundError, ValueError) as error:
            if strict:
                raise
            skipped_rows.append({"run_dir": str(run_dir), "reason": str(error)})
            continue
        run_rows.append(_run_index_row(manifest, run_dir))
        fold_rows.extend(_add_identity(current_fold_rows, manifest, run_dir))
        task_rows.extend(_add_identity(current_task_rows, manifest, run_dir))

    output = Path(output_dir).expanduser().resolve()
    _write_csv(output / "run_index.csv", run_rows, RUN_INDEX_FIELDS)
    _write_csv(
        output / "all_methods_per_fold.csv",
        fold_rows,
        ["backbone", "run_dir", *PER_FOLD_FIELDS],
    )
    _write_csv(
        output / "all_methods_per_task.csv",
        task_rows,
        ["backbone", "run_dir", *PER_TASK_FIELDS],
    )
    _write_csv(output / "skipped_runs.csv", skipped_rows, ["run_dir", "reason"])
    return {
        "runs": len(run_rows),
        "fold_rows": len(fold_rows),
        "task_rows": len(task_rows),
        "skipped": len(skipped_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    counts = aggregate(args.input, args.output, strict=args.strict)
    print(
        f"Aggregated {counts['runs']} runs, {counts['fold_rows']} fold rows, "
        f"{counts['task_rows']} task rows; skipped {counts['skipped']} runs."
    )
    print(f"Outputs saved to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
