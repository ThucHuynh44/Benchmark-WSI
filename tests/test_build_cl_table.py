import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_cl_table import build_table


class ContinualLearningTableTests(unittest.TestCase):
    @staticmethod
    def _write_matrix(path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("fold", "after_task", "eval_task", "acc", "bacc", "n"),
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_recomputes_documented_metrics_from_eval_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "derpp_titan_buffer30_2tasks"
            class_dir = experiment / "evaluation/class_il"
            task_dir = experiment / "evaluation/task_il"
            class_dir.mkdir(parents=True)
            task_dir.mkdir(parents=True)
            manifest = {
                "method": "derpp",
                "backbone": "titan",
                "num_tasks": 2,
                "folds": [0],
            }
            (class_dir / "run_manifest.json").write_text(json.dumps(manifest))
            (task_dir / "run_manifest.json").write_text(json.dumps(manifest))

            class_rows = [
                {"fold": 0, "after_task": 0, "eval_task": 0, "acc": 0.8, "bacc": 0.7, "n": 2},
                {"fold": 0, "after_task": 1, "eval_task": 0, "acc": 0.6, "bacc": 0.5, "n": 2},
                {"fold": 0, "after_task": 1, "eval_task": 1, "acc": 1.0, "bacc": 0.9, "n": 6},
            ]
            task_rows = [
                {"fold": 0, "after_task": 0, "eval_task": 0, "acc": 0.9, "bacc": 0.85, "n": 2},
                {"fold": 0, "after_task": 1, "eval_task": 0, "acc": 0.9, "bacc": 0.8, "n": 2},
                {"fold": 0, "after_task": 1, "eval_task": 1, "acc": 1.0, "bacc": 0.95, "n": 6},
            ]
            self._write_matrix(class_dir / "eval_matrix.csv", class_rows)
            self._write_matrix(task_dir / "eval_matrix.csv", task_rows)

            output = root / "table.csv"
            rows = build_table(
                str(root),
                str(output),
                expected_tasks=2,
                check_predictions=False,
                strict=True,
            )
            self.assertEqual(len(rows), 1)
            result = rows[0]
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["buffer_size"], "30")
            self.assertEqual(result["mACC_mean"], "0.8500")
            self.assertEqual(result["bACC_mean"], "0.7000")
            self.assertEqual(result["masked_bACC_mean"], "0.8750")
            self.assertEqual(result["BWT_mean"], "-0.2000")
            self.assertEqual(result["FGT_mean"], "0.2000")
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
