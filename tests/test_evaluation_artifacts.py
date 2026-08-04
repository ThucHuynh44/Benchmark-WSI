import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.artifacts import (
    EVAL_MATRIX_FIELDS,
    PER_FOLD_FIELDS,
    append_evaluation,
    evaluation_metrics,
    finalize_artifacts,
    initialize_artifacts,
    write_manifest,
)
from scripts.summarize_results import aggregate


class CanonicalMetricTests(unittest.TestCase):
    def test_binary_imbalanced_metrics_match_benchmark_definitions(self):
        targets = np.asarray([0, 0, 0, 1])
        predictions = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.1, 0.9],
        ])
        metrics = evaluation_metrics(targets, predictions, probabilities, loss=0.25)
        self.assertAlmostEqual(metrics["acc"], 0.75)
        self.assertAlmostEqual(metrics["bacc"], 5 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.8 * 0.75 + (2 / 3) * 0.25)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["kappa"], 0.5)
        self.assertEqual(metrics["n"], 4)

    def test_multiclass_metrics_are_finite(self):
        targets = np.asarray([0, 1, 2, 0, 1, 2])
        probabilities = np.eye(3)[targets] * 0.8 + 0.2 / 3
        predictions = probabilities.argmax(axis=1)
        metrics = evaluation_metrics(targets, predictions, probabilities, loss=0.1)
        for key in ("acc", "bacc", "macro_f1", "weighted_f1", "auroc", "kappa"):
            self.assertAlmostEqual(metrics[key], 1.0)


class CanonicalArtifactTests(unittest.TestCase):
    def test_artifact_schema_and_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = initialize_artifacts(directory, total_classes=2)
            write_manifest(directory, {"backbone": "titan", "method": "sgd"})
            for after_task, acc in ((0, 0.8), (1, 0.6)):
                row = {field: "" for field in EVAL_MATRIX_FIELDS}
                row.update({
                    "method": "sgd",
                    "fold": 0,
                    "mode": "class-il-seen",
                    "after_task": after_task,
                    "eval_task": 0,
                    "task_name": "camelyon17",
                    "acc": acc,
                    "bacc": acc,
                    "macro_f1": acc,
                    "weighted_f1": acc,
                    "auroc": acc,
                    "kappa": acc,
                    "loss": 1 - acc,
                    "n": 4,
                    "eval_time_sec": 0.2,
                    "confusion_matrix_path": "matrix.csv",
                })
                append_evaluation(artifacts, row, [])
            finalize_artifacts(
                artifacts,
                "sgd",
                {0: 3.0},
                {
                    0: {
                        "total_parameters": 100,
                        "trainable_parameters": 75,
                        "peak_gpu_allocated_mib": 12.5,
                        "peak_gpu_reserved_mib": 16.0,
                    }
                },
            )

            root = Path(directory)
            self.assertTrue((root / "run_manifest.json").is_file())
            self.assertTrue((root / "eval_matrix.csv").is_file())
            self.assertTrue((root / "per_slide_predictions.csv").is_file())
            self.assertTrue((root / "per_fold_summary.csv").is_file())
            self.assertTrue((root / "per_task_summary.csv").is_file())
            with (root / "per_fold_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), PER_FOLD_FIELDS)
            self.assertEqual(rows[0]["final_acc"], "0.6")
            self.assertEqual(rows[0]["training_time"], "3.0")
            self.assertEqual(rows[0]["total_parameters"], "100")
            self.assertEqual(rows[0]["trainable_parameters"], "75")
            self.assertEqual(rows[0]["peak_gpu_allocated_mib"], "12.5")
            self.assertEqual(rows[0]["peak_gpu_reserved_mib"], "16.0")
            self.assertEqual(json.loads((root / "run_manifest.json").read_text())["method"], "sgd")

            aggregate_dir = root / "aggregate"
            counts = aggregate([directory], str(aggregate_dir), strict=True)
            self.assertEqual(counts["runs"], 1)
            self.assertEqual(counts["fold_rows"], 3)  # fold 0, mean, std
            self.assertEqual(counts["task_rows"], 1)
            self.assertTrue((aggregate_dir / "all_methods_per_fold.csv").is_file())


if __name__ == "__main__":
    unittest.main()
