import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils.training import (
    EarlyStopping,
    _resource_statistics,
    early_stopping_from_args,
    parameter_statistics,
)


class TrainingResourceTests(unittest.TestCase):
    def test_parameter_statistics_counts_total_and_trainable(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        module[0].weight.requires_grad = False
        stats = parameter_statistics(module)
        self.assertEqual(stats["total_parameters"], 26)
        self.assertEqual(stats["trainable_parameters"], 14)

    def test_cpu_resource_statistics_marks_cuda_peaks_unavailable(self):
        model = SimpleNamespace(device=torch.device("cpu"))
        stats = _resource_statistics(
            model, {"total_parameters": 26, "trainable_parameters": 14}
        )
        self.assertFalse(stats["cuda_available"])
        self.assertIsNone(stats["peak_gpu_allocated_mib"])
        self.assertIsNone(stats["peak_gpu_reserved_mib"])
        self.assertEqual(stats["initial_total_parameters"], 26)
        self.assertEqual(stats["final_total_parameters"], 26)
        self.assertEqual(stats["parameter_growth"], 0)

    def test_resource_statistics_records_dynamic_parameter_growth(self):
        model = SimpleNamespace(device=torch.device("cpu"))
        stats = _resource_statistics(
            model,
            {"total_parameters": 42, "trainable_parameters": 18},
            initial_parameters={
                "total_parameters": 26,
                "trainable_parameters": 14,
            },
        )
        self.assertEqual(stats["total_parameters"], 42)
        self.assertEqual(stats["initial_total_parameters"], 26)
        self.assertEqual(stats["final_total_parameters"], 42)
        self.assertEqual(stats["parameter_growth"], 16)


class EarlyStoppingTests(unittest.TestCase):
    def _call(self, stopper, epoch, loss, checkpoint_path):
        with patch("utils.training.checkpoint_payload", return_value={}), patch(
            "utils.training.torch.save"
        ) as save:
            stopper(epoch, loss, object(), checkpoint_path, object(), 0)
        return save.called

    def test_min_delta_patience_and_min_epoch(self):
        stopper = EarlyStopping(
            patience=2, min_epoch=3, min_delta=0.1, verbose=False, enabled=True
        )
        path = Path(tempfile.gettempdir()) / "unused-early-stop.pt"
        self.assertTrue(self._call(stopper, 0, 1.0, path))
        self.assertFalse(self._call(stopper, 1, 0.95, path))
        self.assertFalse(stopper.early_stop)
        self.assertFalse(self._call(stopper, 2, 0.94, path))
        self.assertTrue(stopper.early_stop)

    def test_disabled_stopping_still_saves_best_checkpoint(self):
        args = SimpleNamespace(
            early_stopping=False,
            early_stopping_patience=1,
            early_stopping_min_epoch=1,
            early_stopping_min_delta=0.0,
            early_stopping_verbose=False,
        )
        stopper = early_stopping_from_args(args)
        path = Path(tempfile.gettempdir()) / "unused-early-stop.pt"
        self.assertTrue(self._call(stopper, 0, 1.0, path))
        self.assertFalse(self._call(stopper, 1, 2.0, path))
        self.assertFalse(stopper.early_stop)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "patience"):
            EarlyStopping(patience=0)
        with self.assertRaisesRegex(ValueError, "min_epoch"):
            EarlyStopping(min_epoch=-1)
        with self.assertRaisesRegex(ValueError, "min_delta"):
            EarlyStopping(min_delta=-0.1)


if __name__ == "__main__":
    unittest.main()
