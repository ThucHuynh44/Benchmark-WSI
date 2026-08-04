import unittest
from types import SimpleNamespace

import torch

from utils.optim import build_optimizer


class SharedOptimizerTests(unittest.TestCase):
    def test_builds_adamw_with_common_hyperparameters(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        args = SimpleNamespace(
            optimizer="adamw",
            lr=1.0e-5,
            optim_wd=1.0e-4,
            adam_eps=1.0e-8,
        )
        optimizer = build_optimizer([parameter], args)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1.0e-4)
        self.assertEqual(optimizer.param_groups[0]["eps"], 1.0e-8)

    def test_learning_rate_override_keeps_adamw_and_weight_decay(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        args = SimpleNamespace(
            optimizer="adamw",
            lr=1.0e-5,
            optim_wd=2.0e-4,
            adam_eps=1.0e-7,
        )
        optimizer = build_optimizer([parameter], args, lr=3.0e-5)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 2.0e-4)
        self.assertEqual(optimizer.param_groups[0]["eps"], 1.0e-7)

    def test_rejects_non_adamw_optimizer(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        args = SimpleNamespace(optimizer="adam", lr=1.0e-5, optim_wd=0.0)
        with self.assertRaisesRegex(ValueError, "Only optimizer='adamw'"):
            build_optimizer([parameter], args)


if __name__ == "__main__":
    unittest.main()
