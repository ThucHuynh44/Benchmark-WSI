import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from configs.experiment_loader import config_to_argv
from models import validate_model_configuration
from utils.main import parse_args


class MethodConfigTests(unittest.TestCase):
    def test_all_method_configs_parse(self):
        config_path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        expected = {
            "agem", "derpp", "er_ace", "ewc_on",
            "gdumb", "joint", "lwsr", "lwf", "micil", "qpmil_vl", "sgd",
        }
        raw = yaml.safe_load(config_path.read_text())
        self.assertEqual(set(raw["methods"]), expected)
        baseline_methods = expected - {"lwsr", "micil", "qpmil_vl"}
        supported = [
            *((method, backbone) for method in baseline_methods
              for backbone in ("generic_mil", "titan", "feather")),
            *((method, backbone) for method in ("lwsr", "micil")
              for backbone in ("titan", "feather")),
            ("qpmil_vl", "titan"),
        ]
        for method, backbone in supported:
            with self.subTest(method=method, backbone=backbone), patch.object(
                sys,
                "argv",
                [
                    "main.py", "--config", str(config_path), "--model", method,
                    "--backbone", backbone,
                ],
            ):
                args = parse_args()
                validate_model_configuration(args)
                self.assertEqual(args.model, method)
                self.assertEqual(args.backbone, backbone)
                self.assertEqual(args.dataset, "seq-wsi")
                self.assertEqual(args.batch_size, 1)
                self.assertEqual(args.optimizer, "adamw")
                self.assertEqual(args.adam_eps, 1.0e-8)
                self.assertEqual(args.optim_wd, 1.0e-4)
                self.assertTrue(args.early_stopping)
                self.assertGreater(args.early_stopping_patience, 0)
                self.assertGreaterEqual(args.early_stopping_min_epoch, 0)
                self.assertGreaterEqual(args.early_stopping_min_delta, 0.0)
                self.assertFalse(args.evaluate_fwt)
                self.assertEqual(args.feature_dim, 768)
                buffer_size = raw["methods"][method].get("buffer_size")
                buffer_tag = (
                    f"buffer{buffer_size}" if buffer_size is not None else "nobuffer"
                )
                self.assertEqual(
                    args.exp_desc, f"{method}_{backbone}_{buffer_tag}_10tasks"
                )
                expected_patch_budget = 400 if backbone == "titan" else 0
                self.assertEqual(args.backbone_max_patches, expected_patch_budget)

    def test_new_method_defaults(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        raw = yaml.safe_load(path.read_text())
        cases = {"lwsr": "titan", "micil": "feather", "qpmil_vl": "titan"}
        for method, backbone in cases.items():
            with self.subTest(method=method), patch.object(
                sys,
                "argv",
                [
                    "main.py", "--config", str(path), "--model", method,
                    "--backbone", backbone,
                ],
            ):
                args = parse_args()
            for name, value in raw["methods"][method].items():
                self.assertEqual(getattr(args, name), value, name)

    def test_new_methods_reject_unsupported_backbones_and_freezing(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        invalid = [
            ("lwsr", "generic_mil", []),
            ("micil", "generic_mil", []),
            ("qpmil_vl", "feather", []),
            ("lwsr", "titan", ["--backbone_freeze"]),
            ("micil", "feather", ["--backbone_freeze"]),
            ("lwsr", "titan", ["--feature_dim", "512"]),
            ("micil", "feather", ["--feature_dim", "512"]),
            ("qpmil_vl", "titan", ["--feature_dim", "512"]),
        ]
        for method, backbone, extra in invalid:
            with self.subTest(method=method, backbone=backbone, extra=extra), patch.object(
                sys,
                "argv",
                [
                    "main.py", "--config", str(path), "--model", method,
                    "--backbone", backbone, *extra,
                ],
            ):
                args = parse_args()
            with self.assertRaises(ValueError):
                validate_model_configuration(args)

    def test_cli_overrides_yaml(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        with patch.object(
            sys,
            "argv",
            [
                "main.py", "--config", str(path), "--model", "sgd",
                "--backbone", "titan", "--folds", "0", "--n_epochs", "3",
                "--backbone_max_patches", "123",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.folds, "0")
        self.assertEqual(args.n_epochs, 3)
        self.assertEqual(args.backbone, "titan")
        self.assertEqual(args.backbone_max_patches, 123)

    def test_early_stopping_can_be_disabled_from_cli(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        with patch.object(
            sys,
            "argv",
            [
                "main.py", "--config", str(path), "--model", "sgd",
                "--backbone", "titan", "--no-early_stopping",
                "--early_stopping_patience", "2",
            ],
        ):
            args = parse_args()
        self.assertFalse(args.early_stopping)
        self.assertEqual(args.early_stopping_patience, 2)

    def test_yaml_false_emits_boolean_optional_flags(self):
        self.assertEqual(
            config_to_argv({
                "early_stopping": False,
                "early_stopping_verbose": False,
            }),
            ["--no-early_stopping", "--no-early_stopping_verbose"],
        )

    def test_fwt_evaluation_can_be_enabled_explicitly(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        with patch.object(
            sys,
            "argv",
            [
                "main.py", "--config", str(path), "--model", "sgd",
                "--backbone", "titan", "--evaluate_fwt",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.evaluate_fwt)

    def test_method_specific_cli_overrides_yaml(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        with patch.object(
            sys,
            "argv",
            [
                "main.py", "--config", str(path), "--model", "micil",
                "--backbone", "titan", "--micil_replay",
                "--no-micil_weight_norm", "--buffer_size", "17",
                "--bags_per_update", "2",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.micil_replay)
        self.assertFalse(args.micil_weight_norm)
        self.assertEqual(args.buffer_size, 17)
        self.assertEqual(args.bags_per_update, 2)
        self.assertEqual(args.exp_desc, "micil_titan_buffer17_10tasks")

    def test_configs_do_not_contain_credentials(self):
        path = Path(__file__).parents[1] / "configs" / "methods.yaml"
        self.assertNotIn("hf_token", path.read_text())


if __name__ == "__main__":
    unittest.main()
