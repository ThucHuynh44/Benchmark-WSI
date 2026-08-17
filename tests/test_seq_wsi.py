import csv
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import yaml

from configs.loader import load_dataset_config
from datasets.seq_wsi import (
    PreflightError,
    Sequential_Generic_MIL_Dataset,
    normalize_slide_id,
    parse_folds,
)


def _args(config_path, feature_dim=8):
    return SimpleNamespace(
        batch_size=1,
        dataset_config=str(config_path),
        reverse_task_order=False,
        feature_dim=feature_dim,
        num_workers=0,
        backbone="generic_mil",
        backbone_hidden_dim=8,
        backbone_dropout=0.0,
        backbone_kwargs=None,
        backbone_max_patches=0,
        backbone_freeze=False,
        backbone_model_id=None,
        backbone_revision=None,
        patch_size_level0_fallback=1024,
    )


def _write_h5(path, dim=8, patch_size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones((3, dim), dtype=np.float32))
        coords = handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int64))
        if patch_size is not None:
            coords.attrs["patch_size_level0"] = patch_size


class DatasetConfigTests(unittest.TestCase):
    def test_fold_parser(self):
        self.assertEqual(parse_folds("all"), list(range(10)))
        self.assertEqual(parse_folds("0,2,4-6"), [0, 2, 4, 5, 6])
        with self.assertRaises(ValueError):
            parse_folds("7-3")

    def test_id_normalization(self):
        self.assertEqual(normalize_slide_id("sample.svs"), "sample")
        self.assertEqual(normalize_slide_id("123.0"), "123")
        self.assertEqual(normalize_slide_id("case.0"), "case.0")

    def test_placeholder_and_reverse_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "datasets.yaml"
            path.write_text(yaml.safe_dump({
                "data_root": directory,
                "task_order": ["camelyon17", "esca"],
                "reverse_task_order": True,
                "annotations": {"camelyon17": "{data_root}/ann.csv"},
                "features": {"camelyon17": "{data_root}/features"},
                "split_dirs": {"camelyon17": "{data_root}/splits"},
            }))
            config = load_dataset_config(str(path))
            self.assertEqual(config["task_order"], ["esca", "camelyon17"])
            self.assertEqual(config["annotations"]["camelyon17"], f"{directory}/ann.csv")

    def test_ten_task_offsets_total_27(self):
        config = Path(__file__).parents[1] / "configs" / "datasets.yaml.example"
        dataset = Sequential_Generic_MIL_Dataset(_args(config, feature_dim=768))
        self.assertEqual(dataset.task_num_classes, [4, 2, 3, 2, 2, 2, 2, 3, 2, 5])
        self.assertEqual(dataset.class_offsets, [0, 4, 6, 9, 11, 13, 15, 17, 20, 22])
        self.assertEqual(dataset.total_num_classes, 27)
        reverse_args = _args(config, feature_dim=768)
        reverse_args.reverse_task_order = True
        reversed_dataset = Sequential_Generic_MIL_Dataset(reverse_args)
        self.assertEqual(reversed_dataset.task_num_classes, list(reversed(dataset.task_num_classes)))
        self.assertEqual(reversed_dataset.total_num_classes, 27)


class LoaderSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.features = self.root / "features"
        self.splits = self.root / "splits"
        self.splits.mkdir()

        annotation_rows = []
        cam_split = {}
        for split_name in ("train", "val", "test"):
            ids = []
            for local_label, label_name in enumerate(("negative", "itc", "micro", "macro")):
                slide_id = f"cam_{split_name}_{local_label}"
                ids.append(slide_id + (".svs" if local_label == 0 else ""))
                annotation_rows.append({"slide_id": slide_id, "case_id": slide_id, "label": label_name})
                _write_h5(self.features / "cam" / f"{slide_id}.h5")
            cam_split[split_name] = ids
        pd.DataFrame(annotation_rows).to_csv(self.root / "cam.csv", index=False)
        pd.DataFrame(cam_split).to_csv(self.splits / "cam_splits_0.csv", index=False)

        embedded = {}
        for split_name in ("train", "val", "test"):
            embedded[split_name] = [f"esca_{split_name}_0", f"esca_{split_name}_1"]
            embedded[f"{split_name}_label"] = [0, 1]
            for slide_id in embedded[split_name]:
                _write_h5(self.features / "esca" / f"{slide_id}.h5")
        pd.DataFrame(embedded).to_csv(self.splits / "esca_splits_0.csv", index=False)

        config = {
            "data_root": str(self.root),
            "task_order": ["camelyon17", "esca"],
            "annotations": {"camelyon17": str(self.root / "cam.csv")},
            "features": {
                "camelyon17": str(self.features / "cam"),
                "esca": str(self.features / "esca"),
            },
            "split_dirs": {
                "camelyon17": str(self.splits / "cam"),
                "esca": str(self.splits / "esca"),
            },
        }
        # Loader expects split directories, so use task-specific directories.
        (self.splits / "cam").mkdir()
        (self.splits / "esca").mkdir()
        (self.splits / "cam_splits_0.csv").replace(self.splits / "cam" / "splits_0.csv")
        (self.splits / "esca_splits_0.csv").replace(self.splits / "esca" / "splits_0.csv")
        self.config_path = self.root / "datasets.yaml"
        self.config_path.write_text(yaml.safe_dump(config))

    def tearDown(self):
        self.temp.cleanup()

    def test_annotation_and_embedded_schemas_global_labels(self):
        dataset = Sequential_Generic_MIL_Dataset(_args(self.config_path))
        dataset.preflight(0)
        train0, _, _ = dataset.get_data_loaders(0, 0)
        train1, _, _ = dataset.get_data_loaders(0, 1)
        self.assertEqual(sorted(train0.dataset.slide_data.label.tolist()), [0, 1, 2, 3])
        self.assertEqual(sorted(train1.dataset.slide_data.label.tolist()), [4, 5])

    def test_patch_size_attr_and_fallback(self):
        attributed = self.features / "cam" / "cam_train_0.h5"
        with h5py.File(attributed, "r+") as handle:
            handle["coords"].attrs["patch_size_level0"] = 512
        dataset = Sequential_Generic_MIL_Dataset(_args(self.config_path))
        train0, _, _ = dataset.get_data_loaders(0, 0)
        values = {sample[2] for sample in train0.dataset}
        self.assertIn(512, values)
        self.assertIn(1024, values)

    def test_strict_preflight_reports_missing_feature(self):
        (self.features / "esca" / "esca_test_1.h5").unlink()
        dataset = Sequential_Generic_MIL_Dataset(_args(self.config_path))
        with self.assertRaises(PreflightError) as context:
            dataset.preflight(0)
        self.assertIn("esca/test/esca_test_1", str(context.exception))

    def test_strict_preflight_reports_overlap(self):
        path = self.splits / "esca" / "splits_0.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "val"] = frame.loc[0, "train"]
        frame.to_csv(path, index=False)
        with self.assertRaises(PreflightError) as context:
            Sequential_Generic_MIL_Dataset(_args(self.config_path)).preflight(0)
        self.assertIn("train/val overlap", str(context.exception))

    def test_strict_preflight_reports_bad_h5_shape(self):
        path = self.features / "esca" / "esca_test_1.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("features", data=np.ones((3, 7), dtype=np.float32))
            handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int64))
        with self.assertRaises(PreflightError) as context:
            Sequential_Generic_MIL_Dataset(_args(self.config_path)).preflight(0)
        self.assertIn("features must have shape [N,8]", str(context.exception))

    def test_strict_preflight_reports_invalid_label(self):
        frame = pd.read_csv(self.root / "cam.csv")
        frame.loc[0, "label"] = "unknown"
        frame.to_csv(self.root / "cam.csv", index=False)
        with self.assertRaises(PreflightError) as context:
            Sequential_Generic_MIL_Dataset(_args(self.config_path)).preflight(0)
        self.assertIn("invalid annotation label", str(context.exception))

    def test_two_task_training_checkpoint_smoke(self):
        from models.sgd import Sgd
        from utils.training import train

        args = _args(self.config_path)
        args.dataset = "seq-wsi"
        args.model = "sgd"
        args.exp_desc = "smoke"
        args.lr = 1e-3
        args.n_epochs = 1
        args.csv_log = False
        args.tensorboard = False
        args.validation = False
        args.seed = 0
        dataset = Sequential_Generic_MIL_Dataset(args)
        dataset.preflight(0)
        args.n_tasks = dataset.N_TASKS
        args.task_order = dataset.task_order
        args.task_num_classes = dataset.task_num_classes
        args.class_offsets = dataset.class_offsets
        args.num_classes = dataset.total_num_classes
        args.n_classes_per_task = tuple(dataset.task_num_classes)
        model = Sgd(dataset.get_backbone(), dataset.get_loss(), args, None)

        previous = os.getcwd()
        try:
            os.chdir(self.root)
            train(model, dataset, args, 0)
        finally:
            os.chdir(previous)
        checkpoint = self.root / "checkpoints/smoke/fold_0/task1_checkpoint.pt"
        self.assertTrue(checkpoint.is_file())
        components = self.root / "results/smoke/evaluation/train_components.csv"
        self.assertTrue(components.is_file())
        with components.open(newline="", encoding="utf-8") as handle:
            component_rows = list(csv.DictReader(handle))
        self.assertEqual(len(component_rows), 2)
        self.assertTrue(all(row["loss"] for row in component_rows))
        for mode in ("class_il", "task_il"):
            artifact_dir = self.root / "results/smoke/evaluation" / mode
            self.assertTrue((artifact_dir / "run_manifest.json").is_file())
            self.assertTrue((artifact_dir / "eval_matrix.csv").is_file())
            self.assertTrue((artifact_dir / "per_slide_predictions.csv").is_file())
            self.assertTrue((artifact_dir / "per_fold_summary.csv").is_file())
            self.assertTrue((artifact_dir / "per_task_summary.csv").is_file())
            self.assertTrue((artifact_dir / "confusion_matrices").is_dir())

        from utils.training import load_checkpoint

        for field, value in (
            ("backbone_revision", "different-revision"),
            ("backbone_freeze", True),
            ("backbone_max_patches", 3),
        ):
            incompatible_args = _args(self.config_path)
            setattr(incompatible_args, field, value)
            incompatible = Sequential_Generic_MIL_Dataset(incompatible_args)
            with self.subTest(checkpoint_field=field), self.assertRaises(ValueError):
                load_checkpoint(model, checkpoint, incompatible, 0)


if __name__ == "__main__":
    unittest.main()
