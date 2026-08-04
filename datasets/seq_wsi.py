"""Configuration-driven continual WSI dataset stream."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from backbone.generic_mil import build_mil_backbone
from configs.loader import load_dataset_config
from datasets.utils.continual_dataset import ContinualDataset


TASK_SPECS = {
    "camelyon17": {
        "labels": ["negative", "itc", "micro", "macro"],
        "schema": "annotation",
        "label_col": "label",
    },
    "brca": {
        "labels": ["IDC", "ILC"],
        "schema": "annotation",
        "label_col": "oncotree_code",
        "ignore": ["MDLC", "PD", "ACBC", "IMMC", "BRCNOS", "BRCA", "SPC", "MBC", "MPT"],
    },
    "rcc": {
        "labels": ["CCRCC", "PRCC", "CHRCC"],
        "schema": "annotation",
        "label_col": "oncotree_code",
    },
    "nsclc": {
        "labels": ["LUAD", "LUSC"],
        "schema": "annotation",
        "label_col": "oncotree_code",
    },
    "esca": {"labels": [0, 1], "schema": "embedded"},
    "tgct": {"labels": [0, 1], "schema": "embedded"},
    "cesc": {"labels": [0, 1], "schema": "embedded"},
    "bracs": {
        "labels": ["Group_BT", "Group_AT", "Group_MT"],
        "schema": "annotation",
        "label_col": "label",
    },
    "herohe": {
        "labels": ["Negative", "Positive"],
        "schema": "annotation",
        "label_col": "label",
    },
    "ubc_ocean": {
        "labels": ["HGSC", "EC", "CC", "LGSC", "MC"],
        "schema": "annotation",
        "label_col": "label",
    },
}


class PreflightError(RuntimeError):
    """Raised when a fold cannot be trained without modifying source data."""

    def __init__(self, fold: int, errors: Sequence[str]):
        self.fold = int(fold)
        self.errors = list(errors)
        details = "\n".join(f"  - {item}" for item in self.errors)
        super().__init__(f"Strict preflight failed for fold {fold}:\n{details}")


def normalize_slide_id(value) -> str:
    """Normalize IDs used by annotations, split CSVs and feature filenames."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    lower = value.lower()
    if lower.endswith(".svs") or lower.endswith(".h5"):
        value = value[:-4]
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    return value


def parse_folds(spec: str, available: int = 10) -> List[int]:
    """Parse ``all``, comma lists and inclusive ranges into unique fold IDs."""
    if str(spec).strip().lower() == "all":
        return list(range(available))
    folds: List[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending fold range: {token}")
            folds.extend(range(start, end + 1))
        else:
            folds.append(int(token))
    if not folds:
        raise ValueError("At least one fold must be selected")
    invalid = sorted({fold for fold in folds if fold < 0 or fold >= available})
    if invalid:
        raise ValueError(f"Fold IDs must be in [0, {available - 1}], got {invalid}")
    return list(dict.fromkeys(folds))


class MILBatch(NamedTuple):
    features: torch.Tensor
    coords: torch.Tensor
    patch_size_level0: torch.Tensor
    labels: torch.Tensor


def collate_MIL(batch):
    if len(batch) != 1:
        raise ValueError("Variable-length MIL bags require --batch_size 1")
    item = batch[0]
    return MILBatch(
        features=item[0],
        coords=item[1],
        patch_size_level0=torch.as_tensor(item[2], dtype=torch.long),
        labels=torch.as_tensor([item[3]], dtype=torch.long),
    )


def resolve_feature_path(feature_root: str, slide_id: str) -> Path:
    root = Path(feature_root)
    stem = normalize_slide_id(slide_id)
    candidates = (
        root / f"{stem}.h5",
        root / "h5_files" / f"{stem}.h5",
        root / "features_conch_v15" / f"{stem}.h5",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing HDF5 for slide '{stem}'; checked: {', '.join(map(str, candidates))}")


class WSIBagDataset(Dataset):
    """A split represented by normalized slide IDs and global labels."""

    def __init__(
        self,
        records: pd.DataFrame,
        feature_root: str,
        feature_dim: int,
        patch_size_level0_fallback: int = 1024,
    ):
        self.slide_data = records.reset_index(drop=True).copy()
        self.feature_root = feature_root
        self.feature_dim = int(feature_dim)
        self.patch_size_level0_fallback = _validate_patch_size(
            patch_size_level0_fallback, "patch_size_level0_fallback"
        )

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, index):
        row = self.slide_data.iloc[index]
        path = resolve_feature_path(self.feature_root, row.slide_id)
        with h5py.File(path, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                missing = [key for key in ("features", "coords") if key not in handle]
                raise KeyError(f"{path}: missing HDF5 datasets {missing}")
            features = torch.from_numpy(handle["features"][:]).float()
            coords = torch.from_numpy(handle["coords"][:]).long()
            patch_size_level0 = _validate_patch_size(
                handle["coords"].attrs.get(
                    "patch_size_level0", self.patch_size_level0_fallback
                ),
                f"{path}: coords.attrs['patch_size_level0']",
            )
        _validate_arrays(path, features.shape, coords.shape, self.feature_dim)
        return features, coords, patch_size_level0, int(row.label)


def _validate_patch_size(value, source: str) -> int:
    try:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError
            value = value.item()
        patch_size = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    if patch_size <= 0:
        raise ValueError(f"{source} must be a positive integer, got {patch_size}")
    return patch_size


def _validate_arrays(path, feature_shape, coord_shape, feature_dim: int) -> None:
    if len(feature_shape) != 2 or feature_shape[1] != feature_dim:
        raise ValueError(
            f"{path}: features must have shape [N,{feature_dim}], got {tuple(feature_shape)}"
        )
    if len(coord_shape) != 2 or coord_shape[1] != 2:
        raise ValueError(f"{path}: coords must have shape [N,2], got {tuple(coord_shape)}")
    if feature_shape[0] != coord_shape[0]:
        raise ValueError(
            f"{path}: patch count mismatch features={feature_shape[0]} coords={coord_shape[0]}"
        )
    if feature_shape[0] == 0:
        raise ValueError(f"{path}: empty WSI bag")


class Sequential_Generic_MIL_Dataset(ContinualDataset):
    NAME = "seq-wsi"
    SETTING = "class-il"
    TRANSFORM = None

    def __init__(self, args):
        super().__init__(args)
        if getattr(args, "batch_size", 1) != 1:
            raise ValueError("seq-wsi supports batch_size=1 only")
        backbone_name = str(getattr(args, "backbone", "generic_mil")).lower()
        if backbone_name in {"titan", "feather"}:
            if int(getattr(args, "feature_dim", 768)) != 768:
                raise ValueError(f"{backbone_name} requires --feature_dim 768")
            from backbone.pretrained_mil import (
                FEATHER_MODEL_ID,
                FEATHER_REVISION,
                TITAN_MODEL_ID,
                TITAN_REVISION,
            )
            defaults = {
                "titan": (TITAN_MODEL_ID, TITAN_REVISION, 400),
                "feather": (FEATHER_MODEL_ID, FEATHER_REVISION, 0),
            }
            model_id, revision, max_patches = defaults[backbone_name]
            if getattr(args, "backbone_model_id", None) is None:
                args.backbone_model_id = model_id
            if getattr(args, "backbone_revision", None) is None:
                args.backbone_revision = revision
            if getattr(args, "backbone_max_patches", None) is None:
                args.backbone_max_patches = max_patches
        elif getattr(args, "backbone_max_patches", None) is None:
            args.backbone_max_patches = 0
        _validate_patch_size(
            getattr(args, "patch_size_level0_fallback", 1024),
            "patch_size_level0_fallback",
        )
        self.config = load_dataset_config(getattr(args, "dataset_config", None))
        self.task_order = list(self.config["task_order"])
        if getattr(args, "reverse_task_order", False):
            self.task_order.reverse()
        unknown = [task for task in self.task_order if task not in TASK_SPECS]
        if unknown:
            raise ValueError(f"Unknown WSI tasks in task_order: {unknown}")
        if len(set(self.task_order)) != len(self.task_order):
            raise ValueError(f"task_order contains duplicates: {self.task_order}")

        self.task_num_classes = [len(TASK_SPECS[name]["labels"]) for name in self.task_order]
        self.class_offsets = np.cumsum([0] + self.task_num_classes[:-1]).astype(int).tolist()
        self.total_num_classes = int(sum(self.task_num_classes))
        self.N_TASKS = len(self.task_order)
        # Compatibility only; callers must use task_num_classes/task_slice.
        self.N_CLASSES_PER_TASK = tuple(self.task_num_classes)
        self.current_task = 0
        self.val_loader = None
        self._split_cache: Dict[Tuple[int, int], Tuple[WSIBagDataset, WSIBagDataset, WSIBagDataset]] = {}

    def task_slice(self, task_id: int) -> slice:
        start = self.class_offsets[int(task_id)]
        return slice(start, start + self.task_num_classes[int(task_id)])

    def seen_class_count(self, task_id: int) -> int:
        task_slice = self.task_slice(task_id)
        return int(task_slice.stop)

    def metadata(self, fold: int) -> dict:
        return {
            "fold": int(fold),
            "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "total_num_classes": self.total_num_classes,
            "dataset_config": self.config["config_path"],
            "backbone": getattr(self.args, "backbone", "generic_mil"),
            "feature_dim": int(getattr(self.args, "feature_dim", 768)),
            "optimizer_config": {
                "name": str(getattr(self.args, "optimizer", "adamw")).lower(),
                "lr": float(getattr(self.args, "lr", 1.0e-5)),
                "weight_decay": float(getattr(self.args, "optim_wd", 0.0)),
                "eps": float(getattr(self.args, "adam_eps", 1.0e-8)),
            },
            "backbone_config": {
                "name": getattr(self.args, "backbone", "generic_mil"),
                "model_id": getattr(self.args, "backbone_model_id", None),
                "revision": getattr(self.args, "backbone_revision", None),
                "pretrained": getattr(self.args, "backbone", "generic_mil") in {"titan", "feather"},
                "freeze": bool(getattr(self.args, "backbone_freeze", False)),
                "max_patches": int(getattr(self.args, "backbone_max_patches", 0) or 0),
                "patch_size_level0_fallback": int(
                    getattr(self.args, "patch_size_level0_fallback", 1024)
                ),
                "feature_dim": int(getattr(self.args, "feature_dim", 768)),
                "hidden_dim": int(getattr(self.args, "backbone_hidden_dim", 384)),
                "dropout": float(getattr(self.args, "backbone_dropout", 0.0)),
                "kwargs": getattr(self.args, "backbone_kwargs", None),
                "num_classes": self.total_num_classes,
            },
        }

    def _paths_for_task(self, task_name: str, fold: int) -> Tuple[str, str, str]:
        split_dir = self.config["split_dirs"].get(task_name, "")
        feature_root = self.config["features"].get(task_name, "")
        annotation = self.config["annotations"].get(task_name, "")
        split_path = str(Path(split_dir) / f"splits_{fold}.csv")
        return annotation, feature_root, split_path

    def _annotation_labels(self, task_name: str, annotation_path: str) -> Dict[str, int]:
        spec = TASK_SPECS[task_name]
        if not annotation_path or not Path(annotation_path).is_file():
            raise FileNotFoundError(f"{task_name}: missing annotation CSV: {annotation_path or '<unset>'}")
        frame = pd.read_csv(annotation_path, dtype=str)
        required = {"slide_id", spec["label_col"]}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{task_name}: annotation missing columns {sorted(missing)}")
        label_map = {str(label): index for index, label in enumerate(spec["labels"])}
        ignored = {str(label) for label in spec.get("ignore", [])}
        labels: Dict[str, int] = {}
        for _, row in frame.iterrows():
            slide_id = normalize_slide_id(row["slide_id"])
            raw_label = str(row[spec["label_col"]]).strip()
            if raw_label in ignored:
                continue
            if raw_label not in label_map:
                raise ValueError(f"{task_name}: invalid annotation label {raw_label!r} for slide {slide_id}")
            local_label = label_map[raw_label]
            if slide_id in labels and labels[slide_id] != local_label:
                raise ValueError(f"{task_name}: conflicting labels for slide {slide_id}")
            labels[slide_id] = local_label
        return labels

    @staticmethod
    def _split_ids(frame: pd.DataFrame, column: str) -> List[str]:
        if column not in frame:
            raise ValueError(f"split CSV missing column {column!r}")
        return [normalize_slide_id(value) for value in frame[column].dropna() if normalize_slide_id(value)]

    def _read_records(self, task_id: int, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        task_name = self.task_order[task_id]
        spec = TASK_SPECS[task_name]
        annotation_path, _, split_path = self._paths_for_task(task_name, fold)
        if not Path(split_path).is_file():
            raise FileNotFoundError(f"{task_name}: missing split CSV: {split_path}")
        frame = pd.read_csv(split_path, dtype=str)
        offset = self.class_offsets[task_id]
        result = []
        if spec["schema"] == "annotation":
            annotation = self._annotation_labels(task_name, annotation_path)
            for split_name in ("train", "val", "test"):
                ids = self._split_ids(frame, split_name)
                missing = [slide_id for slide_id in ids if slide_id not in annotation]
                if missing:
                    examples = ", ".join(missing[:5])
                    raise ValueError(
                        f"{task_name}: {len(missing)} {split_name} slide(s) have no valid annotation "
                        f"(examples: {examples})"
                    )
                result.append(pd.DataFrame({
                    "slide_id": ids,
                    "label": [annotation[slide_id] + offset for slide_id in ids],
                }))
        else:
            for split_name in ("train", "val", "test"):
                ids = self._split_ids(frame, split_name)
                label_col = f"{split_name}_label"
                if label_col not in frame:
                    raise ValueError(f"{task_name}: split CSV missing column {label_col!r}")
                raw_labels = frame.loc[frame[split_name].notna(), label_col].tolist()
                if len(raw_labels) != len(ids):
                    raise ValueError(f"{task_name}: {split_name} IDs and labels have different lengths")
                local_labels = []
                for slide_id, value in zip(ids, raw_labels):
                    try:
                        label = int(float(str(value)))
                    except (TypeError, ValueError):
                        raise ValueError(f"{task_name}: invalid label {value!r} for slide {slide_id}")
                    if label < 0 or label >= len(spec["labels"]):
                        raise ValueError(f"{task_name}: out-of-range label {label} for slide {slide_id}")
                    local_labels.append(label + offset)
                result.append(pd.DataFrame({"slide_id": ids, "label": local_labels}))
        return tuple(result)

    def _datasets_for_task(self, task_id: int, fold: int):
        key = (int(fold), int(task_id))
        if key not in self._split_cache:
            records = self._read_records(task_id, fold)
            task_name = self.task_order[task_id]
            feature_root = self.config["features"].get(task_name, "")
            feature_dim = int(getattr(self.args, "feature_dim", 768))
            fallback = int(getattr(self.args, "patch_size_level0_fallback", 1024))
            self._split_cache[key] = tuple(
                WSIBagDataset(record, feature_root, feature_dim, fallback) for record in records
            )
        return self._split_cache[key]

    def preflight(self, fold: int) -> None:
        errors: List[str] = []
        feature_dim = int(getattr(self.args, "feature_dim", 768))
        try:
            fallback = _validate_patch_size(
                getattr(self.args, "patch_size_level0_fallback", 1024),
                "patch_size_level0_fallback",
            )
        except ValueError as error:
            raise PreflightError(fold, [str(error)])
        for task_id, task_name in enumerate(self.task_order):
            try:
                records = self._read_records(task_id, fold)
                _, feature_root, _ = self._paths_for_task(task_name, fold)
                split_sets = [set(frame.slide_id) for frame in records]
                for left, right, names in ((0, 1, "train/val"), (0, 2, "train/test"), (1, 2, "val/test")):
                    overlap = split_sets[left].intersection(split_sets[right])
                    if overlap:
                        errors.append(
                            f"{task_name}: {names} overlap ({len(overlap)} slides; example {sorted(overlap)[0]})"
                        )
                expected = set(range(self.task_slice(task_id).start, self.task_slice(task_id).stop))
                for split_name, record in zip(("train", "val", "test"), records):
                    actual = set(map(int, record.label.tolist()))
                    missing_classes = sorted(expected.difference(actual))
                    if missing_classes:
                        errors.append(f"{task_name}: {split_name} missing global classes {missing_classes}")
                    duplicates = record.slide_id[record.slide_id.duplicated()].unique().tolist()
                    if duplicates:
                        errors.append(f"{task_name}: duplicate {split_name} slide IDs {duplicates[:5]}")
                    for slide_id in record.slide_id:
                        try:
                            path = resolve_feature_path(feature_root, slide_id)
                            with h5py.File(path, "r") as handle:
                                if "features" not in handle or "coords" not in handle:
                                    missing = [key for key in ("features", "coords") if key not in handle]
                                    raise KeyError(f"missing datasets {missing}")
                                _validate_arrays(
                                    path,
                                    handle["features"].shape,
                                    handle["coords"].shape,
                                    feature_dim,
                                )
                                _validate_patch_size(
                                    handle["coords"].attrs.get("patch_size_level0", fallback),
                                    f"{path}: coords.attrs['patch_size_level0']",
                                )
                        except Exception as error:
                            errors.append(f"{task_name}/{split_name}/{slide_id}: {error}")
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise PreflightError(fold, errors)

    def get_data_loaders(self, fold: int, task_id: Optional[int] = None):
        task_id = self.current_task if task_id is None else int(task_id)
        if task_id < 0 or task_id >= self.N_TASKS:
            raise IndexError(f"task_id={task_id} outside [0,{self.N_TASKS - 1}]")
        train_set, val_set, test_set = self._datasets_for_task(task_id, fold)
        workers = int(getattr(self.args, "num_workers", 0))
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=workers, collate_fn=collate_MIL)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_MIL)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_MIL)
        if task_id == len(self.test_loaders):
            self.test_loaders.append(test_loader)
        elif task_id < len(self.test_loaders):
            self.test_loaders[task_id] = test_loader
        else:
            raise RuntimeError("Tasks must be loaded sequentially")
        self.train_loader, self.val_loader = train_loader, val_loader
        self.current_task = max(self.current_task, task_id + 1)
        self.i = self.current_task
        return train_loader, val_loader, test_loader

    def get_joint_data_loaders(self, fold: int):
        train_sets, val_sets = [], []
        self.test_loaders = []
        for task_id in range(self.N_TASKS):
            train_set, val_set, test_set = self._datasets_for_task(task_id, fold)
            train_sets.append(train_set)
            val_sets.append(val_set)
            self.test_loaders.append(DataLoader(
                test_set, batch_size=1, shuffle=False,
                num_workers=int(getattr(self.args, "num_workers", 0)), collate_fn=collate_MIL,
            ))
        workers = int(getattr(self.args, "num_workers", 0))
        self.train_loader = DataLoader(ConcatDataset(train_sets), batch_size=1, shuffle=True, num_workers=workers, collate_fn=collate_MIL)
        self.val_loader = DataLoader(ConcatDataset(val_sets), batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_MIL)
        self.current_task = self.N_TASKS
        self.i = self.N_TASKS
        return self.train_loader, self.val_loader, self.test_loaders[-1]

    def get_backbone(self):
        return build_mil_backbone(self.args, num_classes=self.total_num_classes)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None
