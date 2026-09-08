import copy
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.atlas_v3 import (
    DISTRIBUTION_MODES,
    build_model_from_components,
    get_parser,
    validate_args,
)
from scripts.atlas_v3_registry import EXPECTED_MODES, SETTING_IDS, load_registry
from scripts.run_atlas_v3_ablations import build_command


ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "configs" / "atlas_v3_ablations.yaml"
Batch = namedtuple("Batch", "features coords patch_size_level0 labels")


class TinyCore(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.encoder = nn.Linear(8, 4)
        self.classifier = nn.Linear(4, classes)

    def forward_embedding(self, features):
        return torch.tanh(self.encoder(features)).mean(0, keepdim=True)


class TinyFeather(nn.Module):
    supports_ssl = False

    def __init__(self, classes=4):
        super().__init__()
        self.model = TinyCore(classes)

    def get_classifier(self):
        return self.model.classifier

    def forward_with_embedding(self, features, coords, patch_size_level0):
        embedding = self.model.forward_embedding(features)
        return {"embedding": embedding, "logits": self.model.classifier(embedding)}


def args(**overrides):
    values = dict(
        optimizer="adamw",
        lr=1e-2,
        optim_wd=0.0,
        adam_eps=1e-8,
        backbone="feather",
        feature_dim=8,
        backbone_freeze=False,
        backbone_max_patches=0,
        num_classes=4,
        n_tasks=2,
        task_num_classes=[2, 2],
        class_offsets=[0, 2],
        task_order=["brca", "nsclc"],
        seed=5,
        fold=1,
        atlasv3_distribution_mode="prototype",
        atlasv3_distribution_alpha=0.1,
        atlasv3_distribution_rho=0.5,
        atlasv3_distribution_rank=8,
        atlasv3_distribution_clusters=2,
        atlasv3_distribution_tau_multi=0.1,
        atlasv3_distribution_beta=0.25,
        atlasv3_distribution_tau_task=0.1,
        atlasv3_pt_steps=2,
        atlasv3_pt_lr=0.05,
        atlasv3_pt_samples_per_class=4,
        atlasv3_pt_margin=0.2,
        ablation_id=None,
        ablation_group=None,
        ablation_config_hash=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def bag(label, patches=7, shift=0.0):
    generator = torch.Generator().manual_seed(1000 + label * 31 + patches)
    features = torch.randn(patches, 8, generator=generator) + float(shift)
    coords = torch.arange(patches * 2).reshape(patches, 2)
    return features, coords, torch.tensor(1024), torch.tensor([label])


class Bags(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        features, coords, patch_size, label = self.values[index]
        return features, coords, patch_size, int(label.item())


def collate(batch):
    features, coords, patch_size, label = batch[0]
    return Batch(features, coords, torch.as_tensor(patch_size), torch.tensor([label]))


class TaskData:
    def __init__(self, task, values):
        self.current_task = task + 1
        self.train_loader = DataLoader(
            Bags(values), batch_size=1, shuffle=False, collate_fn=collate
        )
        self.val_loaders = [self.train_loader]


def build(options):
    torch.manual_seed(3)
    with patch("models.atlas_v3.validate_args", return_value=None), patch(
        "models.utils.continual_model.get_device", return_value=torch.device("cpu")
    ):
        return build_model_from_components(
            options, F.cross_entropy, None, TinyFeather(options.num_classes)
        ).to("cpu")


class AtlasV3RegistryTests(unittest.TestCase):
    def test_registry_has_exactly_the_eleven_frozen_modes(self):
        registry = load_registry(REGISTRY)
        self.assertEqual(tuple(registry["variants"]), SETTING_IDS)
        self.assertEqual(len(SETTING_IDS), 11)
        for variant_id, variant in registry["variants"].items():
            self.assertEqual(
                variant["overrides"],
                {"atlasv3_distribution_mode": EXPECTED_MODES[variant_id]},
            )
            command = build_command(registry, variant, 2)
            self.assertIn("atlas_v3", command)
            self.assertIn("feather", command)

    def test_implementation_has_no_removed_mechanism_code(self):
        source = (ROOT / "models" / "atlas_v3.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("lora", source)
        self.assertNotIn("replay", source)
        self.assertNotIn("save_buffer", source)


class AtlasV3Tests(unittest.TestCase):
    def test_parser_exposes_exact_classifier_modes(self):
        parser = get_parser()
        parsed = parser.parse_args(
            ["--dataset", "seq-wsi", "--exp_desc", "test", "--model", "atlas_v3"]
        )
        self.assertEqual(parsed.atlasv3_distribution_mode, "prototype")
        action = next(
            item
            for item in parser._actions
            if item.dest == "atlasv3_distribution_mode"
        )
        self.assertEqual(
            tuple(action.choices), ("prototype", "oas_lda", *DISTRIBUTION_MODES)
        )

    def test_validation_rejects_invalid_backbone_and_distribution_values(self):
        baseline = args(feature_dim=768)
        validate_args(baseline)
        for override in (
            {"backbone": "titan"},
            {"feature_dim": 512},
            {"backbone_max_patches": 1},
            {"atlasv3_distribution_rank": 3},
            {"atlasv3_distribution_clusters": 4},
            {"atlasv3_distribution_rho": 1.1},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_args(args(**{"feature_dim": 768, **override}))

    def test_all_parameters_are_frozen(self):
        model = build(args())
        self.assertTrue(model.TRAINING_FREE)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.net.parameters()))
        self.assertFalse(hasattr(model, "memory"))

    def test_ncm_and_oas_lda_fit_train_only_statistics(self):
        task = TaskData(
            0,
            [
                bag(0, patches=7, shift=-1.0),
                bag(0, patches=8, shift=-0.8),
                bag(1, patches=7, shift=0.8),
                bag(1, patches=8, shift=1.0),
            ],
        )
        ncm = build(args())
        ncm.begin_task(task)
        ncm.end_task(task)
        self.assertTrue(ncm.net.prototype_valid[:2].all())
        self.assertTrue(torch.isfinite(ncm(*bag(0, patches=9)[:3])[0][:, :2]).all())

        lda = build(args(atlasv3_distribution_mode="oas_lda"))
        lda.begin_task(task)
        lda.end_task(task)
        self.assertEqual(lda.net.lda_counts[:2].tolist(), [2, 2])
        self.assertTrue(bool(lda.net.lda_fitted))

    def test_distribution_state_and_checkpoint_round_trip(self):
        options = args(atlasv3_distribution_mode="diag")
        model = build(options)
        task = TaskData(
            0,
            [
                bag(0, patches=7, shift=-1.0),
                bag(0, patches=8, shift=-0.8),
                bag(1, patches=7, shift=0.8),
                bag(1, patches=8, shift=1.0),
            ],
        )
        model.begin_task(task)
        model.end_task(task)
        self.assertEqual(model.net.distribution_head.class_count[:2].tolist(), [2, 2])
        self.assertEqual(model.calibration_history[0]["split"], "validation")

        state = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        restored = build(options)
        restored.load_state_dict(state, strict=True)
        restored.load_checkpoint_state(method_state, strict=True)
        self.assertEqual(restored.calibration_history, model.calibration_history)


if __name__ == "__main__":
    unittest.main()
