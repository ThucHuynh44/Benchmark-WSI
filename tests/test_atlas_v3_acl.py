import copy
from collections import namedtuple
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.atlas_v3_acl import ACL_MODES, OAS_MODES, build_model_from_components
from models.utils.atlas_transport import (
    bootstrap_gates,
    fit_lowrank_residual,
    mean_coverage,
)
from scripts.atlas_v3_acl_registry import EXPECTED_MODES, SETTING_IDS, load_registry
from scripts.run_atlas_v3_acl_ablations import build_command


ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "configs" / "atlas_v3_acl_ablations.yaml"
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
        optimizer="adamw", lr=1e-2, optim_wd=0.0, adam_eps=1e-8,
        backbone="feather", feature_dim=8, backbone_freeze=False,
        backbone_max_patches=0, num_classes=4, n_tasks=2,
        task_num_classes=[2, 2], class_offsets=[0, 2],
        task_order=["brca", "nsclc"], seed=5, fold=1,
        atlasv3_acl_mode="acl", atlasv3_acl_temperature=0.1,
        atlasv3_acl_hist_weight=1.0, atlasv3_acl_hist_temperature=0.1,
        atlasv3_acl_hist_margin=0.2, atlasv3_acl_hist_topk=2,
        atlasv3_acl_transport_rank=2, atlasv3_acl_transport_ridge=1e-3,
        atlasv3_acl_ldc_steps=3, atlasv3_acl_ldc_lr=1e-2,
        atlasv3_acl_sdc_sigma=0.3, atlasv3_acl_coverage_energy=0.95,
        atlasv3_acl_bootstrap_samples=6, atlasv3_acl_uncertainty_beta=10.0,
        atlasv3_acl_reliability_floor=0.1,
        atlasv3_acl_reliability_momentum=0.5,
        atlasv3_acl_task_margin=0.2, atlasv3_acl_task_weight=0.1,
        ablation_id=None, ablation_group=None, ablation_config_hash=None,
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
        self.train_loader = DataLoader(Bags(values), batch_size=1, shuffle=True, collate_fn=collate)

    def _datasets_for_task(self, task_id, fold):
        dataset = Bags(task_values(int(task_id)))
        return dataset, dataset, dataset


def task_values(task):
    start = task * 2
    return [
        bag(start, patches=7, shift=-1.0),
        bag(start, patches=8, shift=-0.8),
        bag(start + 1, patches=7, shift=0.8),
        bag(start + 1, patches=8, shift=1.0),
    ]


def build(options):
    torch.manual_seed(3)
    with patch("models.atlas_v3_acl.validate_args", return_value=None), patch(
        "models.utils.continual_model.get_device", return_value=torch.device("cpu")
    ):
        return build_model_from_components(
            options, F.cross_entropy, None, TinyFeather(options.num_classes)
        ).to("cpu")


def learn_task(model, task):
    dataset = TaskData(task, task_values(task))
    model.begin_task(dataset)
    if not model.TRAINING_FREE:
        for batch in dataset.train_loader:
            features, coords, patch_size = model.prepare_inputs(
                batch.features, batch.coords, batch.patch_size_level0, training=True
            )
            model.observe(features, coords, patch_size, batch.labels, task=task)
    model.end_task(dataset)
    return dataset


class AtlasV3ACLRegistryTests(unittest.TestCase):
    def test_registry_has_exactly_fourteen_non_confounded_settings(self):
        registry = load_registry(REGISTRY)
        self.assertEqual(tuple(registry["variants"]), SETTING_IDS)
        self.assertEqual(len(SETTING_IDS), 14)
        self.assertIn("atlasv3_acl_histneg_lowrank_transport", SETTING_IDS)
        for variant_id, variant in registry["variants"].items():
            self.assertEqual(variant["overrides"], {"atlasv3_acl_mode": EXPECTED_MODES[variant_id]})
            command = build_command(registry, variant, 2)
            self.assertIn("atlas_v3_acl", command)
            self.assertIn("feather", command)

    def test_training_model_has_no_old_validation_or_exemplar_path(self):
        source = (ROOT / "models" / "atlas_v3_acl.py").read_text(encoding="utf-8")
        self.assertNotIn("val_loader", source)
        self.assertNotIn("save_buffer", source)
        self.assertNotIn("FullBagReplayBuffer", source)


class TransportTests(unittest.TestCase):
    def test_lowrank_fit_recovers_supported_drift_and_caps_rank(self):
        torch.manual_seed(4)
        source = torch.randn(20, 6)
        delta = torch.zeros(6, 6)
        delta[:, :2] = torch.randn(6, 2) * 0.05
        target = source @ (torch.eye(6) + delta)
        fitted = fit_lowrank_residual(source, target, rank=2, ridge=1e-6)
        self.assertLessEqual(fitted.effective_rank, 2)
        self.assertLess(F.mse_loss(fitted.map(source), target).item(), 1e-5)

    def test_coverage_and_bootstrap_are_deterministic(self):
        torch.manual_seed(9)
        source = torch.randn(12, 5)
        target = source + 0.02 * torch.randn(12, 5)
        points = source[:3]
        main = fit_lowrank_residual(source, target, rank=2, ridge=1e-3)
        coverage = mean_coverage(points, source, energy=0.95)
        first = bootstrap_gates(points, source, target, coverage, main, rank=2, ridge=1e-3, samples=8, beta=10.0, seed=31)
        second = bootstrap_gates(points, source, target, coverage, main, rank=2, ridge=1e-3, samples=8, beta=10.0, seed=31)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])


class AtlasV3ACLTests(unittest.TestCase):
    def test_modes_match_registry(self):
        self.assertEqual(set(ACL_MODES), set(EXPECTED_MODES.values()))

    def test_encoder_lifecycle_and_no_exemplar_memory(self):
        model = build(args())
        dataset = TaskData(0, task_values(0))
        model.begin_task(dataset)
        self.assertTrue(model.net.backbone.model.encoder.weight.requires_grad)
        self.assertFalse(model.net.classifier.weight.requires_grad)
        self.assertFalse(hasattr(model, "memory"))
        model.end_task(dataset)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.net.backbone.parameters()))
        self.assertIsNone(model._pair_pre_raw)

    def test_all_modes_complete_two_tasks_with_finite_logits(self):
        for mode in ACL_MODES:
            with self.subTest(mode=mode):
                model = build(args(atlasv3_acl_mode=mode))
                learn_task(model, 0)
                learn_task(model, 1)
                logits = model(*bag(2, patches=9)[:3])[0]
                self.assertTrue(torch.isfinite(logits[:, :4]).all())
                self.assertEqual(model.completed_tasks, 2)
                self.assertEqual(len(model.transport_history), 2)
                if mode not in OAS_MODES:
                    self.assertEqual(model.net.raw_scatter.numel(), 0)

    def test_static_and_oracle_oas_controls_are_explicit(self):
        static = build(args(atlasv3_acl_mode="oas_static"))
        learn_task(static, 0)
        old_mean = static.net.raw_mean[:2].clone()
        old_gate = static.net.last_step_gate[:2].clone()
        learn_task(static, 1)
        self.assertTrue(torch.equal(static.net.raw_mean[:2], old_mean))
        self.assertTrue(torch.equal(static.net.last_step_gate[:2], old_gate))
        self.assertEqual(
            static.transport_history[-1]["transport_fallback_reason"],
            "static_old_statistics",
        )

        oracle = build(args(atlasv3_acl_mode="oas_oracle"))
        learn_task(oracle, 0)
        learn_task(oracle, 1)
        self.assertGreater(oracle.transport_history[-1]["oracle_revisited_wsis"], 0)
        self.assertTrue(oracle.transport_history[-1]["diagnostic_only"])
        class_rows = oracle.transport_history[-1]["oracle_class_diagnostics"]
        self.assertEqual(len(class_rows), 2)
        self.assertIn("mean_drift_cosine", class_rows[0])
        self.assertIn("mean_residual_ungated_cosine", class_rows[0])
        self.assertIn("mean_residual_gated_cosine", class_rows[0])
        self.assertIn("covariance_drift_relative_frobenius", class_rows[0])
        self.assertIn("covariance_residual_gated_relative_frobenius", class_rows[0])
        for row in class_rows:
            for key, value in row.items():
                self.assertTrue(
                    not isinstance(value, float) or torch.isfinite(torch.tensor(value)),
                    msg=f"non-finite oracle diagnostic {key}={value}",
                )
        metadata = oracle.get_run_metadata()["atlas_v3_acl_config"]
        self.assertTrue(metadata["diagnostic_only"])
        self.assertTrue(metadata["revisits_old_train_data"])
        self.assertIn("oracle_recompute_with_drift_probe", metadata["implementation_semantics"])
        json.dumps(oracle.get_run_metadata(), allow_nan=False)

    def test_histneg_is_zero_on_first_task_and_positive_afterward(self):
        model = build(args(atlasv3_acl_mode="histneg"))
        first = TaskData(0, task_values(0))
        model.begin_task(first)
        batch = next(iter(first.train_loader))
        features, coords, patch_size = model.prepare_inputs(batch.features, batch.coords, batch.patch_size_level0, training=True)
        metrics = model.observe(features, coords, patch_size, batch.labels, task=0)
        self.assertEqual(metrics["loss_histneg"], 0.0)
        model.end_task(first)
        second = TaskData(1, task_values(1))
        model.begin_task(second)
        batch = next(iter(second.train_loader))
        features, coords, patch_size = model.prepare_inputs(batch.features, batch.coords, batch.patch_size_level0, training=True)
        metrics = model.observe(features, coords, patch_size, batch.labels, task=1)
        self.assertGreater(metrics["loss_histneg"], 0.0)

    def test_in_process_best_checkpoint_restore_preserves_pair_cache(self):
        model = build(args())
        dataset = TaskData(0, task_values(0))
        model.begin_task(dataset)
        cached = model._pair_pre_raw.clone()
        model.load_checkpoint_state(copy.deepcopy(model.get_checkpoint_state()), strict=True)
        self.assertTrue(torch.equal(model._pair_pre_raw, cached))
        model.end_task(dataset)

    def test_checkpoint_round_trip_after_task_boundary(self):
        options = args(atlasv3_acl_mode="gated")
        model = build(options)
        learn_task(model, 0)
        state = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        for transient in ("pair_pre_raw", "pair_labels", "pair_indices"):
            self.assertNotIn(transient, method_state)
            self.assertFalse(any(transient in key for key in state))
        restored = build(options)
        restored.load_state_dict(state, strict=True)
        restored.load_checkpoint_state(method_state, strict=True)
        self.assertEqual(restored.transport_history, model.transport_history)
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.net.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
