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

from models.atlas_v3_acl import (
    ACL_MODES,
    NORMALIZED_OAS_MODES,
    OAS_MODES,
    build_model_from_components,
)
from models.utils.atlas_transport import (
    bootstrap_gates,
    distribution_coverage,
    fit_full_residual,
    fit_lowrank_residual,
)
from scripts.atlas_v3_acl_registry import (
    EXPECTED_EPOCHS,
    EXPECTED_MODES,
    EXPECTED_RANKS,
    FULL_RANK_IDS,
    SETTING_IDS,
    load_registry,
)
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
        atlasv3_acl_transport_rank=2, atlasv3_acl_transport_ridge=1e-3,
        atlasv3_acl_transport_full_rank=False,
        atlasv3_acl_transport_mean_scale=1.0,
        atlasv3_acl_transport_cov_scale=1.0,
        atlasv3_acl_coverage_energy=0.95,
        atlasv3_acl_bootstrap_samples=6, atlasv3_acl_uncertainty_beta=10.0,
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
    def test_registry_has_core_and_rank_ablation_settings(self):
        registry = load_registry(REGISTRY)
        self.assertEqual(tuple(registry["variants"]), SETTING_IDS)
        self.assertEqual(len(SETTING_IDS), 17)
        for variant_id, variant in registry["variants"].items():
            expected = {"atlasv3_acl_mode": EXPECTED_MODES[variant_id]}
            if variant_id in EXPECTED_RANKS:
                expected["atlasv3_acl_transport_rank"] = EXPECTED_RANKS[variant_id]
            if variant_id in FULL_RANK_IDS:
                expected["atlasv3_acl_transport_full_rank"] = True
            if variant_id in EXPECTED_EPOCHS:
                expected["n_epochs"] = EXPECTED_EPOCHS[variant_id]
            self.assertEqual(variant["overrides"], expected)
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

    def test_full_fit_keeps_complete_ridge_residual(self):
        torch.manual_seed(14)
        source = torch.randn(20, 6)
        target = source @ (torch.eye(6) + 0.03 * torch.randn(6, 6))
        lowrank = fit_lowrank_residual(source, target, rank=2, ridge=1e-6)
        full = fit_full_residual(source, target, ridge=1e-6)
        self.assertGreater(full.effective_rank, lowrank.effective_rank)
        self.assertLessEqual(
            F.mse_loss(full.map(source), target),
            F.mse_loss(lowrank.map(source), target),
        )

    def test_coverage_and_bootstrap_are_deterministic(self):
        torch.manual_seed(9)
        source = torch.randn(12, 5)
        target = source + 0.02 * torch.randn(12, 5)
        points = source[:3]
        main = fit_lowrank_residual(source, target, rank=2, ridge=1e-3)
        scatters = torch.eye(5).repeat(3, 1, 1)
        counts = torch.full((3,), 4)
        coverage = distribution_coverage(points, scatters, counts, source, energy=0.95)
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

    def test_zero_acl_epochs_is_a_true_frozen_encoder_control(self):
        model = build(args(
            n_epochs=0,
            atlasv3_acl_mode="gated_transport_normalized_oas_no_histneg",
        ))
        before = copy.deepcopy(model.net.backbone.state_dict())
        dataset = learn_task(model, 0)
        self.assertTrue(model.TRAINING_FREE)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.net.backbone.parameters()))
        for name, value in model.net.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(bool(model.net.lda_fitted))
        self.assertEqual(model.transport_history[-1]["effective_rank"], 0.0)

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


    def test_normalized_oas_static_uses_normalized_statistics_and_no_histneg(self):
        model = build(args(atlasv3_acl_mode="normalized_oas_static"))
        dataset = learn_task(model, 0)
        raw, labels, _ = model._collect_loader(dataset.train_loader)
        normalized = F.normalize(raw.float(), dim=1, eps=1.0e-8)
        for label in range(2):
            expected = normalized[labels == label].mean(0)
            self.assertTrue(torch.allclose(model.net.raw_mean[label].cpu(), expected, atol=1e-6))

        features, coords, patch_size, _ = bag(0, patches=9)
        prepared = model.prepare_inputs(features, coords, patch_size, training=False)
        embedding = model.net.encode(*prepared)
        expected_logits = F.linear(
            F.normalize(embedding.float(), dim=1, eps=1.0e-8),
            model.net.lda_weight,
            model.net.lda_bias,
        )
        actual_logits = model(features, coords, patch_size)[0]
        self.assertTrue(torch.allclose(actual_logits[:, :2], expected_logits[:, :2], atol=1e-6))

        second = TaskData(1, task_values(1))
        model.begin_task(second)
        batch = next(iter(second.train_loader))
        prepared = model.prepare_inputs(
            batch.features, batch.coords, batch.patch_size_level0, training=True
        )
        metrics = model.observe(*prepared, batch.labels, task=1)
        self.assertNotIn("loss_histneg", metrics)

    def test_main_method_transports_distribution_statistics(self):
        mode = "gated_transport_normalized_oas_no_histneg"
        model = build(args(atlasv3_acl_mode=mode))
        learn_task(model, 0)
        learn_task(model, 1)
        row = model.transport_history[-1]
        self.assertEqual(row["transport_kind"], mode)
        self.assertIn("coverage_mean", row)
        self.assertIn("step_gate_mean", row)
        self.assertTrue(torch.isfinite(model.net.raw_mean[:4]).all())
        self.assertEqual(NORMALIZED_OAS_MODES, {
            "normalized_oas_static", "transport_normalized_oas",
            "gated_transport_normalized_oas_no_histneg",
        })

    def test_full_rank_setting_uses_untruncated_transport_end_to_end(self):
        model = build(args(
            atlasv3_acl_mode="gated_transport_normalized_oas_no_histneg",
            atlasv3_acl_transport_full_rank=True,
        ))
        learn_task(model, 0)
        learn_task(model, 1)
        row = model.transport_history[-1]
        self.assertEqual(row["requested_rank"], "full")
        self.assertGreater(row["effective_rank"], 0.0)
        self.assertEqual(
            model.get_run_metadata()["atlas_v3_acl_config"]["implementation_semantics"],
            "acl_only_normalized_oas_gated_full_ridge_transport_v1",
        )

    def test_normalized_oas_transport_controls_exclude_histneg(self):
        for mode, gated in (
            ("transport_normalized_oas", False),
            ("gated_transport_normalized_oas_no_histneg", True),
        ):
            with self.subTest(mode=mode):
                model = build(args(atlasv3_acl_mode=mode))
                learn_task(model, 0)
                second = TaskData(1, task_values(1))
                model.begin_task(second)
                batch = next(iter(second.train_loader))
                prepared = model.prepare_inputs(
                    batch.features,
                    batch.coords,
                    batch.patch_size_level0,
                    training=True,
                )
                metrics = model.observe(*prepared, batch.labels, task=1)
                self.assertNotIn("loss_histneg", metrics)
                model.end_task(second)
                row = model.transport_history[-1]
                self.assertEqual(row["transport_kind"], mode)
                self.assertEqual("step_gate_mean" in row, gated)

    def test_zero_transport_scales_are_an_exact_static_statistics_control(self):
        model = build(args(
            atlasv3_acl_mode="gated_transport_normalized_oas_no_histneg",
            atlasv3_acl_transport_mean_scale=0.0,
            atlasv3_acl_transport_cov_scale=0.0,
        ))
        learn_task(model, 0)
        old_mean = model.net.raw_mean[:2].clone()
        old_scatter = model.net.raw_scatter[:2].clone()
        learn_task(model, 1)
        self.assertTrue(torch.equal(model.net.raw_mean[:2], old_mean))
        self.assertTrue(torch.equal(model.net.raw_scatter[:2], old_scatter))
        row = model.transport_history[-1]
        self.assertEqual(row["applied_mean_gate_mean"], 0.0)
        self.assertEqual(row["applied_covariance_gate_mean"], 0.0)


    def test_in_process_best_checkpoint_restore_preserves_pair_cache(self):
        model = build(args())
        dataset = TaskData(0, task_values(0))
        model.begin_task(dataset)
        cached = model._pair_pre_raw.clone()
        model.load_checkpoint_state(copy.deepcopy(model.get_checkpoint_state()), strict=True)
        self.assertTrue(torch.equal(model._pair_pre_raw, cached))
        model.end_task(dataset)

    def test_checkpoint_round_trip_after_task_boundary(self):
        options = args(atlasv3_acl_mode="gated_transport_normalized_oas_no_histneg")
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
