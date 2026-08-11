import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.amil import Amil, validate_args
from models.utils.amil_memory import (
    PseudoBag,
    PseudoBagMemoryPool,
    maxminrand_select,
)
from utils.training import checkpoint_payload, load_checkpoint


CPU = torch.device("cpu")


class FakeAttentionBackbone(torch.nn.Module):
    supports_ssl = False
    has_genuine_patch_attention = True

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.attention_head = torch.nn.Linear(feature_dim, 1)
        self.encoder = torch.nn.Linear(feature_dim, 8)
        self.classifier = torch.nn.Linear(8, num_classes)

    def forward(self, inputs):
        features, coords, patch_size = inputs
        if coords.shape != (features.shape[0], 2) or int(patch_size) <= 0:
            raise ValueError("invalid fake WSI metadata")
        attention = torch.softmax(
            self.attention_head(features).transpose(0, 1), dim=1
        )
        embedding = torch.tanh(self.encoder(attention @ features))
        logits = self.classifier(embedding)
        return (
            logits,
            logits.softmax(dim=1),
            logits.argmax(dim=1),
            attention,
            logits.sum() * 0.0,
        )


class FakeUniformBackbone(FakeAttentionBackbone):
    has_genuine_patch_attention = False


class FakeTaskDataset:
    def __init__(self, task_id):
        self.current_task = int(task_id) + 1


class FakeCheckpointDataset:
    def __init__(self, args):
        self.args = args

    def metadata(self, fold):
        return {
            "fold": int(fold),
            "task_order": ["a", "b", "c"],
            "task_num_classes": list(self.args.task_num_classes),
            "class_offsets": list(self.args.class_offsets),
            "total_num_classes": self.args.num_classes,
            "optimizer_config": {
                "name": "adamw",
                "lr": self.args.lr,
                "weight_decay": self.args.optim_wd,
                "eps": self.args.adam_eps,
            },
            "backbone_config": {
                "name": self.args.backbone,
                "feature_dim": self.args.feature_dim,
            },
        }


def method_args(**overrides):
    values = dict(
        lr=1.0e-3,
        optimizer="adamw",
        optim_wd=0.0,
        adam_eps=1.0e-8,
        backbone="generic_mil",
        feature_dim=4,
        backbone_freeze=False,
        backbone_max_patches=0,
        num_classes=5,
        seed=17,
        buffer_size=5,
        minibatch_size=1,
        bags_per_update=1,
        pmp_k=3,
        alpha=1.0,
        beta=1.0,
        kd_temperature=1.0,
        class_offsets=[0, 2, 3],
        task_num_classes=[2, 1, 2],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_bag(label, *, patches=5, feature_dim=4, value=None):
    if value is None:
        features = torch.randn(patches, feature_dim)
    else:
        offsets = torch.linspace(-0.2, 0.2, patches).unsqueeze(1)
        features = torch.full((patches, feature_dim), float(value)) + offsets
    coords = torch.arange(patches * 2, dtype=torch.long).reshape(patches, 2)
    return features, coords, torch.tensor(1024), torch.tensor([label])


def targetless_bag(label, *, origin=0, patches=3, num_classes=4):
    del num_classes
    features, coords, patch_size, tensor_label = make_bag(
        label, patches=patches, feature_dim=4, value=label
    )
    return PseudoBag(
        features=features,
        coords=coords,
        patch_size=patch_size,
        label=tensor_label,
        origin_task_id=origin,
    )


def refresh_pairs(entries, num_classes, offset=0.0):
    return [
        (
            torch.full((1, entry.features.shape[0]), 1.0 / entry.features.shape[0]),
            torch.arange(num_classes, dtype=torch.float32).unsqueeze(0) + offset,
        )
        for entry in entries
    ]


def build_model(args=None, backbone=None):
    args = args or method_args()
    backbone = backbone or FakeAttentionBackbone(args.feature_dim, args.num_classes)
    with patch("models.utils.continual_model.get_device", return_value=CPU):
        torch.manual_seed(7)
        model = Amil(backbone, F.cross_entropy, args, None)
    return model.to(CPU)


class MaxMinRandTests(unittest.TestCase):
    def test_quotas_are_disjoint_sorted_and_reproducible(self):
        attention = torch.arange(10, dtype=torch.float32).unsqueeze(0)
        left_generator = torch.Generator().manual_seed(9)
        right_generator = torch.Generator().manual_seed(9)
        left = maxminrand_select(attention, 8, generator=left_generator)
        right = maxminrand_select(attention, 8, generator=right_generator)

        self.assertTrue(torch.equal(left, right))
        self.assertEqual(left.numel(), 8)
        self.assertEqual(torch.unique(left).numel(), 8)
        self.assertTrue(torch.equal(left, left.sort().values))
        self.assertTrue({0, 1}.issubset(set(left.tolist())))
        self.assertTrue({8, 9}.issubset(set(left.tolist())))

    def test_small_bag_uses_every_patch_and_preserves_alignment(self):
        attention = torch.tensor([[0.4, 0.1, 0.3, 0.2]])
        indices = maxminrand_select(
            attention, 400, generator=torch.Generator().manual_seed(3)
        )
        self.assertTrue(torch.equal(indices, torch.arange(4)))
        features = torch.arange(12).reshape(4, 3)
        coords = torch.arange(8).reshape(4, 2) + 100
        selected_features = features.index_select(0, indices)
        selected_coords = coords.index_select(0, indices)
        for row, index in enumerate(indices.tolist()):
            self.assertTrue(torch.equal(selected_features[row], features[index]))
            self.assertTrue(torch.equal(selected_coords[row], coords[index]))


class BalancedReservoirTests(unittest.TestCase):
    @staticmethod
    def _offer(pool, label, origin):
        decision = pool.consider(label)
        if decision is not None:
            pool.commit(decision, targetless_bag(label, origin=origin))
        return decision

    def _ready_two_class_pool(self):
        pool = PseudoBagMemoryPool(5, pmp_k=3, seed=13, num_classes=4)
        pool.start_update(2, task_id=0)
        for label in (0, 0, 0, 0, 1, 1, 1, 1):
            self._offer(pool, label, 0)
        entries = pool.all(CPU, require_targets=False)
        pool.refresh_targets(
            refresh_pairs(entries, 4),
            target_seen_class_count=2,
            target_snapshot_task=0,
        )
        return pool

    def test_quota_shrink_preserves_historical_seen_counts(self):
        pool = self._ready_two_class_pool()
        before = pool.seen_count
        self.assertEqual(before[:2], (4, 4))

        pool.start_update(4, task_id=1)
        self.assertEqual(pool.seen_count, before)
        active_quotas = [value for value in pool.quotas[:4]]
        self.assertEqual(sum(active_quotas), 5)
        self.assertLessEqual(max(active_quotas) - min(active_quotas), 1)

        label = 2
        candidates = 20
        for _ in range(candidates):
            self._offer(pool, label, 1)
        self.assertEqual(pool.seen_count[label], candidates)
        self.assertLessEqual(len(pool), pool.capacity)

    def test_pending_refresh_blocks_replay_and_checkpoint(self):
        pool = self._ready_two_class_pool()
        pool.start_update(4, task_id=1)
        self._offer(pool, 2, 1)
        with self.assertRaisesRegex(RuntimeError, "refresh"):
            pool.sample(1, CPU)
        with self.assertRaisesRegex(RuntimeError, "refresh"):
            pool.state_dict()

    def test_atomic_refresh_origin_and_round_trip_rng(self):
        pool = self._ready_two_class_pool()
        old_entries = pool.all(CPU)
        old_attention = old_entries[0].target_attention.clone()
        pool.start_update(4, task_id=1)
        self._offer(pool, 2, 1)
        entries = pool.all(CPU, require_targets=False)
        targets = refresh_pairs(entries, 4, offset=2.0)
        invalid = list(targets)
        invalid[-1] = (torch.ones(1, 999), invalid[-1][1])
        with self.assertRaises(ValueError):
            pool.refresh_targets(
                invalid,
                target_seen_class_count=4,
                target_snapshot_task=1,
            )
        self.assertTrue(pool.refresh_required)
        self.assertTrue(torch.equal(
            pool.all(CPU, require_targets=False)[0].target_attention,
            old_attention,
        ))

        pool.refresh_targets(
            targets,
            target_seen_class_count=4,
            target_snapshot_task=1,
        )
        refreshed = pool.all(CPU)
        self.assertEqual(pool.target_snapshot_task, 1)
        self.assertTrue(all(entry.target_seen_class_count == 4 for entry in refreshed))
        self.assertIn(0, {entry.origin_task_id for entry in refreshed})
        self.assertIn(1, {entry.origin_task_id for entry in refreshed})

        restored = PseudoBagMemoryPool(5, pmp_k=3, seed=99, num_classes=4)
        restored.load_state_dict(pool.state_dict())
        self.assertEqual(restored.seen_count, pool.seen_count)
        self.assertEqual(restored.quotas, pool.quotas)
        self.assertEqual(
            [int(item.label) for item in restored.sample(3, CPU)],
            [int(item.label) for item in pool.sample(3, CPU)],
        )

        tampered_state = pool.state_dict()
        tampered_state["entries"][0]["target_seen_class_count"] = 1
        with self.assertRaisesRegex(ValueError, "target class count"):
            restored.load_state_dict(tampered_state)

        probe = torch.linspace(0, 1, 7).unsqueeze(0)
        self.assertTrue(torch.equal(
            maxminrand_select(probe, 5, generator=restored.selection_generator),
            maxminrand_select(probe, 5, generator=pool.selection_generator),
        ))
        pool.start_update(4, task_id=2)
        restored.start_update(4, task_id=2)
        left_decision = pool.consider(2)
        right_decision = restored.consider(2)
        self.assertEqual(left_decision, right_decision)


class AmilLossAndLifecycleTests(unittest.TestCase):
    def test_ce_and_logit_kd_use_different_seen_class_slices(self):
        model = build_model()
        model.current_seen_class_count = 3
        logits = torch.tensor([[2.0, 0.5, -1.0, 20.0, 30.0]])
        label = torch.tensor([1])
        actual_ce = model._classification_loss(logits, label)
        expected_ce = F.cross_entropy(logits[:, :3], label)
        self.assertTrue(torch.allclose(actual_ce, expected_ce))

        target = torch.tensor([[0.1, 1.2, 99.0, -50.0, 70.0]])
        actual_kd = model.logit_distillation_loss(logits, target, 2)
        expected_kd = F.kl_div(
            F.log_softmax(logits[:, :2], dim=1),
            F.softmax(target[:, :2], dim=1),
            reduction="batchmean",
        )
        self.assertTrue(torch.allclose(actual_kd, expected_kd))

    def test_distillation_identity_and_perturbation(self):
        model = build_model()
        attention = torch.tensor([[0.1, 0.2, 0.7]])
        logits = torch.tensor([[0.2, -0.5, 1.1, 8.0, -9.0]])
        self.assertAlmostEqual(
            float(model.attention_distillation_loss(attention, attention)),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(model.logit_distillation_loss(logits, logits, 2)),
            0.0,
            places=6,
        )
        self.assertGreater(
            float(model.attention_distillation_loss(
                torch.tensor([[0.7, 0.2, 0.1]]), attention
            )),
            0.0,
        )

    def test_three_task_refresh_replay_and_origins(self):
        model = build_model()
        model.begin_task(FakeTaskDataset(0))
        first_bags = [make_bag(0, value=-0.5), make_bag(1, value=0.5)]
        metrics = model.observe_many(first_bags, task=0)
        self.assertEqual(metrics["replay_bags"], 0.0)
        for bag in first_bags:
            model.save_buffer(*bag, task=0)
        model.end_task()
        first_targets = {
            int(entry.label): entry.target_logits.clone()
            for entry in model.buffer.all(CPU)
        }
        self.assertEqual(model.buffer.target_snapshot_task, 0)

        model.begin_task(FakeTaskDataset(1))
        replay_metrics = model.observe_many([make_bag(2, value=1.0)], task=1)
        self.assertEqual(replay_metrics["replay_bags"], 1.0)
        with torch.no_grad():
            model.net.classifier.bias.add_(0.25)
        model.save_buffer(*make_bag(2, value=1.0), task=1)
        model.end_task()
        second_entries = model.buffer.all(CPU)
        self.assertEqual(model.buffer.target_snapshot_task, 1)
        self.assertTrue(all(entry.target_seen_class_count == 3 for entry in second_entries))
        self.assertIn(0, {entry.origin_task_id for entry in second_entries})
        self.assertIn(1, {entry.origin_task_id for entry in second_entries})
        retained_old = [entry for entry in second_entries if int(entry.label) in first_targets]
        self.assertTrue(any(
            not torch.allclose(entry.target_logits, first_targets[int(entry.label)])
            for entry in retained_old
        ))

        model.begin_task(FakeTaskDataset(2))
        final_metrics = model.observe_many([make_bag(3, value=1.5)], task=2)
        self.assertEqual(final_metrics["replay_bags"], 1.0)

    def test_optimizer_objective_is_mean_of_current_and_replay_bag_losses(self):
        model = build_model()
        model.begin_task(FakeTaskDataset(0))
        old_bag = make_bag(0, value=-0.4)
        model.save_buffer(*old_bag, task=0)
        model.end_task()
        model.begin_task(FakeTaskDataset(1))
        replay_entry = model.buffer.all(CPU)[0]
        current_bag = make_bag(2, value=0.8)

        current_features, current_coords, current_patch, current_label = current_bag
        current_logits, _ = model._forward_attention(
            current_features, current_coords, current_patch
        )
        current_ce = model._classification_loss(current_logits, current_label)
        replay_ce, replay_attn, replay_logits = model._replay_losses(replay_entry)
        expected = (
            current_ce
            + replay_ce
            + model.alpha * replay_attn
            + model.beta * replay_logits
        ) / 2

        with patch.object(model.buffer, "sample", return_value=[replay_entry]):
            metrics = model.observe_many([current_bag], task=1)
        self.assertAlmostEqual(metrics["loss"], float(expected.detach()), places=6)

    def test_checkpoint_round_trip_restores_network_optimizer_and_memory(self):
        args = method_args()
        model = build_model(args)
        model.begin_task(FakeTaskDataset(0))
        bag = make_bag(0, value=-0.2)
        model.observe_many([bag], task=0)
        model.save_buffer(*bag, task=0)
        model.end_task()

        dataset = FakeCheckpointDataset(args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amil.pt"
            torch.save(checkpoint_payload(model, dataset, 0), path)
            restored = build_model(args)
            load_checkpoint(restored, path, dataset, 0)
        for name, value in model.net.state_dict().items():
            torch.testing.assert_close(value, restored.net.state_dict()[name])
        self.assertEqual(restored.buffer.seen_count, model.buffer.seen_count)
        self.assertEqual(restored.buffer.target_snapshot_task, 0)
        self.assertTrue(torch.equal(
            restored.buffer.all(CPU)[0].target_logits,
            model.buffer.all(CPU)[0].target_logits,
        ))

    def test_feather_contract_runs_without_remote_weights(self):
        args = method_args(
            backbone="feather",
            feature_dim=768,
            num_classes=3,
            class_offsets=[0, 2],
            task_num_classes=[2, 1],
        )
        model = build_model(
            args, FakeAttentionBackbone(args.feature_dim, args.num_classes)
        )
        model.begin_task(FakeTaskDataset(0))
        bag = make_bag(0, patches=3, feature_dim=768, value=0.1)
        self.assertTrue(torch.isfinite(torch.tensor(
            model.observe_many([bag], task=0)["loss"]
        )))
        model.save_buffer(*bag, task=0)
        model.end_task()
        self.assertEqual(model.buffer.target_snapshot_task, 0)

    def test_invalid_backbone_freeze_patch_cap_and_weights_fail_fast(self):
        invalid_args = (
            method_args(backbone="titan"),
            method_args(backbone_freeze=True),
            method_args(backbone_max_patches=1),
            method_args(alpha=0.0),
            method_args(beta=0.0),
        )
        for args in invalid_args:
            with self.subTest(args=vars(args)), self.assertRaises(ValueError):
                validate_args(args)
        with self.assertRaisesRegex(ValueError, "genuine patch attention"):
            build_model(backbone=FakeUniformBackbone(4, 5))


if __name__ == "__main__":
    unittest.main()
