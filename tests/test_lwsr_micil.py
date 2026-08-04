import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.lwsr import Lwsr
from models.micil import Micil
from models.utils.wsi_replay import VariableBagReservoir
from utils.training import checkpoint_payload, load_checkpoint


class FakeEmbeddingBackbone(torch.nn.Module):
    supports_ssl = False

    def __init__(self, num_classes=6, embedding_dim=8):
        super().__init__()
        self.encoder = torch.nn.Linear(768, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def forward_with_embedding(self, features, coords, patch_size_level0):
        del coords
        if int(patch_size_level0) <= 0:
            raise ValueError("invalid patch size")
        embedding = torch.tanh(self.encoder(features.mean(dim=0, keepdim=True)))
        logits = self.classifier(embedding)
        attention = torch.full(
            (1, features.shape[0]),
            1.0 / features.shape[0],
            device=features.device,
        )
        return {
            "logits": logits,
            "embedding": embedding,
            "attention": attention,
            "auxiliary_loss": logits.sum() * 0.0,
        }

    def forward(self, inputs):
        output = self.forward_with_embedding(*inputs)
        logits = output["logits"]
        return (
            logits,
            logits.softmax(dim=1),
            logits.argmax(dim=1),
            output["attention"],
            output["auxiliary_loss"],
        )

    def get_classifier(self):
        return self.classifier


def method_args(**overrides):
    values = dict(
        lr=1e-3,
        backbone="titan",
        feature_dim=768,
        backbone_freeze=False,
        num_classes=6,
        seed=7,
        buffer_size=4,
        minibatch_size=2,
        bags_per_update=2,
        buffer_max_patches=4,
        pair_loss_weight=1.0,
        ce_loss_weight=1.0,
        dc_loss_weight=0.01,
        micil_replay=False,
        micil_weight_norm=True,
        kd_loss_weight=10.0,
        embedding_loss_weight=1.0,
        distillation_temperature=2.0,
        class_offsets=[0, 2, 4],
        task_num_classes=[2, 2, 2],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_bag(label, patch_count=7, value=None, device=torch.device("cpu")):
    if value is None:
        features = torch.randn(patch_count, 768, device=device)
    else:
        features = torch.full((patch_count, 768), float(value), device=device)
    coords = torch.arange(patch_count * 2, device=device).reshape(patch_count, 2)
    return features, coords, torch.tensor(1024, device=device), torch.tensor([label], device=device)


def build(method, args):
    torch.manual_seed(3)
    model = method(
        FakeEmbeddingBackbone(args.num_classes),
        torch.nn.functional.cross_entropy,
        args,
        None,
    )
    model.net.to(model.device)
    return model


class FakeTaskDataset:
    def __init__(self, task, targets):
        self.current_task = int(task) + 1
        self.train_loader = SimpleNamespace(
            dataset=SimpleNamespace(targets=list(targets))
        )


class FakeCheckpointDataset:
    def metadata(self, fold):
        return {
            "fold": int(fold),
            "task_order": ["first", "second", "third"],
            "task_num_classes": [2, 2, 2],
            "class_offsets": [0, 2, 4],
            "total_num_classes": 6,
            "backbone_config": {"name": "titan", "feature_dim": 768},
        }


class VariableBagReservoirTests(unittest.TestCase):
    def test_seeded_cpu_storage_sampling_and_round_trip(self):
        left = VariableBagReservoir(3, max_patches=4, seed=11)
        right = VariableBagReservoir(3, max_patches=4, seed=11)
        for label, patch_count in enumerate((8, 5, 7, 9, 6)):
            bag = make_bag(label, patch_count=patch_count, value=label)
            left.add(*bag)
            right.add(*bag)

        self.assertEqual(left.labels, right.labels)
        self.assertEqual(left.num_seen_examples, 5)
        self.assertTrue(all(item.features.device.type == "cpu" for item in left.all(torch.device("cpu"))))
        self.assertTrue(all(item.features.shape[0] == 4 for item in left.all(torch.device("cpu"))))
        self.assertEqual(
            [item.index for item in left.sample(2, torch.device("cpu"))],
            [item.index for item in right.sample(2, torch.device("cpu"))],
        )

        restored = VariableBagReservoir(3, max_patches=4, seed=99)
        restored.load_state_dict(left.state_dict())
        self.assertEqual(restored.labels, left.labels)
        self.assertEqual(restored.num_seen_examples, left.num_seen_examples)
        self.assertEqual(
            [item.index for item in restored.sample(3, torch.device("cpu"))],
            [item.index for item in left.sample(3, torch.device("cpu"))],
        )

    def test_checkpoint_rejects_buffer_configuration_mismatch(self):
        source = VariableBagReservoir(3, max_patches=4, seed=0)
        source.add(*make_bag(0))
        target = VariableBagReservoir(2, max_patches=4, seed=0)
        with self.assertRaisesRegex(ValueError, "capacity"):
            target.load_state_dict(source.state_dict())


class LwsrTests(unittest.TestCase):
    def test_pair_and_distance_losses_match_hand_constructed_relations(self):
        embeddings = torch.tensor([[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        self.assertAlmostEqual(float(Lwsr.pair_loss(embeddings, labels, 6)), 0.0)

        reference_embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
        reference = torch.cdist(reference_embeddings, reference_embeddings)
        current = reference_embeddings.index_select(0, torch.tensor([2, 0]))
        loss = Lwsr.distance_consistency_loss(
            reference, current, torch.tensor([2, 0])
        )
        self.assertAlmostEqual(float(loss), 0.0)

    def test_later_task_without_replay_still_has_pair_and_ce(self):
        model = build(Lwsr, method_args(buffer_size=0))
        metrics = model.observe_many(
            [make_bag(2, device=model.device)], task=1
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertGreater(metrics["loss_pair"], 0.0)
        self.assertGreater(metrics["loss_ce"], 0.0)
        self.assertEqual(metrics["replay_bags"], 0.0)

    def test_replay_dcr_and_checkpoint_round_trip(self):
        args = method_args()
        model = build(Lwsr, args)
        model.observe_many(
            [make_bag(0, value=0.0, device=model.device), make_bag(1, value=1.0, device=model.device)],
            task=0,
        )
        model.save_buffer(*make_bag(0, value=0.0, device=model.device), task=0)
        model.save_buffer(*make_bag(1, value=1.0, device=model.device), task=0)
        model.end_task()
        self.assertEqual(tuple(model.previous_dist_matrix.shape), (2, 2))

        with torch.no_grad():
            model.net.encoder.weight.add_(0.01)
        metrics = model.observe_many(
            [make_bag(2, value=0.5, device=model.device)], task=1
        )
        self.assertEqual(metrics["replay_bags"], 2.0)
        self.assertGreater(metrics["loss_dcr"], 0.0)

        restored = build(Lwsr, args)
        restored.load_checkpoint_state(model.get_checkpoint_state())
        self.assertEqual(restored.buffer.labels, model.buffer.labels)
        self.assertTrue(
            torch.equal(restored.previous_dist_matrix, model.previous_dist_matrix)
        )
        incompatible = build(Lwsr, method_args(dc_loss_weight=0.02))
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            incompatible.load_checkpoint_state(model.get_checkpoint_state())


class MicilTests(unittest.TestCase):
    def test_class_balancing_changes_single_bag_cross_entropy(self):
        model = build(Micil, method_args(micil_replay=False))
        model.begin_task(FakeTaskDataset(0, [0, 0, 0, 1]))
        equal_logits = torch.zeros((1, 2), device=model.device)
        common_loss = model._classification_loss(
            equal_logits, torch.tensor([0], device=model.device)
        )
        rare_loss = model._classification_loss(
            equal_logits, torch.tensor([1], device=model.device)
        )
        self.assertAlmostEqual(float(rare_loss / common_loss), 3.0, places=5)
        self.assertGreater(float(rare_loss), float(common_loss))

    def test_kd_matches_upstream_temperature_formula_without_rescaling(self):
        model = build(Micil, method_args(micil_replay=False))
        model.old_class_count = 2
        student = torch.tensor([[0.2, 1.1, -0.7]], device=model.device)
        teacher = torch.tensor([[1.4, -0.3, 0.5]], device=model.device)
        expected = torch.nn.functional.kl_div(
            torch.log_softmax(student[:, :2] / model.temperature, dim=1),
            torch.softmax(teacher[:, :2] / model.temperature, dim=1),
            reduction="batchmean",
        )
        actual = model._distillation_loss(student, teacher)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(
            torch.allclose(actual, expected * (model.temperature ** 2))
        )

    def test_no_replay_task_zero_then_frozen_teacher(self):
        model = build(Micil, method_args(micil_replay=False))
        self.assertFalse(hasattr(model, "buffer"))
        model.begin_task(FakeTaskDataset(0, [0, 0, 0, 1]))
        self.assertAlmostEqual(float(model.class_weights[0]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(model.class_weights[1]), 2.0, places=5)

        metrics = model.observe_many(
            [make_bag(0, device=model.device), make_bag(1, device=model.device)], task=0
        )
        self.assertEqual(metrics["loss_kd"], 0.0)
        self.assertEqual(metrics["loss_embedding"], 0.0)
        model.end_task()
        self.assertIsNotNone(model.teacher)
        self.assertFalse(model.teacher.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.teacher.parameters()))

        model.begin_task(FakeTaskDataset(1, [2, 2, 3, 3]))
        with torch.no_grad():
            model.net.encoder.weight.add_(0.01)
        later = model.observe_many(
            [make_bag(2, value=0.5, device=model.device)], task=1
        )
        self.assertTrue(torch.isfinite(torch.tensor(later["loss"])))
        self.assertGreater(later["loss_embedding"], 0.0)
        norms = model.net.get_classifier().weight.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_optional_replay_and_checkpoint_mode_guard(self):
        args = method_args(micil_replay=True)
        model = build(Micil, args)
        model.begin_task(FakeTaskDataset(0, [0, 1]))
        model.observe_many([make_bag(0, device=model.device)], task=0)
        model.save_buffer(*make_bag(0, device=model.device), task=0)
        model.save_buffer(*make_bag(1, device=model.device), task=0)
        model.end_task()
        model.begin_task(FakeTaskDataset(1, [2, 3]))
        metrics = model.observe_many([make_bag(2, device=model.device)], task=1)
        self.assertEqual(metrics["replay_bags"], 2.0)

        state = model.get_checkpoint_state()
        restored = build(Micil, args)
        restored.load_checkpoint_state(state)
        self.assertEqual(restored.buffer.labels, model.buffer.labels)
        self.assertIsNotNone(restored.teacher)
        self.assertFalse(restored.teacher.training)

        no_replay = build(Micil, method_args(micil_replay=False))
        with self.assertRaisesRegex(ValueError, "replay mode"):
            no_replay.load_checkpoint_state(state)
        incompatible = build(
            Micil,
            method_args(micil_replay=True, distillation_temperature=3.0),
        )
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            incompatible.load_checkpoint_state(state)

        dataset = FakeCheckpointDataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "micil.pt"
            torch.save(checkpoint_payload(model, dataset, 0), path)
            framework_restored = build(Micil, args)
            load_checkpoint(framework_restored, path, dataset, 0)
        self.assertEqual(framework_restored.buffer.labels, model.buffer.labels)
        self.assertIsNotNone(framework_restored.teacher)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    framework_restored.net.state_dict().values(),
                    model.net.state_dict().values(),
                )
            )
        )

    def test_invalid_backbone_feature_and_freeze_fail_fast(self):
        for override in (
            {"backbone": "generic_mil"},
            {"feature_dim": 512},
            {"backbone_freeze": True},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                build(Micil, method_args(**override))


if __name__ == "__main__":
    unittest.main()
