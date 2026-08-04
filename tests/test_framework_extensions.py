import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from backbone.pretrained_mil import FeatherMILBackbone, TitanMILBackbone
from datasets.seq_wsi import MILBatch


class _FakeTitanVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(768, 768)

    def forward(self, features, coords, patch_size_level0, no_proj=True):
        del coords
        if not no_proj or patch_size_level0 != 1024:
            raise AssertionError("TITAN adapter changed the native call contract")
        return self.projection(features.mean(dim=0, keepdim=True))


class _FakeFeatherCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(768, 512)
        self.classifier = torch.nn.Linear(512, 27)


class _FakeFeather(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeFeatherCore()

    def forward(self, features, return_attention=True, return_slide_feats=True):
        if not return_attention or not return_slide_feats:
            raise AssertionError("FEATHER adapter must request attention and slide features")
        embedding = self.model.encoder(features.mean(dim=1))
        logits = self.model.classifier(embedding)
        attention = torch.softmax(features[..., 0], dim=1).unsqueeze(1)
        return {
            "results": {"logits": logits},
            "log": {"attention": attention, "slide_feats": embedding},
        }


class BackboneEmbeddingContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = torch.randn(11, 768)
        self.coords = torch.arange(22).reshape(11, 2)
        self.patch_size = torch.tensor(1024)

    def _assert_contract(self, model, embedding_dim, classifier):
        legacy = model([self.features, self.coords, self.patch_size])
        enriched = model.forward_with_embedding(
            self.features, self.coords, self.patch_size
        )

        self.assertEqual(len(legacy), 5)
        self.assertEqual(
            set(enriched),
            {"logits", "embedding", "attention", "auxiliary_loss"},
        )
        self.assertEqual(enriched["logits"].shape, (1, 27))
        self.assertEqual(enriched["embedding"].shape, (1, embedding_dim))
        self.assertEqual(enriched["attention"].shape, (1, 11))
        self.assertEqual(enriched["auxiliary_loss"].shape, torch.Size([]))
        self.assertTrue(torch.allclose(legacy[0], enriched["logits"]))
        self.assertTrue(torch.allclose(legacy[3], enriched["attention"]))
        self.assertIs(model.get_classifier(), classifier)

        (enriched["logits"].sum() + enriched["embedding"].sum()).backward()
        self.assertIsNotNone(classifier.weight.grad)

    def test_titan_embedding_contract_and_legacy_forward(self):
        model = TitanMILBackbone(_FakeTitanVision(), num_classes=27)
        self._assert_contract(model, 768, model.classifier)
        self.assertIsNotNone(model.vision_encoder.projection.weight.grad)

    def test_feather_embedding_contract_and_legacy_forward(self):
        model = FeatherMILBackbone(_FakeFeather())
        self._assert_contract(model, 512, model.model.model.classifier)
        self.assertIsNotNone(model.model.model.encoder.weight.grad)


class TrainingHelperContractTests(unittest.TestCase):
    @staticmethod
    def _batch(index, patch_count):
        return MILBatch(
            features=torch.full((patch_count, 768), float(index)),
            coords=torch.zeros(patch_count, 2, dtype=torch.long),
            patch_size_level0=torch.tensor(1024),
            labels=torch.tensor([index], dtype=torch.long),
        )

    def test_logical_batches_preserve_variable_bags_and_flush_remainder(self):
        from utils.training import _iter_logical_batches

        source = [self._batch(index, count) for index, count in enumerate((1, 3, 2, 5, 4))]
        groups = list(_iter_logical_batches(iter(source), group_size=2))

        self.assertEqual([len(group) for group in groups], [2, 2, 1])
        self.assertEqual(
            [[batch.features.shape[0] for batch in group] for group in groups],
            [[1, 3], [2, 5], [4]],
        )
        self.assertIs(groups[-1][0], source[-1])

    def test_loss_value_accepts_scalar_and_dictionary_results(self):
        from utils.training import _loss_value

        self.assertEqual(float(_loss_value(1.25)), 1.25)
        self.assertEqual(float(_loss_value(torch.tensor(2.5))), 2.5)
        result = {"loss": torch.tensor(3.75), "ce_loss": torch.tensor(1.0)}
        self.assertEqual(float(_loss_value(result)), 3.75)
        with self.assertRaises(ValueError):
            _loss_value({"ce_loss": torch.tensor(1.0)})


class _CheckpointDataset:
    def metadata(self, fold):
        return {
            "fold": int(fold),
            "task_order": ["task-a", "task-b"],
            "task_num_classes": [2, 3],
            "class_offsets": [0, 2],
            "total_num_classes": 5,
            "backbone_config": {"name": "titan", "revision": "pinned"},
        }


class _StatefulMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.device = torch.device("cpu")
        self.sample_count = 17
        self.replay_labels = torch.tensor([1, 4], dtype=torch.long)
        self.loaded_strict = None

    def get_checkpoint_state(self):
        return {
            "sample_count": self.sample_count,
            "replay_labels": self.replay_labels.clone(),
        }

    def load_checkpoint_state(self, state, strict=True):
        self.loaded_strict = strict
        self.sample_count = int(state["sample_count"])
        self.replay_labels = state["replay_labels"].clone()


class CheckpointHookTests(unittest.TestCase):
    def test_method_state_round_trip_uses_strict_load_hook(self):
        from utils.training import checkpoint_payload, load_checkpoint

        model = _StatefulMethod()
        dataset = _CheckpointDataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            payload = checkpoint_payload(model, dataset, fold=3)
            self.assertIn("method_state", payload)
            torch.save(payload, path)

            with torch.no_grad():
                model.weight.fill_(-9.0)
            model.sample_count = 0
            model.replay_labels = torch.empty(0, dtype=torch.long)
            load_checkpoint(model, path, dataset, fold=3)

        self.assertTrue(torch.equal(model.weight.detach(), torch.tensor([2.0])))
        self.assertEqual(model.sample_count, 17)
        self.assertTrue(torch.equal(model.replay_labels, torch.tensor([1, 4])))
        self.assertIs(model.loaded_strict, True)


class CompatibilityValidationTests(unittest.TestCase):
    @staticmethod
    def _args(model, backbone, *, freeze=False, feature_dim=768):
        return SimpleNamespace(
            model=model,
            backbone=backbone,
            backbone_freeze=freeze,
            feature_dim=feature_dim,
        )

    def test_rejects_unsupported_method_backbone_pairs_before_construction(self):
        from models import validate_model_configuration

        for model, backbone in (
            ("lwsr", "generic_mil"),
            ("micil", "generic_mil"),
            ("qpmil_vl", "feather"),
        ):
            with self.subTest(model=model, backbone=backbone), self.assertRaises(
                ValueError
            ):
                validate_model_configuration(self._args(model, backbone))

    def test_rejects_frozen_or_wrong_dimension_method_configuration(self):
        from models import validate_model_configuration

        for model in ("lwsr", "micil"):
            with self.subTest(model=model, reason="frozen"), self.assertRaises(
                ValueError
            ):
                validate_model_configuration(
                    self._args(model, "titan", freeze=True)
                )
            with self.subTest(model=model, reason="feature_dim"), self.assertRaises(
                ValueError
            ):
                validate_model_configuration(
                    self._args(model, "titan", feature_dim=512)
                )


if __name__ == "__main__":
    unittest.main()
