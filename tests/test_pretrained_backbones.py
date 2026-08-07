import unittest
from types import SimpleNamespace

import torch

from backbone.generic_mil import GenericMILBackbone
from backbone.pretrained_mil import (
    FeatherMILBackbone,
    TitanMILBackbone,
    _initialize_feather_classifier,
)
from models.utils.continual_model import ContinualModel


class FakeTitanVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(768, 768)

    def forward(self, features, coords, patch_size_level0, no_proj=True):
        assert patch_size_level0 in (512, 1024)
        return self.projection(features.mean(dim=0, keepdim=True))


class FakeFeatherCore(torch.nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.encoder = torch.nn.Linear(768, 512)
        self.classifier = torch.nn.Linear(512, num_classes)


class FakeFeather(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeFeatherCore()

    def forward(self, features, return_attention=True, return_slide_feats=True):
        slide = self.model.encoder(features.mean(dim=1))
        logits = self.model.classifier(slide)
        attention = torch.softmax(features[..., 0], dim=1).unsqueeze(1)
        return {"results": {"logits": logits}, "log": {
            "attention": attention,
            "slide_feats": slide,
        }}


class FakeNativeFeather(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeFeatherCore(num_classes=0)
        del self.model.classifier
        self.model.num_classes = 0
        self.config = SimpleNamespace(num_classes=0)

    def initialize_classifier(self, num_classes):
        self.model.classifier = torch.nn.Linear(512, int(num_classes))
        torch.nn.init.kaiming_uniform_(
            self.model.classifier.weight, nonlinearity="relu"
        )
        torch.nn.init.zeros_(self.model.classifier.bias)


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.randn(19, 768)
        self.coords = torch.arange(38).reshape(19, 2)
        self.patch_size = torch.tensor(1024)

    def _assert_contract_and_backward(self, model, real_attention):
        output = model([self.features, self.coords, self.patch_size])
        self.assertEqual(len(output), 5)
        self.assertEqual(output[0].shape, (1, 27))
        self.assertEqual(output[1].shape, (1, 27))
        self.assertEqual(output[2].shape, (1,))
        self.assertEqual(output[3].shape, (1, 19))
        output[0].sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        if not real_attention:
            self.assertTrue(torch.allclose(output[3], torch.full((1, 19), 1 / 19)))

    def test_titan_contract_freeze_and_backward(self):
        model = TitanMILBackbone(FakeTitanVision(), 27)
        self._assert_contract_and_backward(model, real_attention=False)
        frozen = TitanMILBackbone(FakeTitanVision(), 27, freeze=True)
        self.assertFalse(any(parameter.requires_grad for parameter in frozen.vision_encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in frozen.classifier.parameters()))

    def test_feather_contract_freeze_and_backward(self):
        model = FeatherMILBackbone(FakeFeather())
        self._assert_contract_and_backward(model, real_attention=True)
        frozen = FeatherMILBackbone(FakeFeather(), freeze=True)
        self.assertFalse(any(parameter.requires_grad for parameter in frozen.model.model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in frozen.model.model.classifier.parameters()))

    def test_feather_downstream_classifier_is_explicitly_initialized(self):
        remote = FakeNativeFeather()
        classifier = _initialize_feather_classifier(remote, 27)
        self.assertEqual(classifier.in_features, 512)
        self.assertEqual(classifier.out_features, 27)
        self.assertEqual(remote.config.num_classes, 27)
        self.assertEqual(remote.model.num_classes, 27)
        self.assertTrue(all(
            torch.isfinite(parameter).all() for parameter in classifier.parameters()
        ))
        self.assertTrue(torch.equal(classifier.bias, torch.zeros_like(classifier.bias)))


class SamplingTests(unittest.TestCase):
    def _model(self, max_patches):
        args = SimpleNamespace(lr=1e-3, backbone_max_patches=max_patches)
        return ContinualModel(
            GenericMILBackbone(8, 27, hidden_dim=8),
            torch.nn.functional.cross_entropy,
            args,
            None,
        )

    def test_eval_sampling_is_deterministic_and_train_respects_budget(self):
        model = self._model(10)
        features = torch.arange(800, dtype=torch.float32).reshape(100, 8)
        coords = torch.arange(200).reshape(100, 2)
        patch_size = torch.tensor(1024)
        eval_a = model.prepare_inputs(features, coords, patch_size, training=False)
        eval_b = model.prepare_inputs(features, coords, patch_size, training=False)
        self.assertEqual(eval_a[0].shape, (10, 8))
        self.assertTrue(torch.equal(eval_a[0], eval_b[0]))
        self.assertTrue(torch.equal(eval_a[1], eval_b[1]))
        train = model.prepare_inputs(features, coords, patch_size, training=True)
        self.assertEqual(train[0].shape, (10, 8))
        self.assertEqual(train[1].shape, (10, 2))
        self.assertEqual(int(train[2]), 1024)

    def test_zero_budget_keeps_full_bag(self):
        model = self._model(0)
        features = torch.randn(41, 8)
        coords = torch.zeros(41, 2, dtype=torch.long)
        prepared = model.prepare_inputs(features, coords, 512, training=True)
        self.assertEqual(prepared[0].shape[0], 41)
        self.assertEqual(int(prepared[2]), 512)


if __name__ == "__main__":
    unittest.main()
