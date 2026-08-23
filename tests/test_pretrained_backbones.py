import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backbone.generic_mil import GenericMILBackbone
from backbone.pretrained_mil import (
    FeatherMILBackbone,
    GigaPathMILBackbone,
    TitanMILBackbone,
    _load_gigapath_slide,
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
        # Native FEATHER logs signed, pre-softmax attention scores [B,1,N].
        attention = features[..., 0].unsqueeze(1)
        return {"results": {"logits": logits}, "log": {
            "attention": attention,
            "slide_feats": slide,
        }}


class FakeFeatherWithoutAttention(FakeFeather):
    def forward(self, features, return_attention=True, return_slide_feats=True):
        del return_attention, return_slide_feats
        slide = self.model.encoder(features.mean(dim=1))
        return {
            "results": {"logits": self.model.classifier(slide)},
            "log": {"slide_feats": slide},
        }


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


class FakeGigaPathSlide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Linear(1536, 768)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 768))
        self.tile_size = 256
        self.slide_ngrids = 1000
        self.tile_size_at_forward = None
        self.all_layer_embed = None

    def coords_to_pos(self, coords, tile_size=256):
        grid = torch.floor(coords / tile_size)
        return (grid[..., 0] * self.slide_ngrids + grid[..., 1]).long() + 1

    def forward(self, features, coords, all_layer_embed=False):
        self.tile_size_at_forward = self.tile_size
        self.all_layer_embed = all_layer_embed
        embedded = self.patch_embed.proj(features).mean(dim=1)
        return [embedded, torch.full_like(embedded, -999)]


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
        self.assertTrue(torch.isfinite(output[3]).all())
        self.assertTrue(torch.all(output[3] >= 0))
        self.assertTrue(torch.allclose(output[3].sum(dim=1), torch.ones(1)))
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
        output = model([self.features, self.coords, self.patch_size])
        expected = torch.softmax(self.features[:, 0], dim=0).unsqueeze(0)
        self.assertTrue(torch.allclose(output[3], expected))
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

    def test_patch_attention_capabilities_are_explicit(self):
        self.assertTrue(GenericMILBackbone.has_genuine_patch_attention)
        self.assertTrue(FeatherMILBackbone.has_genuine_patch_attention)
        self.assertFalse(TitanMILBackbone.has_genuine_patch_attention)
        self.assertFalse(GigaPathMILBackbone.has_genuine_patch_attention)

    def test_gigapath_runtime_tile_size_output_index_and_backward(self):
        encoder = FakeGigaPathSlide()
        model = GigaPathMILBackbone(encoder, 27)
        self.assertEqual(encoder.tile_size, 256)
        features = torch.randn(19, 1536)
        coords = torch.tensor([[0, 1023], [1024, 2048]]).repeat(10, 1)[:19]
        output = model([features, coords, torch.tensor(1024)])
        self.assertEqual(encoder.tile_size_at_forward, 1024)
        self.assertFalse(encoder.all_layer_embed)
        self.assertEqual(output[0].shape, (1, 27))
        self.assertEqual(output[3].shape, (1, 19))
        self.assertTrue(torch.allclose(output[3], torch.full((1, 19), 1 / 19)))
        enriched = model.forward_with_embedding(features, coords, 1024)
        self.assertEqual(enriched["embedding"].shape, (1, 768))
        enriched["logits"].sum().backward()
        self.assertIsNotNone(encoder.patch_embed.proj.weight.grad)

        frozen = GigaPathMILBackbone(FakeGigaPathSlide(), 27, freeze=True)
        self.assertFalse(any(
            parameter.requires_grad for parameter in frozen.slide_encoder.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad for parameter in frozen.classifier.parameters()
        ))

    def test_gigapath_coords_to_pos_uses_runtime_tile_size(self):
        encoder = FakeGigaPathSlide()
        coords = torch.tensor([[[0, 1023], [1024, 2048], [1023999, 1023999]]])
        positions = encoder.coords_to_pos(coords, 1024)
        expected_grid = torch.floor(coords / 1024).long()
        expected = expected_grid[..., 0] * 1000 + expected_grid[..., 1] + 1
        self.assertTrue(torch.equal(positions, expected))

    def test_generic_mil_exposes_embedding_and_classifier_contract(self):
        model = GenericMILBackbone(768, 27, hidden_dim=16)
        enriched = model.forward_with_embedding(
            self.features, self.coords, self.patch_size
        )
        self.assertEqual(enriched["embedding"].shape, (1, 16))
        self.assertEqual(enriched["attention"].shape, (1, 19))
        self.assertIs(model.get_classifier(), model.classifier)

    def test_feather_does_not_fallback_to_uniform_attention(self):
        model = FeatherMILBackbone(FakeFeatherWithoutAttention())
        with self.assertRaisesRegex(ValueError, "genuine patch attention"):
            model([self.features, self.coords, self.patch_size])


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


class GigaPathCheckpointContractTests(unittest.TestCase):
    def _load(self, mutate=None):
        template = FakeGigaPathSlide()
        state = {key: value.detach().clone() for key, value in template.state_dict().items()}
        if mutate is not None:
            mutate(state)
        calls = {}

        def factory(**kwargs):
            calls["kwargs"] = dict(kwargs)
            return FakeGigaPathSlide()

        package = types.ModuleType("gigapath")
        package.__path__ = []
        module = types.ModuleType("gigapath.slide_encoder")
        module.gigapath_slide_enc12l768d = factory
        package.slide_encoder = module
        with patch.dict(
            sys.modules,
            {"gigapath": package, "gigapath.slide_encoder": module},
        ), patch("torch.load", return_value={"model": state}):
            loaded = _load_gigapath_slide("slide_encoder.pth")
        return loaded, calls

    def test_constructor_and_exact_release_contract(self):
        loaded, calls = self._load()
        self.assertIsInstance(loaded, FakeGigaPathSlide)
        self.assertEqual(calls["kwargs"], {"in_chans": 1536, "global_pool": True})
        self.assertNotIn("tile_size", calls["kwargs"])

    def test_missing_cls_token_is_never_allowed(self):
        with self.assertRaisesRegex(RuntimeError, "cls_token"):
            self._load(lambda state: state.pop("cls_token"))

    def test_future_unexpected_key_fails(self):
        with self.assertRaisesRegex(RuntimeError, "contract changed"):
            self._load(lambda state: state.update({"future.weight": torch.ones(1)}))

    def test_projection_shape_and_finite_weights_are_validated(self):
        with self.assertRaisesRegex(ValueError, "projection mismatch"):
            self._load(lambda state: state.__setitem__(
                "patch_embed.proj.weight", torch.ones(768, 12)
            ))
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self._load(lambda state: state["cls_token"].fill_(float("nan")))


if __name__ == "__main__":
    unittest.main()
