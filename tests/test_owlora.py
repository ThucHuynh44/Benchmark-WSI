import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.owlora import Owlora
from models.utils.owlora import (
    OWLoRALinear,
    attach_owlora,
    project_current_gradients,
)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, features, coords, patch_size, no_proj=True):
        del coords, patch_size, no_proj
        return self.projection(features.mean(dim=0, keepdim=True))


class TinyTitan(nn.Module):
    supports_ssl = False

    def __init__(self):
        super().__init__()
        self.vision_encoder = TinyVision()
        self.classifier = nn.Linear(4, 5)

    def get_classifier(self):
        return self.classifier

    def forward(self, inputs):
        features, coords, patch_size = inputs
        embedding = self.vision_encoder(features, coords, patch_size, no_proj=True)
        logits = self.classifier(embedding)
        return logits, logits.softmax(1), logits.argmax(1), None, logits.sum() * 0.0


class TinyFeatherCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 5)


class TinyFeatherRemote(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyFeatherCore()

    def forward(self, features, return_attention=True, return_slide_feats=True):
        del return_attention, return_slide_feats
        embedding = self.model.encoder(features.mean(dim=1))
        logits = self.model.classifier(embedding)
        return {
            "results": {"logits": logits},
            "log": {
                "attention": torch.softmax(features[..., 0], dim=1).unsqueeze(1),
                "slide_feats": embedding,
            },
        }


class TinyFeather(nn.Module):
    supports_ssl = False

    def __init__(self):
        super().__init__()
        self.model = TinyFeatherRemote()

    def get_classifier(self):
        return self.model.model.classifier

    def forward(self, inputs):
        features, _, _ = inputs
        output = self.model(
            features.unsqueeze(0), return_attention=True, return_slide_feats=True
        )
        logits = output["results"]["logits"]
        return logits, logits.softmax(1), logits.argmax(1), None, logits.sum() * 0.0


class DatasetState:
    def __init__(self, current_task):
        self.current_task = current_task


def make_args(backbone="titan"):
    return SimpleNamespace(
        backbone=backbone,
        feature_dim=768,
        backbone_freeze=False,
        num_classes=5,
        n_tasks=3,
        task_num_classes=[2, 1, 2],
        class_offsets=[0, 2, 3],
        task_order=["a", "b", "c"],
        owlora_rank=2,
        owlora_svd_energy=0.8,
        owlora_orthogonal_weight=0.25,
        bags_per_update=1,
        optimizer="adamw",
        lr=5.0e-2,
        optim_wd=0.1,
        adam_eps=1.0e-8,
        backbone_max_patches=0,
        seed=7,
    )


def make_batch(label):
    features = torch.tensor(
        [[0.4, -0.1, 0.2, 0.7], [0.3, 0.5, -0.2, 0.1]],
        dtype=torch.float32,
    )
    coords = torch.zeros(2, 2, dtype=torch.long)
    return features, coords, torch.tensor(1024), torch.tensor([label])


def build_method(backbone="titan"):
    net = TinyTitan() if backbone == "titan" else TinyFeather()
    with patch(
        "models.utils.continual_model.get_device",
        return_value=torch.device("cpu"),
    ):
        return Owlora(net, F.cross_entropy, make_args(backbone), None)


class OWLoRALayerTests(unittest.TestCase):
    def test_attach_is_initially_noop_excludes_classifier_and_triples_qkv_rank(self):
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(4, 12)
                self.alias = self.qkv
                self.classifier = nn.Linear(12, 3)

            def forward(self, x):
                return self.classifier(self.qkv(x))

        torch.manual_seed(4)
        root = Root()
        x = torch.randn(2, 4)
        expected = root(x)
        classifier = root.classifier
        modules = attach_owlora(
            root, root_name="encoder", classifier=classifier, rank=2
        )
        self.assertEqual(len(modules), 1)
        wrapped = next(iter(modules.values()))
        self.assertIsInstance(wrapped, OWLoRALinear)
        self.assertEqual(wrapped.adapter_rank, 6)
        self.assertIs(root.qkv, root.alias)
        self.assertIs(root.classifier, classifier)
        self.assertEqual(len(wrapped.lora_layers), 0)
        self.assertTrue(torch.allclose(root(x), expected))

    def test_low_memory_projection_matches_source_formula(self):
        module = OWLoRALinear(3, 4, adapter_rank=2)
        reference = module.add_reference(2)
        previous = module.add_task_adapter()
        current = module.add_task_adapter()
        with torch.no_grad():
            reference.down.weight.copy_(torch.tensor([[1., 0., 0.], [0., 0.5, 0.]]))
            reference.up.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10)
            previous.down.weight.copy_(torch.tensor([[0., 0., 1.], [0.5, 0., 0.]]))
            previous.up.weight.copy_(torch.flip(reference.up.weight, dims=(0,)))
        down_grad = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        up_grad = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 7
        current.down.weight.grad = down_grad.clone()
        current.up.weight.grad = up_grad.clone()
        old_layers = (reference, previous)
        expected_down = down_grad - sum(
            down_grad @ (old.down.weight.t() @ old.down.weight)
            for old in old_layers
        )
        expected_up = up_grad - sum(
            (old.up.weight @ old.up.weight.t()) @ up_grad
            for old in old_layers
        )
        project_current_gradients({"linear": module})
        self.assertTrue(torch.allclose(current.down.weight.grad, expected_down))
        self.assertTrue(torch.allclose(current.up.weight.grad, expected_up))


class OWLoRAMethodTests(unittest.TestCase):
    def test_titan_and_feather_wrap_encoder_but_not_classifier(self):
        titan = build_method("titan")
        self.assertIsInstance(titan.net.vision_encoder.projection, OWLoRALinear)
        self.assertNotIsInstance(titan.classifier, OWLoRALinear)
        feather = build_method("feather")
        self.assertIsInstance(feather.net.model.model.encoder, OWLoRALinear)
        self.assertNotIsInstance(feather.classifier, OWLoRALinear)

    def test_seen_class_ce_classifier_isolation_and_dynamic_growth(self):
        torch.manual_seed(11)
        model = build_method("titan")
        wrapped = next(iter(model.owlora_modules.values()))
        initial_parameters = sum(parameter.numel() for parameter in model.net.parameters())
        self.assertEqual(len(wrapped.lora_layers), 0)

        model.observe_many([make_batch(1)], task=0)
        model.end_task()
        self.assertEqual(len(wrapped.lora_layers), 2)
        self.assertEqual(wrapped.task_adapter_count, 1)
        self.assertGreater(
            sum(parameter.numel() for parameter in model.net.parameters()),
            initial_parameters,
        )

        model.begin_task(DatasetState(current_task=2))
        base_before = wrapped.weight.detach().clone()
        classifier_before = model.classifier.weight.detach().clone()
        logits_before = model.net(list(make_batch(2)[:3]))[0]
        expected_ce = F.cross_entropy(logits_before[:, :3].float(), torch.tensor([2]))
        result = model.observe_many([make_batch(2)], task=1)
        self.assertAlmostEqual(result["loss_ce"], float(expected_ce), places=6)
        self.assertTrue(torch.equal(wrapped.weight.detach(), base_before))
        self.assertTrue(torch.equal(model.classifier.weight[:2], classifier_before[:2]))
        self.assertTrue(torch.equal(model.classifier.weight[3:], classifier_before[3:]))
        self.assertFalse(torch.equal(model.classifier.weight[2], classifier_before[2]))

        model.end_task()
        self.assertEqual(len(wrapped.lora_layers), 3)
        self.assertEqual(wrapped.task_adapter_count, 2)
        model.begin_task(DatasetState(current_task=3))
        model.end_task()
        self.assertEqual(len(wrapped.lora_layers), 3)

    def test_checkpoint_reconstructs_dynamic_layout_on_fresh_model(self):
        torch.manual_seed(19)
        source = build_method("titan")
        source.observe_many([make_batch(0)], task=0)
        source.end_task()
        source.begin_task(DatasetState(current_task=2))
        source.observe_many([make_batch(2)], task=1)
        source.end_task()
        state_dict = copy.deepcopy(source.state_dict())
        method_state = copy.deepcopy(source.get_checkpoint_state())

        restored = build_method("titan")
        self.assertEqual(
            len(next(iter(restored.owlora_modules.values())).lora_layers), 0
        )
        restored.load_state_dict(state_dict, strict=True)
        restored.load_checkpoint_state(method_state, strict=True)
        self.assertEqual(
            len(next(iter(restored.owlora_modules.values())).lora_layers), 3
        )
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        self.assertEqual(source.get_checkpoint_state(), restored.get_checkpoint_state())

    def test_loading_task_zero_best_checkpoint_removes_newer_adapters(self):
        torch.manual_seed(23)
        best_task_zero = build_method("titan")
        state_dict = copy.deepcopy(best_task_zero.state_dict())
        method_state = copy.deepcopy(best_task_zero.get_checkpoint_state())

        expanded = build_method("titan")
        expanded.observe_many([make_batch(1)], task=0)
        expanded.end_task()
        self.assertEqual(
            len(next(iter(expanded.owlora_modules.values())).lora_layers), 2
        )
        expanded.load_state_dict(state_dict, strict=True)
        expanded.load_checkpoint_state(method_state, strict=True)
        self.assertEqual(
            len(next(iter(expanded.owlora_modules.values())).lora_layers), 0
        )


if __name__ == "__main__":
    unittest.main()
