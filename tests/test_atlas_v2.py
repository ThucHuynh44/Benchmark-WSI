import copy
from collections import namedtuple
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.atlas_v2 import (
    CoMELOWLoRALinear, FullBagReplayBuffer, SVDOrthogonalLoRALinear,
    StandardLoRALinear,
    build_model_from_components, get_parser, validate_args,
)


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
        if coords.shape != (features.shape[0], 2) or int(patch_size_level0) <= 0:
            raise ValueError("bad bag metadata")
        embedding = self.model.forward_embedding(features)
        return {"embedding": embedding, "logits": self.model.classifier(embedding)}


def args(**overrides):
    values = dict(
        optimizer="adamw", lr=1e-2, optim_wd=0.0, adam_eps=1e-8,
        backbone="feather", feature_dim=8, backbone_freeze=False,
        backbone_max_patches=0, num_classes=4, n_tasks=2,
        task_num_classes=[2, 2], class_offsets=[0, 2],
        task_order=["brca", "nsclc"], seed=5, fold=1,
        buffer_size=0, minibatch_size=1, bags_per_update=1,
        atlasv2_lora=False, atlasv2_replay=False, atlasv2_prototype=False,
        atlasv2_realign=False, atlasv2_prompt=False, atlasv2_nce=False,
        atlasv2_train_classifier=True, atlasv2_lora_rank=2,
        atlasv2_lora_alpha=2.0, atlasv2_lora_merge_scale=1.0,
        atlasv2_svd_orthogonal=False, atlasv2_svd_energy=0.99,
        atlasv2_comel_owlora=False, atlasv2_comel_svd_energy=0.99,
        atlasv2_comel_orthogonal_weight=1.0,
        atlasv2_prompt_fusion=0.5, atlasv2_prompt_ce_weight=1.0,
        atlasv2_nce_temperature=0.07, atlasv2_nce_weight=1.0,
        atlasv2_text_model_id="fixed", atlasv2_text_revision="fixed",
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
        self.train_loader = DataLoader(
            Bags(values), batch_size=1, shuffle=False, collate_fn=collate
        )


ANCHORS = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def build(options):
    torch.manual_seed(3)
    backbone = TinyFeather(options.num_classes)
    anchors = ANCHORS if options.atlasv2_prompt else None
    # Unit tests deliberately use an 8-D toy bag; production validation of the
    # required 768-D FEATHER contract is covered separately.
    with patch("models.atlas_v2.validate_args", return_value=None), patch(
        "models.utils.continual_model.get_device", return_value=torch.device("cpu")
    ):
        return build_model_from_components(
            options, F.cross_entropy, None, backbone, anchors
        ).to("cpu")


class LoRATests(unittest.TestCase):
    def test_standard_merge_is_exact_dense_addition(self):
        torch.manual_seed(4)
        source = nn.Linear(5, 3)
        layer = StandardLoRALinear(source, rank=2, alpha=2.0)
        with torch.no_grad():
            layer.lora_a.normal_()
            layer.lora_b.normal_()
        x = torch.randn(6, 5)
        before = layer(x)
        layer.merge()
        after = layer(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-5))
        self.assertEqual(torch.count_nonzero(layer.lora_b), 0)
        self.assertFalse(layer.weight.requires_grad)

    def test_svd_basis_projects_future_update_and_merge_is_exact(self):
        torch.manual_seed(14)
        layer = SVDOrthogonalLoRALinear(
            nn.Linear(5, 4, bias=False), rank=2, alpha=2.0,
            max_basis_rank=4, energy_threshold=0.99,
        )
        with torch.no_grad():
            layer.lora_a.normal_()
            layer.lora_b.normal_()
        layer.merge()
        old_basis = layer.historical_basis().clone()
        self.assertGreater(old_basis.shape[1], 0)
        self.assertTrue(torch.allclose(
            old_basis.t() @ old_basis,
            torch.eye(old_basis.shape[1]), atol=1e-5,
        ))

        with torch.no_grad():
            layer.lora_a.normal_()
            layer.lora_b.normal_()
        projected_delta = layer.projected_lora_b() @ layer.lora_a
        self.assertTrue(torch.allclose(
            old_basis.t() @ projected_delta,
            torch.zeros(old_basis.shape[1], projected_delta.shape[1]),
            atol=1e-5,
        ))
        inputs = torch.randn(7, 5)
        before = layer(inputs)
        layer.merge()
        after = layer(inputs)
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_comel_owlora_is_cumulative_and_projects_current_gradients(self):
        torch.manual_seed(22)
        layer = CoMELOWLoRALinear(
            nn.Linear(5, 4, bias=False), rank=2, n_tasks=2,
            energy_threshold=0.99,
        )
        self.assertFalse(layer.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in layer.task_adapters[0].parameters()))
        self.assertTrue(all(not p.requires_grad for p in layer.task_adapters[1].parameters()))
        with torch.no_grad():
            layer.task_adapters[0].up.weight.normal_()
            layer.task_adapters[1].up.weight.normal_()
        inputs = torch.randn(3, 5)
        task0_output = layer(inputs)
        layer.set_task(1)
        task1_output = layer(inputs)
        expected = task0_output + layer.task_adapters[1](inputs)
        self.assertTrue(torch.allclose(task1_output, expected, atol=1e-6))
        self.assertTrue(all(not p.requires_grad for p in layer.task_adapters[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in layer.task_adapters[1].parameters()))
        self.assertTrue(torch.isfinite(layer.orthogonality_penalty()))

        current = layer.current_adapter()
        current.down.weight.grad = torch.randn_like(current.down.weight)
        current.up.weight.grad = torch.randn_like(current.up.weight)
        down_before = current.down.weight.grad.clone()
        up_before = current.up.weight.grad.clone()
        historical = layer.historical_adapters()
        expected_down = down_before - sum(
            (down_before @ old.down.weight.detach().t()) @ old.down.weight.detach()
            for old in historical
        )
        expected_up = up_before - sum(
            old.up.weight.detach() @ (old.up.weight.detach().t() @ up_before)
            for old in historical
        )
        layer.project_current_gradients()
        self.assertTrue(torch.allclose(current.down.weight.grad, expected_down))
        self.assertTrue(torch.allclose(current.up.weight.grad, expected_up))


class FullBagReplayTests(unittest.TestCase):
    def test_capacity_balance_full_bags_and_checkpoint(self):
        memory = FullBagReplayBuffer(30, feature_dim=8, num_classes=4, seed=9)
        original_lengths = {}
        for index in range(80):
            label = index % 4
            value = bag(label, patches=5 + index % 9)
            entry = memory.make_entry(*value, origin_task=index // 20)
            original_lengths[(label, entry.priority)] = value[0].shape[0]
            memory.add(entry, range(4))
            self.assertLessEqual(len(memory), 30)
        counts = {label: len(memory.by_label(label, "cpu")) for label in range(4)}
        self.assertEqual(sorted(counts.values()), [7, 7, 8, 8])
        for entry in memory.entries:
            self.assertEqual(entry.features.shape[0], original_lengths[(int(entry.label), entry.priority)])
            self.assertEqual(entry.coords.shape[0], entry.features.shape[0])
        accounting = memory.accounting()
        self.assertEqual(accounting["retained_wsis"], 30)
        self.assertEqual(
            accounting["retained_patch_rows"],
            sum(entry.features.shape[0] for entry in memory.entries),
        )
        state = copy.deepcopy(memory.state_dict())
        restored = FullBagReplayBuffer(30, 8, 4, seed=1)
        restored.load_state_dict(state)
        self.assertEqual([int(e.label) for e in restored.entries], [int(e.label) for e in memory.entries])
        self.assertTrue(all(
            torch.equal(left.features, right.features)
            for left, right in zip(restored.entries, memory.entries)
        ))


class AtlasV2Tests(unittest.TestCase):
    def test_parser_exposes_comel_strategy_defaults(self):
        parsed = get_parser().parse_args([
            "--dataset", "seq-wsi", "--exp_desc", "test", "--model", "atlas_v2",
        ])
        self.assertFalse(parsed.atlasv2_comel_owlora)
        self.assertEqual(parsed.atlasv2_comel_svd_energy, 0.99)
        self.assertEqual(parsed.atlasv2_comel_orthogonal_weight, 1.0)

    def test_production_validation_rejects_hidden_or_invalid_coupling(self):
        baseline = args(feature_dim=768)
        validate_args(baseline)
        invalid = (
            {"backbone": "titan"}, {"feature_dim": 512},
            {"backbone_max_patches": 1},
            {"atlasv2_replay": True, "buffer_size": 29},
            {"atlasv2_realign": True}, {"atlasv2_nce": True},
            {"atlasv2_comel_owlora": True},
            {
                "atlasv2_lora": True, "atlasv2_comel_owlora": True,
                "atlasv2_svd_orthogonal": True,
            },
            {"atlasv2_comel_svd_energy": 1.0},
            {"atlasv2_comel_orthogonal_weight": -1.0},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_args(args(**{"feature_dim": 768, **override}))

    def test_frozen_and_lora_gradient_flow_and_no_hidden_modules(self):
        frozen = build(args())
        frozen.begin_task(TaskData(0, [bag(0)]))
        frozen.observe(*bag(0), task=0)
        base_parameters = [p for name, p in frozen.net.named_parameters() if "classifier" not in name]
        self.assertTrue(all(not parameter.requires_grad for parameter in base_parameters))
        self.assertIsNotNone(frozen.net.classifier.weight.grad)
        self.assertFalse(hasattr(frozen.net, "prototype_bank"))
        self.assertFalse(hasattr(frozen.net, "prompt_projector"))
        self.assertIsNone(frozen.memory)

        lora = build(args(atlasv2_lora=True))
        lora.begin_task(TaskData(0, [bag(0)]))
        lora.observe(*bag(0), task=0)
        self.assertTrue(lora.lora_modules)
        self.assertTrue(all(module.weight.grad is None for module in lora.lora_modules.values()))
        self.assertTrue(all(module.lora_b.grad is not None for module in lora.lora_modules.values()))
        self.assertIsNotNone(lora.net.classifier.weight.grad)

    def test_svd_orthogonal_variant_trains_projected_lora_and_tracks_basis(self):
        options = args(atlasv2_lora=True, atlasv2_svd_orthogonal=True)
        model = build(options)
        task0 = TaskData(0, [bag(0), bag(1)])
        model.begin_task(task0)
        model.observe(*bag(0), task=0)
        self.assertTrue(all(
            isinstance(module, SVDOrthogonalLoRALinear)
            for module in model.lora_modules.values()
        ))
        self.assertTrue(all(module.lora_b.grad is not None for module in model.lora_modules.values()))
        model.end_task(task0)
        self.assertTrue(any(
            int(module.svd_basis_rank) > 0 for module in model.lora_modules.values()
        ))
        state = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        restored = build(options)
        restored.load_state_dict(state)
        restored.load_checkpoint_state(method_state)
        self.assertEqual(
            [int(module.svd_basis_rank) for module in restored.lora_modules.values()],
            [int(module.svd_basis_rank) for module in model.lora_modules.values()],
        )

    def test_comel_owlora_switches_task_adapter_and_round_trips(self):
        options = args(atlasv2_lora=True, atlasv2_comel_owlora=True)
        model = build(options)
        task0 = TaskData(0, [bag(0), bag(1)])
        model.begin_task(task0)
        result = model.observe(*bag(0), task=0)
        self.assertGreaterEqual(result["loss_comel_orthogonal"], 0.0)
        self.assertTrue(all(
            isinstance(module, CoMELOWLoRALinear)
            for module in model.lora_modules.values()
        ))
        self.assertTrue(all(
            module.current_adapter().up.weight.grad is not None
            for module in model.lora_modules.values()
        ))
        before_end = [
            module.task_adapters[0].up.weight.detach().clone()
            for module in model.lora_modules.values()
        ]
        model.end_task(task0)
        self.assertTrue(all(
            torch.equal(before, module.task_adapters[0].up.weight)
            for before, module in zip(before_end, model.lora_modules.values())
        ))

        model.begin_task(TaskData(1, [bag(2), bag(3)]))
        self.assertTrue(all(int(module.active_task) == 1 for module in model.lora_modules.values()))
        self.assertTrue(all(
            all(not p.requires_grad for p in module.task_adapters[0].parameters())
            and all(p.requires_grad for p in module.task_adapters[1].parameters())
            for module in model.lora_modules.values()
        ))
        module_state = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        restored = build(options)
        restored.load_state_dict(module_state)
        restored.load_checkpoint_state(method_state)
        self.assertTrue(all(int(module.active_task) == 1 for module in restored.lora_modules.values()))
        self.assertTrue(all(
            all(not p.requires_grad for p in module.task_adapters[0].parameters())
            and all(p.requires_grad for p in module.task_adapters[1].parameters())
            for module in restored.lora_modules.values()
        ))

    def test_replay_is_one_to_one_and_loss_is_mean_not_sum(self):
        options = args(atlasv2_lora=True, atlasv2_replay=True, buffer_size=30)
        model = build(options)
        task0 = TaskData(0, [bag(0), bag(1)])
        model.begin_task(task0)
        for value in task0.train_loader:
            model.save_buffer(*value, task=0)
        model.end_task(task0)
        model.begin_task(TaskData(1, [bag(2)]))
        result = model.observe(*bag(2), task=1)
        self.assertEqual(result["replay_bags"], 1.0)
        self.assertEqual(result["loss"], result["loss_cls"])
        self.assertLessEqual(len(model.memory), 30)
        self.assertTrue(all(entry.features.shape[0] == entry.coords.shape[0] for entry in model.memory.entries))

    def _two_task_prototype_model(self, realign):
        options = args(
            atlasv2_lora=True, atlasv2_replay=True, atlasv2_prototype=True,
            atlasv2_realign=realign, buffer_size=30,
        )
        model = build(options)
        task0 = TaskData(0, [bag(0, shift=-1), bag(1, shift=1)])
        model.begin_task(task0)
        for value in task0.train_loader:
            model.save_buffer(*value, task=0)
        model.end_task(task0)
        old = model.net.prototype_bank[0].clone()
        task1 = TaskData(1, [bag(2, shift=-0.5), bag(3, shift=0.5)])
        model.begin_task(task1)
        with torch.no_grad():
            for module in model.lora_modules.values():
                module.lora_a.normal_()
                module.lora_b.normal_(std=0.4)
        for value in task1.train_loader:
            model.save_buffer(*value, task=1)
        model.end_task(task1)
        return model, old

    def test_stale_and_realigned_prototype_lifecycle(self):
        stale, old_stale = self._two_task_prototype_model(False)
        self.assertTrue(torch.equal(old_stale, stale.net.prototype_bank[0]))
        realigned, old_realigned = self._two_task_prototype_model(True)
        self.assertFalse(torch.allclose(old_realigned, realigned.net.prototype_bank[0]))
        self.assertTrue(realigned.net.prototype_valid.all())
        self.assertFalse(realigned.net.prototype_bank.requires_grad)

    def test_prototype_inference_does_not_blend_prompt_when_disabled(self):
        model = build(args(atlasv2_prototype=True, atlasv2_train_classifier=False))
        model.net.set_prototype(0, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.seen_class_count = 2
        embedding = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        logits = model.net.inference_logits(embedding, 2)
        self.assertEqual(int(logits.argmax(1)), 0)
        self.assertFalse(hasattr(model.net, "prompt_projector"))

    def test_prompt_and_nce_are_explicit_once_and_have_finite_gradients(self):
        base = dict(
            atlasv2_lora=True, atlasv2_replay=True, atlasv2_prototype=True,
            atlasv2_realign=True, atlasv2_prompt=True, buffer_size=30,
        )
        prompt = build(args(**base))
        prompt.begin_task(TaskData(0, [bag(0)]))
        result = prompt.observe(*bag(0), task=0)
        self.assertEqual(result["loss_atlas_nce"], 0.0)
        self.assertAlmostEqual(result["loss"], result["loss_cls"] + result["loss_prompt"], places=6)
        self.assertIsNotNone(prompt.net.prompt_projector.weight.grad)
        self.assertTrue(torch.isfinite(prompt.net.prompt_projector.weight.grad).all())

        nce = build(args(**base, atlasv2_nce=True))
        nce.begin_task(TaskData(0, [bag(0)]))
        result = nce.observe(*bag(0), task=0)
        self.assertGreater(result["loss_atlas_nce"], 0.0)
        self.assertAlmostEqual(
            result["loss"],
            result["loss_cls"] + result["loss_prompt"] + result["loss_atlas_nce"],
            places=5,
        )
        self.assertTrue(torch.isfinite(nce.net.prompt_projector.weight.grad).all())
        self.assertTrue(all(
            module.lora_b.grad is not None and torch.isfinite(module.lora_b.grad).all()
            for module in nce.lora_modules.values()
        ))
        # Audit the NCE path in isolation so its LoRA/projector gradients cannot
        # be attributed to linear CE or prompt CE.
        nce.opt.zero_grad(set_to_none=True)
        features, coords, patch_size, label = bag(0)
        embedding = nce.net.encode(features, coords, patch_size)
        isolated_nce = nce._semantic_loss(embedding, label, nce.nce_temperature)
        isolated_nce.backward()
        self.assertTrue(torch.isfinite(nce.net.prompt_projector.weight.grad).all())
        self.assertTrue(any(
            module.lora_b.grad is not None
            and float(module.lora_b.grad.abs().sum()) > 0.0
            for module in nce.lora_modules.values()
        ))

        saved_projector = nce.net.prompt_projector.weight.detach().clone()
        restored = build(args(**base, atlasv2_nce=True))
        restored.load_state_dict(copy.deepcopy(nce.state_dict()))
        restored.load_checkpoint_state(copy.deepcopy(nce.get_checkpoint_state()))
        self.assertTrue(torch.equal(restored.net.prompt_projector.weight, saved_projector))

    def test_checkpoint_round_trip_keeps_full_memory_prototypes_and_membership(self):
        options = args(
            atlasv2_lora=True, atlasv2_replay=True, atlasv2_prototype=True,
            atlasv2_realign=True, buffer_size=30,
        )
        model = build(options)
        task0 = TaskData(0, [bag(0, patches=9), bag(1, patches=11)])
        model.begin_task(task0)
        for value in task0.train_loader:
            model.save_buffer(*value, task=0)
        model.end_task(task0)
        module_state = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        restored = build(options)
        restored.load_state_dict(module_state)
        restored.load_checkpoint_state(method_state)
        self.assertEqual(restored.completed_tasks, 1)
        self.assertTrue(torch.equal(restored.net.prototype_bank, model.net.prototype_bank))
        self.assertEqual([e.features.shape[0] for e in restored.memory.entries], [9, 11])
        self.assertTrue(all(
            torch.equal(left.features, right.features)
            for left, right in zip(restored.memory.entries, model.memory.entries)
        ))


if __name__ == "__main__":
    unittest.main()
