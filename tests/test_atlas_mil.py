import copy
import math
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.atlas_mil import build_model_from_components, validate_args
from models.utils.atlas_lora import AtlasLoRALinear
from models.utils.atlas_memory import AtlasMemoryPool, AtlasReplayBag


Batch = namedtuple("Batch", "features coords patch_size_level0 labels")


class TinyAttentionBackbone(nn.Module):
    supports_ssl = False
    has_genuine_patch_attention = True

    def __init__(self, num_classes=4):
        super().__init__()
        self.attention_encoder = nn.Linear(768, 4)
        self.attention_score = nn.Linear(4, 1)
        self.encoder = nn.Linear(768, 4)
        self.classifier = nn.Linear(4, num_classes)

    def get_classifier(self):
        return self.classifier

    def forward_with_embedding(self, features, coords, patch_size_level0):
        if coords.shape != (features.shape[0], 2) or int(patch_size_level0) <= 0:
            raise ValueError("invalid WSI metadata")
        attention = torch.softmax(
            self.attention_score(torch.tanh(self.attention_encoder(features))).t(),
            dim=1,
        )
        encoded = torch.tanh(self.encoder(features))
        embedding = attention @ encoded
        logits = self.classifier(embedding)
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
            logits.softmax(1),
            logits.argmax(1),
            output["attention"],
            output["auxiliary_loss"],
        )


def atlas_args(**overrides):
    values = dict(
        optimizer="adamw",
        lr=1.0e-3,
        optim_wd=0.0,
        adam_eps=1.0e-8,
        backbone="generic_mil",
        feature_dim=768,
        backbone_freeze=False,
        backbone_max_patches=0,
        num_classes=4,
        n_tasks=2,
        task_num_classes=[2, 2],
        class_offsets=[0, 2],
        task_order=["brca", "nsclc"],
        seed=9,
        buffer_size=4,
        minibatch_size=1,
        bags_per_update=1,
        pmp_k=4,
        atlas_rank=2,
        atlas_nce_temperature=0.1,
        atlas_centroid_momentum=0.9,
        latent_mask_ratio=0.5,
        atlas_nce_weight=1.0,
        reconstruction_weight=1.0,
        manifold_weight=1.0,
        attention_weight=1.0,
        atlas_prompt_weight=0.5,
        atlas_logit_scale=5.0,
        atlas_lora_rank=2,
        atlas_lora_mode="semantic",
        atlas_lora_merge_scale=1.0,
        atlas_text_model_id="fake-titan",
        atlas_text_revision="fixed-revision",
        atlas_replay=True,
        atlas_diagnostics=False,
        ablation_id=None,
        ablation_group=None,
        ablation_config_hash=None,
        backbone_model_id=None,
        backbone_revision=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_bag(label, value=0.0, patches=6):
    base = torch.linspace(-0.1, 0.1, patches).unsqueeze(1)
    features = torch.full((patches, 768), float(value)) + base
    coords = torch.arange(patches * 2, dtype=torch.long).reshape(patches, 2)
    return features, coords, torch.tensor(1024), torch.tensor([label])


class BagDataset(Dataset):
    def __init__(self, bags):
        self.bags = list(bags)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        features, coords, patch_size, label = self.bags[index]
        return features, coords, patch_size, int(label.item())


def collate(batch):
    features, coords, patch_size, label = batch[0]
    return Batch(features, coords, torch.as_tensor(patch_size), torch.tensor([label]))


class TaskDataset:
    def __init__(self, task, bags):
        self.current_task = int(task) + 1
        self.train_loader = DataLoader(
            BagDataset(bags), batch_size=1, shuffle=False, collate_fn=collate
        )


def build(args=None):
    args = args or atlas_args()
    torch.manual_seed(4)
    backbone = TinyAttentionBackbone(args.num_classes)
    anchors = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    with patch("models.utils.continual_model.get_device", return_value=torch.device("cpu")):
        return build_model_from_components(
            args, F.cross_entropy, None, backbone, anchors
        ).to("cpu")


class AtlasLoRATests(unittest.TestCase):
    def test_factor_merge_matches_dense_truncated_svd_and_keeps_size(self):
        torch.manual_seed(2)
        layer = AtlasLoRALinear(5, 4, rank=2, bias=False)
        with torch.no_grad():
            layer.merged_up.normal_()
            layer.merged_down.normal_()
            layer.active_up.normal_()
            layer.active_down.normal_()
        dense = layer.merged_delta() + layer.active_delta()
        u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
        expected = (u[:, :2] * singular[:2]) @ vh[:2]
        count = sum(parameter.numel() for parameter in layer.parameters())
        layer.merge_active(rho=1.0)
        self.assertTrue(torch.allclose(layer.merged_delta(), expected, atol=1e-4))
        self.assertEqual(count, sum(parameter.numel() for parameter in layer.parameters()))
        self.assertTrue(torch.count_nonzero(layer.active_up) == 0)

    def test_hard_projection_removes_old_left_subspace(self):
        layer = AtlasLoRALinear(4, 4, rank=1, bias=False)
        with torch.no_grad():
            layer.merged_up[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            layer.merged_down[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            layer.active_up[:, 0] = torch.tensor([2.0, 1.0, 0.0, 0.0])
            layer.active_down[0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        layer.merge_active(rho=0.0)
        self.assertLess(abs(float(layer.merged_delta()[0, 1])), 1.0e-5)


class AtlasMemoryTests(unittest.TestCase):
    def test_balanced_targets_and_checkpoint_round_trip(self):
        pool = AtlasMemoryPool(
            4, pmp_k=3, feature_dim=768, embedding_dim=4,
            num_classes=4, seed=5,
        )
        pool.start_update(2, task_id=0)
        for label in (0, 0, 0, 1, 1, 1):
            decision = pool.consider(label)
            if decision is None:
                continue
            bag = make_bag(label, value=label, patches=3)
            pool.commit(decision, AtlasReplayBag(
                features=bag[0], coords=bag[1], patch_size=bag[2],
                label=bag[3], origin_task_id=0,
            ))
        entries = pool.all("cpu", require_targets=False)
        targets = [
            (
                torch.full((1, entry.features.shape[0]), 1.0 / entry.features.shape[0]),
                torch.full((1, 4), float(entry.label.item())),
            )
            for entry in entries
        ]
        pool.refresh_targets(targets, target_snapshot_task=0)
        self.assertEqual(pool.label_counts().tolist(), [2, 2, 0, 0])
        state = copy.deepcopy(pool.state_dict())
        restored = AtlasMemoryPool(
            4, pmp_k=3, feature_dim=768, embedding_dim=4,
            num_classes=4, seed=100,
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.labels, pool.labels)
        self.assertEqual(restored.target_snapshot_task, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for map-location regression")
    def test_cuda_mapped_bookkeeping_is_normalized_to_cpu(self):
        pool = AtlasMemoryPool(
            4, pmp_k=3, feature_dim=768, embedding_dim=4,
            num_classes=4, seed=5,
        )
        state = pool.state_dict()
        for key in ("seen_count", "quotas", "class_priority"):
            state[key] = state[key].cuda()
        restored = AtlasMemoryPool(
            4, pmp_k=3, feature_dim=768, embedding_dim=4,
            num_classes=4, seed=8,
        )
        restored.load_state_dict(state)
        self.assertEqual(restored._quotas.device.type, "cpu")
        self.assertEqual(restored._seen_count.device.type, "cpu")


class AtlasMethodTests(unittest.TestCase):
    def test_validation_rejects_unsupported_or_invalid_settings(self):
        for override in (
            {"backbone": "titan"},
            {"feature_dim": 512},
            {"backbone_max_patches": 4},
            {"buffer_size": 3},
            {"atlas_lora_rank": 0},
            {"atlas_nce_weight": -1.0},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_args(atlas_args(**override))
        with self.assertRaises(ValueError):
            validate_args(atlas_args(
                atlas_replay=False, attention_weight=1.0, manifold_weight=0.0
            ))

    def test_hybrid_classifier_fallback_and_single_sample_pca(self):
        model = build()
        embedding = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        prompt_only = model.net.logit_scale * (
            F.normalize(embedding, dim=1) @ model.net.projected_prompts().t()
        )
        self.assertTrue(torch.allclose(model.net.logits_from_embedding(embedding), prompt_only))
        model.net.finalize_class(0, embedding)
        self.assertTrue(model.net.atlas_valid[0])
        self.assertEqual(int(model.net.atlas_effective_ranks[0]), 0)
        self.assertTrue(torch.isfinite(model.net.logits_from_embedding(embedding)).all())

    def test_two_task_replay_boundary_and_checkpoint_round_trip(self):
        model = build()
        task0_bags = [make_bag(0, 0.2), make_bag(1, 0.5)]
        dataset0 = TaskDataset(0, task0_bags)
        model.begin_task(dataset0)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        for bag in task0_bags:
            result = model.observe(*bag, task=0)
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertIsNotNone(model.net.prompt_projector.weight.grad)
        self.assertIsNotNone(model.net.decoder[0].weight.grad)
        self.assertTrue(any(
            module.active_up.grad is not None
            for module in model.lora_modules.values()
        ))
        self.assertFalse(model.net.backbone.get_classifier().weight.requires_grad)
        for bag in task0_bags:
            model.save_buffer(*bag, task=0)
        model.end_task(dataset0)
        self.assertEqual(
            parameter_count,
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertEqual(model.completed_tasks, 1)
        self.assertEqual(model.memory.target_snapshot_task, 0)
        self.assertEqual(int(model.net.atlas_finalized.sum()), 2)
        self.assertTrue(all(
            entry.target_embedding is not None and entry.target_attention is not None
            for entry in model.memory.all("cpu")
        ))

        state_dict = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        restored = build()
        restored.load_state_dict(state_dict)
        restored.load_checkpoint_state(method_state)
        probe = make_bag(0, 0.3)
        with torch.no_grad():
            left = model([probe[0], probe[1], probe[2]])[0]
            right = restored([probe[0], probe[1], probe[2]])[0]
        self.assertTrue(torch.allclose(left, right))

        task1_bags = [make_bag(2, 0.8), make_bag(3, 1.1)]
        dataset1 = TaskDataset(1, task1_bags)
        model.begin_task(dataset1)
        result = model.observe(*task1_bags[0], task=1)
        self.assertEqual(result["replay_bags"], 1.0)
        self.assertGreaterEqual(result["loss_attention"], 0.0)

    def test_zero_weight_branches_are_not_called_or_advance_mask_rng(self):
        model = build(atlas_args(
            atlas_nce_weight=0.0,
            reconstruction_weight=0.0,
            manifold_weight=0.0,
            attention_weight=0.0,
        ))
        dataset = TaskDataset(0, [make_bag(0, 0.2), make_bag(1, 0.5)])
        model.begin_task(dataset)
        before = model.mask_generator.get_state().clone()
        with patch.object(model, "_atlas_nce_loss", side_effect=AssertionError), patch.object(
            model, "_masked_reconstruction", side_effect=AssertionError
        ):
            result = model.observe(*make_bag(0, 0.2), task=0)
        self.assertEqual(result["loss_atlas_nce"], 0.0)
        self.assertEqual(result["loss_reconstruction"], 0.0)
        self.assertTrue(torch.equal(before, model.mask_generator.get_state()))

    def test_no_replay_two_task_diagnostics_and_checkpoint(self):
        args = atlas_args(
            atlas_replay=False,
            atlas_diagnostics=True,
            attention_weight=0.0,
            manifold_weight=0.0,
            ablation_id="wo_replay",
            ablation_group="leave_one_out",
            ablation_config_hash="fixed-hash",
        )
        model = build(args)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        task0 = TaskDataset(0, [make_bag(0, 0.2), make_bag(1, 0.5)])
        model.begin_task(task0)
        model.observe(*make_bag(0, 0.2), task=0)
        model.end_task(task0)
        self.assertIsNone(model.memory)
        self.assertEqual(model.completed_tasks, 1)
        self.assertEqual(len(model.diagnostic_history), 1)
        self.assertTrue(math.isnan(model.diagnostic_history[0]["old_current_overlap"]))
        self.assertEqual(model.diagnostic_history[0]["old_current_pair_count"], 0)

        state_dict = copy.deepcopy(model.state_dict())
        method_state = copy.deepcopy(model.get_checkpoint_state())
        restored = build(args)
        restored.load_state_dict(state_dict)
        restored.load_checkpoint_state(method_state)
        self.assertEqual(restored.diagnostic_history, model.diagnostic_history)

        task1 = TaskDataset(1, [make_bag(2, 0.8), make_bag(3, 1.1)])
        model.begin_task(task1)
        result = model.observe(*make_bag(2, 0.8), task=1)
        self.assertEqual(result["replay_bags"], 0.0)
        model.end_task(task1)
        self.assertEqual(model.completed_tasks, 2)
        self.assertEqual(len(model.diagnostic_history), 2)
        self.assertEqual(
            parameter_count, sum(parameter.numel() for parameter in model.parameters())
        )

    def test_subspace_overlap_and_pair_aggregation(self):
        model = build()
        identity = torch.eye(4)
        self.assertAlmostEqual(model._subspace_overlap(identity, 2, identity, 2), 1.0)
        left = identity[:, :1]
        right = identity[:, 1:2]
        self.assertAlmostEqual(model._subspace_overlap(left, 1, right, 1), 0.0)
        self.assertTrue(math.isnan(model._subspace_overlap(left, 0, right, 1)))
        with torch.no_grad():
            model.net.atlas_subspaces[0, :, :2] = identity[:, :2]
            model.net.atlas_subspaces[1, :, :2] = identity[:, :2]
            model.net.atlas_subspaces[2, :, :2] = identity[:, 2:]
            model.net.atlas_effective_ranks[:3] = torch.tensor([2, 2, 2])
        value, count = model._pair_overlap([(0, 1), (0, 2)])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(value, 0.5)


if __name__ == "__main__":
    unittest.main()
