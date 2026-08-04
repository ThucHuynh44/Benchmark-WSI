"""Acceptance-level synthetic streams for the newly integrated WSI methods.

The fakes exercise the public TITAN/FEATHER contracts without loading remote
weights.  Every case advances through two variable-class tasks and keeps the
bags variable-length, while remaining small enough for CPU-only CI.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lwsr import Lwsr
from models.micil import Micil
from models.qpmil_vl import build_model_from_components
from utils.training import _iter_logical_batches


CPU = torch.device("cpu")


class FakeSlideBackbone(nn.Module):
    """Small trainable implementation of the shared slide-backbone API."""

    supports_ssl = False

    def __init__(self, contract: str, num_classes: int = 5):
        super().__init__()
        if contract not in {"titan", "feather"}:
            raise ValueError(f"Unknown fake backbone contract: {contract}")
        self.contract = contract
        self.encoder = nn.Linear(768, 8)
        self.attention_head = nn.Linear(768, 1)
        self.classifier = nn.Linear(8, num_classes)
        self.forward_calls = 0

    def forward_with_embedding(self, features, coords, patch_size_level0):
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError("fake TITAN/FEATHER expects [N,768] features")
        if coords.shape != (features.shape[0], 2):
            raise ValueError("fake TITAN/FEATHER coordinates do not match the bag")
        if int(torch.as_tensor(patch_size_level0).item()) <= 0:
            raise ValueError("patch size must be positive")
        self.forward_calls += 1

        if self.contract == "titan":
            attention = torch.full(
                (1, features.shape[0]),
                1.0 / features.shape[0],
                device=features.device,
                dtype=features.dtype,
            )
        else:
            attention = torch.softmax(
                self.attention_head(features).transpose(0, 1), dim=1
            )
        pooled = attention @ features
        embedding = torch.tanh(self.encoder(pooled))
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
            logits.softmax(dim=1),
            logits.argmax(dim=1),
            output["attention"],
            output["auxiliary_loss"],
        )

    def get_classifier(self):
        return self.classifier


class FakeTaskDataset:
    def __init__(self, task_id: int, labels):
        self.current_task = int(task_id) + 1
        self.train_loader = SimpleNamespace(
            dataset=SimpleNamespace(targets=list(labels))
        )


def method_args(backbone: str, *, replay: bool = False):
    return SimpleNamespace(
        lr=1.0e-3,
        backbone=backbone,
        feature_dim=768,
        backbone_freeze=False,
        backbone_max_patches=0,
        num_classes=5,
        seed=23,
        buffer_size=5,
        minibatch_size=2,
        bags_per_update=2,
        buffer_max_patches=4,
        pair_loss_weight=1.0,
        ce_loss_weight=1.0,
        dc_loss_weight=0.01,
        micil_replay=replay,
        micil_weight_norm=True,
        kd_loss_weight=10.0,
        embedding_loss_weight=1.0,
        distillation_temperature=2.0,
        class_offsets=[0, 2],
        task_num_classes=[2, 3],
    )


def make_bag(label: int, patch_count: int, value: float):
    patch_offsets = torch.linspace(-0.15, 0.15, patch_count).unsqueeze(1)
    channel_offsets = torch.linspace(-0.05, 0.05, 768).unsqueeze(0)
    features = torch.full((patch_count, 768), float(value))
    features = features + patch_offsets + channel_offsets
    coords = torch.arange(patch_count * 2, dtype=torch.long).reshape(
        patch_count, 2
    )
    return (
        features,
        coords,
        torch.tensor(1024, dtype=torch.long),
        torch.tensor([label], dtype=torch.long),
    )


def build_slide_method(method, args):
    with torch.random.fork_rng(), patch(
        "models.utils.continual_model.get_device", return_value=CPU
    ):
        torch.manual_seed(101)
        model = method(
            FakeSlideBackbone(args.backbone, args.num_classes),
            F.cross_entropy,
            args,
            None,
        )
    return model.to(CPU)


def assert_finite_metrics(testcase: unittest.TestCase, metrics):
    testcase.assertIn("loss", metrics)
    testcase.assertTrue(
        all(torch.isfinite(torch.tensor(float(value))) for value in metrics.values())
    )


def assert_same_network(testcase: unittest.TestCase, left, right):
    testcase.assertEqual(set(left.net.state_dict()), set(right.net.state_dict()))
    for name, value in left.net.state_dict().items():
        torch.testing.assert_close(
            value,
            right.net.state_dict()[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )


class LwsrMicilSyntheticMatrixTest(unittest.TestCase):
    def test_lwsr_two_task_titan_feather_matrix(self):
        for backbone in ("titan", "feather"):
            with self.subTest(backbone=backbone):
                model = build_slide_method(Lwsr, method_args(backbone))
                task_zero = [
                    make_bag(0, 3, -0.75),
                    make_bag(1, 5, -0.25),
                ]
                model.begin_task(FakeTaskDataset(0, [0, 1]))
                first = model.observe_many(task_zero, task=0)
                assert_finite_metrics(self, first)
                self.assertEqual(first["replay_bags"], 0.0)
                for bag in task_zero:
                    model.save_buffer(*bag, task=0)
                model.end_task()
                self.assertEqual(tuple(model.previous_dist_matrix.shape), (2, 2))

                task_one = [
                    make_bag(2, 2, 0.25),
                    make_bag(3, 4, 0.75),
                    make_bag(4, 3, 0.50),
                ]
                model.begin_task(FakeTaskDataset(1, [2, 3, 4]))
                second = model.observe_many(task_one, task=1)
                assert_finite_metrics(self, second)
                self.assertEqual(second["replay_bags"], 2.0)
                for bag in task_one:
                    model.save_buffer(*bag, task=1)
                model.end_task()

                self.assertEqual(model.current_task, 1)
                self.assertEqual(model.buffer.num_seen_examples, 5)
                self.assertEqual(tuple(model.previous_dist_matrix.shape), (5, 5))
                self.assertTrue(
                    all(item.features.device.type == "cpu" for item in model.buffer.all(CPU))
                )
                logits = model([task_one[0][0], task_one[0][1], task_one[0][2]])[0]
                self.assertEqual(tuple(logits.shape), (1, 5))
                self.assertTrue(torch.isfinite(logits).all())

    def test_micil_replay_isolated_by_flag_on_titan_and_feather(self):
        for backbone in ("titan", "feather"):
            with self.subTest(backbone=backbone):
                disabled = build_slide_method(
                    Micil, method_args(backbone, replay=False)
                )
                disabled_probe = build_slide_method(
                    Micil, method_args(backbone, replay=False)
                )
                enabled = build_slide_method(
                    Micil, method_args(backbone, replay=True)
                )
                models = (disabled, disabled_probe, enabled)

                task_zero = [
                    make_bag(0, 3, -0.75),
                    make_bag(1, 5, -0.25),
                ]
                for model in models:
                    model.begin_task(FakeTaskDataset(0, [0, 1]))
                    metrics = model.observe_many(task_zero, task=0)
                    assert_finite_metrics(self, metrics)
                    self.assertEqual(metrics["replay_bags"], 0.0)
                assert_same_network(self, disabled, disabled_probe)
                assert_same_network(self, disabled, enabled)

                # Attempting to populate a disabled MICIL buffer is a strict
                # no-op.  Only the enabled model retains these old-task bags.
                for bag in task_zero:
                    self.assertEqual(disabled_probe.save_buffer(*bag, task=0), -1)
                    enabled.save_buffer(*bag, task=0)
                self.assertFalse(hasattr(disabled, "buffer"))
                self.assertFalse(hasattr(disabled_probe, "buffer"))
                self.assertEqual(enabled.buffer.labels, (0, 1))
                for model in models:
                    model.end_task()
                    self.assertFalse(model.teacher.training)
                    self.assertFalse(
                        any(parameter.requires_grad for parameter in model.teacher.parameters())
                    )

                task_one = [
                    make_bag(2, 2, 0.25),
                    make_bag(3, 4, 0.75),
                    make_bag(4, 3, 0.50),
                ]
                for model in models:
                    model.begin_task(FakeTaskDataset(1, [2, 3, 4]))
                calls_before = [model.net.forward_calls for model in models]
                disabled_metrics = disabled.observe_many(task_one, task=1)
                probe_metrics = disabled_probe.observe_many(task_one, task=1)
                enabled_metrics = enabled.observe_many(task_one, task=1)
                for metrics in (disabled_metrics, probe_metrics, enabled_metrics):
                    assert_finite_metrics(self, metrics)

                self.assertEqual(disabled_metrics["replay_bags"], 0.0)
                self.assertEqual(probe_metrics["replay_bags"], 0.0)
                self.assertEqual(enabled_metrics["replay_bags"], 2.0)
                self.assertEqual(
                    disabled.net.forward_calls - calls_before[0], len(task_one)
                )
                self.assertEqual(
                    disabled_probe.net.forward_calls - calls_before[1], len(task_one)
                )
                self.assertEqual(
                    enabled.net.forward_calls - calls_before[2], len(task_one) + 2
                )
                self.assertAlmostEqual(
                    disabled_metrics["loss"], probe_metrics["loss"], places=7
                )
                assert_same_network(self, disabled, disabled_probe)
                self.assertTrue(
                    any(
                        not torch.allclose(value, enabled.net.state_dict()[name])
                        for name, value in disabled.net.state_dict().items()
                    )
                )

                for bag in task_one:
                    enabled.save_buffer(*bag, task=1)
                for model in models:
                    model.end_task()
                self.assertEqual(enabled.buffer.num_seen_examples, 5)


class FakeTokenizer:
    def __init__(self, context_length: int):
        self.context_length = int(context_length)

    def __call__(self, texts):
        tokens = torch.zeros(
            len(texts), self.context_length, dtype=torch.long
        )
        for row, text in enumerate(texts):
            token_count = min(
                max(2, len(text.split()) + 2), self.context_length - 1
            )
            tokens[row, :token_count] = torch.arange(1, token_count + 1)
        return tokens


class FakeTransformer(nn.Module):
    def get_cast_dtype(self):
        return torch.float32

    def forward(self, values, attn_mask=None):
        del attn_mask
        return values + values.mean(dim=1, keepdim=True)


class FakeTitanTextEncoder(nn.Module):
    """Frozen fake TITAN tower with a real 768-D output feature space."""

    def __init__(self, width: int = 16, output_dim: int = 768, context_length: int = 8):
        super().__init__()
        self.heads = 1
        self.pad_id = 0
        self.token_embedding = nn.Embedding(64, width)
        self.positional_embedding = nn.Parameter(
            torch.randn(context_length, width)
        )
        self.transformer = FakeTransformer()
        self.ln_final = nn.LayerNorm(width)
        self.cls_emb = nn.Parameter(torch.randn(width))
        self.text_projection = nn.Linear(width, output_dim, bias=False)
        self.tokenizer = FakeTokenizer(context_length)
        mask = torch.full(
            (context_length, context_length), float("-inf")
        ).triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)

    def build_cls_mask(self, text, cast_dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(
            cls_mask, (0, 1, cls_mask.shape[2], 0), value=True
        )
        additive = torch.zeros(
            cls_mask.shape, dtype=cast_dtype, device=cls_mask.device
        )
        additive.masked_fill_(~cls_mask, float("-inf"))
        return torch.repeat_interleave(additive, self.heads, dim=0)


def qpmil_args():
    return SimpleNamespace(
        task_order=["brca", "rcc"],
        task_num_classes=[2, 3],
        num_classes=5,
        backbone="titan",
        backbone_model_id="fake/TITAN",
        backbone_revision="synthetic-revision",
        backbone_max_patches=400,
        feature_dim=768,
        pool_size=4,
        prompt_length=2,
        match_size=2,
        bags_per_update=2,
        pooling="max",
        csm_logit_scale=10.0,
        classification_logit_scale=1.0,
        alpha=0.5,
        matching_loss_weight=0.5,
        class_similarity_loss_weight=0.5,
        max_grad_norm=1.0,
        adam_eps=1.0e-8,
        lr=1.0e-3,
        optim_wd=1.0e-4,
    )


def build_qpmil():
    args = qpmil_args()
    with torch.random.fork_rng(), patch(
        "models.utils.continual_model.get_device", return_value=CPU
    ):
        torch.manual_seed(211)
        text_encoder = FakeTitanTextEncoder()
        class_features = torch.randn(args.num_classes, args.feature_dim)
        model = build_model_from_components(
            args,
            F.cross_entropy,
            None,
            text_encoder,
            class_features,
        )
    return model.to(CPU), args


class QpmilVlSyntheticMatrixTest(unittest.TestCase):
    def test_qpmil_fake_titan_two_task_stream_and_remainder(self):
        model, args = build_qpmil()
        task_zero = [
            make_bag(0, 1, -0.75),
            make_bag(1, 3, -0.25),
            make_bag(0, 2, -0.50),
        ]
        model.begin_task(FakeTaskDataset(0, [0, 1, 0]))
        model.begin_epoch(0, 0)
        groups = list(_iter_logical_batches(task_zero, args.bags_per_update))
        self.assertEqual([len(group) for group in groups], [2, 1])
        for group in groups:
            assert_finite_metrics(self, model.observe_many(group, task=0))
        epoch_zero = model.end_epoch(0, 0)
        self.assertEqual(
            epoch_zero["key_matches"], len(task_zero) * args.match_size
        )
        first_logits = model(
            [task_zero[0][0], task_zero[0][1], task_zero[0][2]]
        )[0]
        self.assertTrue(torch.isfinite(first_logits[:, :2]).all())
        self.assertTrue(torch.isneginf(first_logits[:, 2:]).all())
        model.end_task(FakeTaskDataset(0, [0, 1, 0]))

        task_one = [
            make_bag(2, 4, 0.25),
            make_bag(3, 2, 0.75),
            make_bag(4, 3, 0.50),
        ]
        model.begin_task(FakeTaskDataset(1, [2, 3, 4]))
        self.assertIsNotNone(model.net.penalty_table)
        self.assertFalse(model.net.tunable_vectors[0].requires_grad)
        self.assertTrue(model.net.tunable_vectors[1].requires_grad)
        model.begin_epoch(1, 0)
        groups = list(_iter_logical_batches(task_one, args.bags_per_update))
        self.assertEqual([len(group) for group in groups], [2, 1])
        for group in groups:
            assert_finite_metrics(self, model.observe_many(group, task=1))
        epoch_one = model.end_epoch(1, 0)
        self.assertEqual(
            epoch_one["key_matches"], len(task_one) * args.match_size
        )
        second_logits = model(
            [task_one[0][0], task_one[0][1], task_one[0][2]]
        )[0]
        self.assertEqual(tuple(second_logits.shape), (1, 5))
        self.assertTrue(torch.isfinite(second_logits).all())
        model.end_task(FakeTaskDataset(1, [2, 3, 4]))

        self.assertEqual(model.current_task, 1)
        self.assertEqual(len(model.completed_key_frequencies), 2)
        self.assertFalse(model.net.prompt_encoder.text_encoder.training)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.net.prompt_encoder.text_encoder.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
