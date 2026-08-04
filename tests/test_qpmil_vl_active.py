"""Unit tests for the active QPMIL-VL integration (no TITAN download)."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.qpmil_vl_prompts import (
    DEFAULT_TASK_ORDER,
    PROMPT_REGISTRY,
    prompt_schema_hash,
    resolve_class_prompts,
)
from models.qpmil_vl import (
    QPMILVLTitanCore,
    TitanPromptEncoder,
    build_model_from_components,
    get_parser,
)
from utils.training import checkpoint_payload


class FakeTokenizer:
    def __init__(self, context_length: int):
        self.context_length = int(context_length)

    def __call__(self, texts):
        output = torch.zeros(
            len(texts), self.context_length, dtype=torch.long
        )
        for row, text in enumerate(texts):
            token_count = min(
                max(2, len(text.split()) + 2), self.context_length - 1
            )
            output[row, :token_count] = torch.arange(1, token_count + 1)
        return output


class FakeTransformer(nn.Module):
    def get_cast_dtype(self):
        return torch.float32

    def forward(self, values, attn_mask=None):
        del attn_mask
        return values + values.mean(dim=1, keepdim=True)


class FakeTextEncoder(nn.Module):
    def __init__(self, width: int = 8, context_length: int = 12):
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
        self.text_projection = nn.Parameter(torch.eye(width))
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


def make_core(task_num_classes=(2, 2), pool_size=4, match_size=2):
    text_encoder = FakeTextEncoder()
    return QPMILVLTitanCore(
        text_encoder=text_encoder,
        class_features=torch.randn(sum(task_num_classes), 8),
        task_num_classes=task_num_classes,
        pool_size=pool_size,
        prompt_length=3,
        match_size=match_size,
        csm_logit_scale=10.0,
    )


def make_args():
    return SimpleNamespace(
        task_order=["brca", "nsclc"],
        task_num_classes=[2, 2],
        num_classes=4,
        backbone="titan",
        backbone_model_id="fake/TITAN",
        backbone_revision="fake-revision",
        backbone_max_patches=400,
        feature_dim=8,
        pool_size=4,
        prompt_length=3,
        match_size=2,
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


def make_wrapper():
    args = make_args()
    # A real pinned TITAN revision is deterministic across checkpoint reloads.
    # Make the injected fake tower deterministic in the same way.
    with torch.random.fork_rng():
        torch.manual_seed(101)
        text_encoder = FakeTextEncoder()
        model = build_model_from_components(
            args,
            F.cross_entropy,
            None,
            text_encoder,
            torch.randn(4, 8),
        )
    model.net.to(model.device)
    return model


class QpmilVlActiveTest(unittest.TestCase):
    def test_prompt_registry_covers_10_tasks_27_classes_and_reverse_order(self):
        forward = resolve_class_prompts(DEFAULT_TASK_ORDER)
        reverse_order = list(reversed(DEFAULT_TASK_ORDER))
        reverse = resolve_class_prompts(reverse_order)
        self.assertEqual(len(DEFAULT_TASK_ORDER), 10)
        self.assertEqual(len(forward), 27)
        self.assertEqual(len(reverse), 27)
        first_reverse_count = len(PROMPT_REGISTRY[reverse_order[0]])
        self.assertEqual(
            reverse[:first_reverse_count],
            [list(value) for value in PROMPT_REGISTRY[reverse_order[0]]],
        )
        self.assertNotEqual(
            prompt_schema_hash(DEFAULT_TASK_ORDER),
            prompt_schema_hash(reverse_order),
        )
        with self.assertRaisesRegex(ValueError, "registered"):
            resolve_class_prompts(["missing_task"])

    def test_parser_defaults_match_qpmil_profile(self):
        args = get_parser().parse_args(
            [
                "--dataset",
                "seq-wsi",
                "--exp_desc",
                "test",
                "--model",
                "qpmil_vl",
            ]
        )
        self.assertEqual(args.backbone, "titan")
        self.assertEqual(args.feature_dim, 768)
        self.assertEqual(args.pool_size, 20)
        self.assertEqual(args.prompt_length, 24)
        self.assertEqual(args.match_size, 5)
        self.assertEqual(args.bags_per_update, 16)
        self.assertEqual(args.backbone_max_patches, 400)
        self.assertEqual(args.pooling, "max")
        self.assertEqual(args.csm_logit_scale, 100.0)
        self.assertEqual(args.alpha, 0.5)
        self.assertEqual(args.matching_loss_weight, 0.5)
        self.assertEqual(args.class_similarity_loss_weight, 0.5)
        self.assertEqual(args.lr, 1.0e-5)
        self.assertEqual(args.optim_wd, 1.0e-4)

    def test_fake_text_tower_is_frozen_but_prompt_gradient_flows(self):
        torch.manual_seed(7)
        core = make_core()
        core.begin_task(0, [])
        bags = [torch.randn(1, 8), torch.randn(3, 8), torch.randn(7, 8)]
        output = core(bags, compute_aux_losses=True)
        self.assertEqual(output["logits"].shape, (3, 2))
        self.assertEqual(output["key_indices"].shape, (3, 2))
        loss = (
            F.cross_entropy(output["logits"], torch.tensor([0, 1, 0]))
            + output["matching_loss"]
            + output["class_similarity_loss"]
        )
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in core.keys))
        self.assertTrue(any(parameter.grad is not None for parameter in core.prompts))
        self.assertIsNotNone(core.tunable_vectors[0].grad)
        self.assertFalse(core.tunable_vectors[1].requires_grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in core.prompt_encoder.text_encoder.parameters()
            )
        )
        self.assertFalse(core.prompt_encoder.text_encoder.training)

    def test_core_handles_one_patch_and_rejects_invalid_bags(self):
        core = make_core()
        core.begin_task(0, [])
        output = core([torch.zeros(1, 8)], compute_aux_losses=True)
        for key in ("logits", "matching_loss", "class_similarity_loss"):
            self.assertTrue(torch.isfinite(output[key]).all())
        with self.assertRaisesRegex(ValueError, "at least one"):
            core([])
        with self.assertRaisesRegex(ValueError, "empty"):
            core([torch.empty(0, 8)])
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            core([torch.randn(2, 7)])
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            core([torch.full((2, 8), float("nan"))])

    def test_prompt_encoder_contract_validation(self):
        with self.assertRaisesRegex(ValueError, "prompt_length"):
            TitanPromptEncoder(FakeTextEncoder(context_length=5), prompt_length=4)
        broken = FakeTextEncoder()
        del broken.tokenizer
        with self.assertRaisesRegex(ValueError, "missing attributes"):
            TitanPromptEncoder(broken, prompt_length=3)

    def test_wrapper_observe_many_padding_key_lifecycle_and_checkpoint_round_trip(self):
        torch.manual_seed(11)
        model = make_wrapper()
        dataset = SimpleNamespace(current_task=1)
        model.begin_task(dataset)
        model.begin_epoch(0, 3)
        keys_before = [value.detach().clone() for value in model.net.keys]
        prompts_before = [value.detach().clone() for value in model.net.prompts]
        batches = [
            (
                torch.randn(patches, 8),
                torch.zeros(patches, 2, dtype=torch.long),
                torch.tensor(256),
                torch.tensor([label]),
            )
            for patches, label in ((1, 0), (4, 1), (2, 0))
        ]
        losses = model.observe_many(batches, task=0)
        self.assertEqual(
            set(losses),
            {
                "loss",
                "classification_loss",
                "matching_loss",
                "class_similarity_loss",
            },
        )
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        )
        epoch_state = model.end_epoch(0, 3)
        self.assertEqual(
            epoch_state["key_matches"], len(batches) * model.net.match_size
        )
        unmatched = (model.active_key_frequency == 0).nonzero().flatten().tolist()
        self.assertTrue(unmatched)
        for index in unmatched:
            torch.testing.assert_close(model.net.keys[index], keys_before[index])
            torch.testing.assert_close(model.net.prompts[index], prompts_before[index])

        logits = model([batches[0][0], batches[0][1], batches[0][2]])[0]
        self.assertEqual(logits.shape, (1, 4))
        self.assertTrue(torch.isfinite(logits[:, :2]).all())
        self.assertTrue(torch.isneginf(logits[:, 2:]).all())

        training_state = model.get_checkpoint_state()
        self.assertEqual(training_state["state_type"], "adaptation_only")
        self.assertNotIn("text_encoder", str(training_state.keys()))
        self.assertEqual(training_state["completed_key_frequencies"], [])
        framework_payload = checkpoint_payload(
            model,
            SimpleNamespace(metadata=lambda fold: {"fold": fold}),
            fold=2,
        )
        self.assertIn("method_state", framework_payload)
        self.assertNotIn("state_dict", framework_payload)
        self.assertNotIn("optimizer_state", framework_payload)

        # Simulate a worse later epoch, then restore the early-stopping winner.
        best_frequency = training_state["active_key_frequency"].clone()
        model.begin_epoch(0, 4)
        model.observe_many([batches[0]], task=0)
        self.assertFalse(torch.equal(model.active_key_frequency, best_frequency))
        model.load_checkpoint_state(training_state)
        torch.testing.assert_close(model.active_key_frequency, best_frequency)
        model.end_task(dataset)
        finalized_state = model.get_checkpoint_state()
        self.assertEqual(len(finalized_state["completed_key_frequencies"]), 1)
        torch.testing.assert_close(
            finalized_state["completed_key_frequencies"][0], best_frequency
        )

        restored = make_wrapper()
        restored.load_checkpoint_state(finalized_state)
        actual = restored([batches[0][0], batches[0][1], batches[0][2]])[0]
        torch.testing.assert_close(actual, logits)
        self.assertTrue(restored._task_finalized)

        broken = copy.deepcopy(finalized_state)
        broken["titan_revision"] = "wrong-revision"
        with self.assertRaisesRegex(ValueError, "titan_revision"):
            make_wrapper().load_checkpoint_state(broken)
        broken = copy.deepcopy(finalized_state)
        broken["prompt_schema_hash"] = "wrong-hash"
        with self.assertRaisesRegex(ValueError, "prompt_schema_hash"):
            make_wrapper().load_checkpoint_state(broken)

    def test_second_task_uses_prior_frequency_and_only_current_vector_trains(self):
        model = make_wrapper()
        dataset = SimpleNamespace(current_task=1)
        model.begin_task(dataset)
        model.begin_epoch(0, 0)
        model.observe_many(
            [
                (
                    torch.randn(2, 8),
                    torch.zeros(2, 2, dtype=torch.long),
                    torch.tensor(256),
                    torch.tensor([0]),
                )
            ],
            task=0,
        )
        model.end_epoch(0, 0)
        model.end_task(dataset)

        dataset.current_task = 2
        model.begin_task(dataset)
        self.assertIsNotNone(model.net.penalty_table)
        self.assertFalse(model.net.tunable_vectors[0].requires_grad)
        self.assertTrue(model.net.tunable_vectors[1].requires_grad)
        model.begin_epoch(1, 0)
        model.observe_many(
            [
                (
                    torch.randn(3, 8),
                    torch.zeros(3, 2, dtype=torch.long),
                    torch.tensor(256),
                    torch.tensor([2]),
                )
            ],
            task=1,
        )
        self.assertIsNone(model.net.tunable_vectors[0].grad)
        self.assertIsNotNone(model.net.tunable_vectors[1].grad)


if __name__ == "__main__":
    unittest.main()
