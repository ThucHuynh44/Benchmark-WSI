import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backbone.generic_mil import GenericMILBackbone
from models.agem import AGem
from models.ewc_on import EwcOn
from models.sgd import Sgd
from utils.precision import PrecisionPolicy, resolve_precision_name, validate_precision_configuration
from utils.training import checkpoint_payload


def _args(**overrides):
    values = {
        "lr": 1.0e-2,
        "optimizer": "adamw",
        "optim_wd": 0.0,
        "adam_eps": 1.0e-8,
        "backbone": "titan",
        "backbone_max_patches": 0,
        "precision": "auto",
        "buffer_size": 2,
        "e_lambda": 0.1,
        "gamma": 0.9,
        "batch_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeScaler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.scale_value = 8.0
        self.backward_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def is_enabled(self):
        return self.enabled

    def scale(self, loss):
        scaler = self

        class ScaledLoss:
            def backward(self, **kwargs):
                scaler.backward_calls += 1
                (loss * scaler.scale_value).backward(**kwargs)

        return ScaledLoss()

    def step(self, optimizer):
        self.step_calls += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
        optimizer.step()

    def update(self):
        self.update_calls += 1

    def get_scale(self):
        return self.scale_value

    def state_dict(self):
        return {"scale": self.scale_value}

    def load_state_dict(self, state):
        self.scale_value = float(state["scale"])


class _FakeAmpPolicy:
    name = "fp16"
    enabled = True

    def __init__(self):
        self.scale_value = 8.0
        self.backward_calls = 0
        self.step_calls = 0
        self.gradient_scale_calls = 0

    def autocast(self):
        return nullcontext()

    def backward(self, loss, **kwargs):
        self.backward_calls += 1
        (loss * self.scale_value).backward(**kwargs)

    def step(self, optimizer):
        self.step_calls += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)
        optimizer.step()

    def gradient_scale(self):
        self.gradient_scale_calls += 1
        return self.scale_value


class PrecisionPolicyTests(unittest.TestCase):
    def test_auto_resolution_and_gigapath_fp32_rejection(self):
        self.assertEqual(resolve_precision_name(_args(backbone="titan")), "fp32")
        self.assertEqual(resolve_precision_name(_args(backbone="gigapath")), "fp16")
        with self.assertRaisesRegex(ValueError, "FlashAttention"):
            validate_precision_configuration(
                _args(backbone="gigapath", precision="fp32")
            )

    def test_fp32_does_not_construct_or_call_amp_helpers(self):
        with patch(
            "torch.cuda.amp.GradScaler",
            side_effect=AssertionError("FP32 must not construct GradScaler"),
        ):
            policy = PrecisionPolicy(_args(), torch.device("cpu"))
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        policy.backward(parameter.square())
        policy.step(optimizer)
        self.assertFalse(policy.enabled)
        self.assertIsNone(policy.scaler)

    def test_fp16_uses_scaler_and_round_trips_state(self):
        with patch("torch.cuda.amp.GradScaler", _FakeScaler):
            policy = PrecisionPolicy(
                _args(precision="fp16"), torch.device("cuda")
            )
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        policy.backward(parameter.square())
        policy.step(optimizer)
        self.assertEqual(policy.scaler.backward_calls, 1)
        self.assertEqual(policy.scaler.step_calls, 1)
        self.assertEqual(policy.scaler.update_calls, 1)
        self.assertEqual(policy.state_dict(), {"scale": 8.0})

    def test_checkpoint_contains_precision_state_only_for_amp(self):
        model = Sgd(
            GenericMILBackbone(8, 3, hidden_dim=8),
            torch.nn.functional.cross_entropy,
            _args(),
            None,
        )
        dataset = SimpleNamespace(metadata=lambda fold: {"fold": fold})
        self.assertNotIn("precision_state", checkpoint_payload(model, dataset, 0))
        model.precision_policy = _FakeAmpPolicy()
        model.precision_policy.state_dict = lambda: {"scale": 8.0}
        payload = checkpoint_payload(model, dataset, 0)
        self.assertEqual(payload["precision_state"]["resolved_precision"], "fp16")


class MethodAmpPathTests(unittest.TestCase):
    def _model(self, cls):
        model = cls(
            GenericMILBackbone(8, 3, hidden_dim=8),
            torch.nn.functional.cross_entropy,
            _args(),
            None,
        )
        model.precision_policy = _FakeAmpPolicy()
        return model

    @staticmethod
    def _bag(label=0):
        return (
            torch.randn(5, 8),
            torch.arange(10).reshape(5, 2),
            torch.tensor(1024),
            torch.tensor([label]),
        )

    def test_baseline_amp_backward_and_step(self):
        model = self._model(Sgd)
        model.observe(*self._bag())
        self.assertEqual(model.precision_policy.backward_calls, 1)
        self.assertEqual(model.precision_policy.step_calls, 1)

    def test_agem_unscales_projected_gradients(self):
        model = self._model(AGem)
        bag = self._bag()
        dataset = SimpleNamespace(train_loader=[bag])
        model.end_task(dataset)
        model.observe(*self._bag(label=1))
        self.assertEqual(model.precision_policy.backward_calls, 2)
        self.assertGreater(model.precision_policy.gradient_scale_calls, 0)
        self.assertEqual(model.precision_policy.step_calls, 1)

    def test_ewc_unscales_fisher_gradients(self):
        model = self._model(EwcOn)
        dataset = SimpleNamespace(train_loader=[self._bag()])
        model.end_task(dataset)
        self.assertIsNotNone(model.fish)
        self.assertTrue(torch.isfinite(model.fish).all())
        self.assertGreater(model.precision_policy.gradient_scale_calls, 0)


if __name__ == "__main__":
    unittest.main()
