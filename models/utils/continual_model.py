# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from utils.optim import build_optimizer


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []
    SUPPORTED_BACKBONES = None
    REQUIRED_FEATURE_DIM = None
    REQUIRES_TRAINABLE_BACKBONE = False
    CHECKPOINT_USES_STATE_DICT = True
    CHECKPOINT_INCLUDE_OPTIMIZER = True

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = build_optimizer(self.net.parameters(), self.args)
        self.device = get_device()

    def prepare_inputs(self, features, coords, patch_size_level0, training=None):
        """Sample a bag before device transfer and return the common backbone input."""
        if training is None:
            training = self.net.training
        max_patches = int(getattr(self.args, "backbone_max_patches", 0) or 0)
        patch_count = int(features.shape[-2])
        if max_patches > 0 and patch_count > max_patches:
            if training:
                indices = torch.randperm(patch_count, device=features.device)[:max_patches]
                indices = indices.sort().values
            else:
                indices = torch.linspace(
                    0, patch_count - 1, steps=max_patches, device=features.device
                ).round().long()
            features = features.index_select(-2, indices)
            coords = coords.index_select(-2, indices)
        return (
            features.to(self.device),
            coords.to(self.device),
            torch.as_tensor(patch_size_level0, dtype=torch.long, device=self.device),
        )

    def forward(self, x, coords=None, patch_size_level0=None) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        if isinstance(x, (list, tuple)):
            return self.net(x)
        if coords is not None:
            return self.net([x, coords, patch_size_level0])
        return self.net(x)

    def observe_many(self, batches, task=None, ssl=False):
        """Fallback logical-batch hook.

        Specialized WSI methods override this method to perform one optimizer
        update across multiple variable-length bags.  Existing methods remain
        compatible: their default ``bags_per_update`` is one, and this fallback
        also gives a well-defined result if a caller groups them explicitly.
        """
        if not batches:
            raise ValueError("observe_many requires at least one WSI bag")
        values = []
        for features, coords, patch_size, labels in batches:
            result = self.observe(
                features, coords, patch_size, labels, task, ssl=ssl
            )
            if isinstance(result, dict):
                if "loss" not in result:
                    raise ValueError("observe() dictionaries must contain a 'loss' key")
                result = result["loss"]
            if torch.is_tensor(result):
                if result.numel() != 1:
                    raise ValueError("observe() loss tensors must be scalar")
                result = result.detach().item()
            values.append(float(result))
        return sum(values) / len(values)

    def get_checkpoint_state(self):
        """Return continual state that is not represented by ``state_dict``."""
        return {}

    def load_checkpoint_state(self, state, strict=True):
        """Restore state produced by :meth:`get_checkpoint_state`."""
        if strict and state not in (None, {}):
            raise ValueError(
                f"{type(self).__name__} does not support non-empty method checkpoint state"
            )

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
