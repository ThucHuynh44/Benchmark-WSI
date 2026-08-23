# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for EWC')
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for EWC online')

    return parser


class EwcOn(ContinualModel):
    SUPPORTS_AMP = True
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(EwcOn, self).__init__(backbone, loss, args, transform)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        fish = torch.zeros_like(self.net.get_params())

        for features, coords, patch_size, labels in dataset.train_loader:
            features, coords, patch_size = self.prepare_inputs(
                features, coords, patch_size, training=True
            )
            labels = labels.to(self.device)
            self.opt.zero_grad()
            logits = self.forward_net([features, coords, patch_size])[0]
            loss = -F.nll_loss(self.logsoft(logits), labels, reduction='none').mean()
            exp_cond_prob = torch.exp(loss.detach())
            self.backward_loss(loss)
            gradients = torch.cat([
                self.unscaled_gradient(parameter)
                for parameter in self.net.parameters()
            ])
            fish += exp_cond_prob * gradients ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):

        self.opt.zero_grad()
        outputs = self.forward_net([features, coords, patch_size])[0]
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        self.backward_loss(loss)
        self.optimizer_step()

        return loss.item()
