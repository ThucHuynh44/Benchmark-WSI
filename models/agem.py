# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via A-GEM.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(AGem, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        features, coords, patch_size, labels = next(iter(dataset.train_loader))
        features, coords, patch_size = self.prepare_inputs(
            features, coords, patch_size, training=True
        )
        self.buffer.add_data(
            examples=[features, coords, patch_size],
            labels=labels.to(self.device),
        )

    def _flat_grads(self):
        return torch.cat([
            parameter.grad.view(-1) if parameter.grad is not None else torch.zeros_like(parameter).view(-1)
            for parameter in self.net.parameters()
        ])

    def _set_grads(self, flat_grad):
        offset = 0
        for parameter in self.net.parameters():
            numel = parameter.numel()
            parameter.grad = flat_grad[offset:offset + numel].view_as(parameter).clone()
            offset += numel

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        self.opt.zero_grad()
        logits = self.net([features, coords, patch_size])[0]
        loss = self.loss(logits, labels)
        loss.backward()

        if not self.buffer.is_empty():
            grad_xy = self._flat_grads().detach().clone()
            buf_inputs, buf_labels = self.buffer.get_data()
            self.opt.zero_grad()
            buf_logits = self.net(buf_inputs)[0]
            penalty = self.loss(buf_logits, buf_labels)
            penalty.backward()
            grad_er = self._flat_grads().detach().clone()
            dot_prod = torch.dot(grad_xy, grad_er)
            if dot_prod.item() < 0:
                denominator = torch.dot(grad_er, grad_er).clamp_min(1e-12)
                grad_xy = grad_xy - (dot_prod / denominator) * grad_er
            self._set_grads(grad_xy)

        self.opt.step()

        return loss.item()
