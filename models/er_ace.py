# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = args.num_classes
        self.task = 0

    def end_task(self, dataset):
        self.task += 1
        
    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net([features, coords, patch_size])[0]
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0., device=self.device)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data()
            loss_re = self.loss(self.net(buf_inputs)[0], buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=[features, coords, patch_size],
                             labels=labels)

        return loss.item()
