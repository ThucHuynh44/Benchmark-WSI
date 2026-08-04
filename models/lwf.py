# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cProfile import label
from copy import deepcopy
import torch
from utils.args import *
import torch.optim as optim
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, default=2,
                        help='Temperature of the softmax function.')
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.current_task = 0
        nc = args.num_classes
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

    def begin_task(self, dataset):
        self.current_task += 1

    def end_task(self, dataset):
        self.old_net = deepcopy(self.net).eval()
        for parameter in self.old_net.parameters():
            parameter.requires_grad = False

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        self.opt.zero_grad()
        backbone_inputs = [features, coords, patch_size]
        logits = self.net(backbone_inputs)[0]
        loss = self.loss(logits, labels)
        if self.old_net is not None and task is not None and task > 0:
            old_class_count = self.args.class_offsets[task]
            with torch.no_grad():
                old_logits = self.old_net(backbone_inputs)[0][:, :old_class_count]
            temperature = self.args.softmax_temp
            teacher = torch.softmax(old_logits / temperature, dim=1)
            student = torch.log_softmax(logits[:, :old_class_count] / temperature, dim=1)
            distillation = torch.nn.functional.kl_div(
                student, teacher, reduction='batchmean'
            ) * (temperature ** 2)
            loss += self.args.alpha * distillation

        loss.backward()
        self.opt.step()

        return loss.item()
