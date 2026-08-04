from utils.buffer import Buffer, Instance_Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from einops import rearrange
import math
import numpy as np
import bisect


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, default=0.2,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, default=0.2,
                        help='Penalty weight.')
    return parser


class ConSlide(ContinualModel):
    NAME = 'conslide'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ConSlide, self).__init__(backbone, loss, args, transform)

        self.buffer = Instance_Buffer(
            self.args.buffer_size,
            self.device,
            n_tasks=getattr(self.args, 'n_tasks', 4),
        )

    def observe(self, inputs0, inputs1, patch_size, labels, task, ssl=False):
        if task == 0 and ssl:
            self.opt.zero_grad()
            outputs = self.net([inputs0, inputs1, patch_size])
            loss = 0.001 * outputs[-1].mean()
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            outputs = self.net([inputs0, inputs1, patch_size])

            loss = self.loss(outputs[0], labels) + 0.00001 * outputs[-1].mean()
            

            if task > 0 and not self.buffer.is_empty():
                candidates = []
                for previous_task in range(task):
                    start = self.args.class_offsets[previous_task]
                    stop = start + self.args.task_num_classes[previous_task]
                    if self.buffer.available_labels(range(start, stop)):
                        candidates.append(previous_task)
                if candidates:
                    replay_task = int(np.random.choice(candidates))
                    start = self.args.class_offsets[replay_task]
                    stop = start + self.args.task_num_classes[replay_task]
                    bag_size = np.random.randint(100, 250)
                    replay_losses = []
                    for bag_features, bag_coords, bag_patch_size, global_label in self.buffer.get_task_bags(
                        range(start, stop), bag_size
                    ):
                        replay_logits = self.net([bag_features, bag_coords, bag_patch_size])[0]
                        replay_target = torch.tensor([global_label], device=self.device)
                        replay_losses.append(self.loss(replay_logits, replay_target))
                    if replay_losses:
                        # Every replayed task contributes one mean loss regardless
                        # of whether it has two, three, four or five classes.
                        loss += self.args.alpha * torch.stack(replay_losses).mean()

            loss.backward()
            self.opt.step()

        return loss.item()
    

    def save_buffer(self, inputs0, inputs1, patch_size, labels, task):

        if inputs0.shape[0] > 1:
            with torch.no_grad():
                sample_count = max(1, math.ceil(0.2 * inputs0.shape[0]))
                samp_idx = np.random.choice(inputs0.shape[0], size=sample_count, replace=False)
                topk_inputs0 = inputs0[samp_idx]
                topk_inputs1 = inputs1[samp_idx]

                self.buffer.add_data(
                    examples=[topk_inputs0, topk_inputs1, patch_size],
                    labels=labels,
                    task_labels=task,
                )
