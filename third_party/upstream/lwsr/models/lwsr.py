from utils.buffer import Retrieval_Buffer as Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from einops import rearrange
import math
import numpy as np
import bisect
from utils.loss import RetrievalLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Lifelong Whole Slide Retrieval')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class LWSR(ContinualModel):
    NAME = 'lwsr'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(LWSR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, n_tasks=4)

    def observe(self, inputs, labels, task):
        criterion = RetrievalLoss()
        if task == 0:
            self.opt.zero_grad()
            results_dict = self.net(data=inputs)
            criterion.set(pred=results_dict['logits'], label=labels, features=results_dict['cls_token'], task=task, is_ours=True)
            loss = criterion.normal_loss_func()
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            results_dict = self.net(data=inputs)

            if not self.buffer.is_empty():
                size = self.args.minibatch_size

                buf_data = self.buffer.get_data(size=size)
                return_index_list = buf_data[0].tolist()

                buf_inputs = buf_data[1]
                buf_labels = buf_data[2]

                buf_outputs = self.net(data=buf_inputs)
                current_dist_mat = self.calc_dist(buf_outputs['cls_token'])

                input_loss_features = torch.concat((results_dict['cls_token'], buf_outputs['cls_token']), dim=0)
                input_loss_labels = torch.cat((labels, buf_labels))
                input_loss_logits = torch.concat((results_dict['logits'], buf_outputs['logits']), dim=0)
                criterion.set(pred=input_loss_logits, label=input_loss_labels, features=input_loss_features, task=task, is_ours=True,
                              previous_dist_mat=self.buffer.previous_dist_matrix, current_dist_mat=current_dist_mat,
                              return_idx=return_index_list)
                loss = criterion.all_loss_func()

            loss.backward()
            self.opt.step()

        return loss.item()

    def save_buffer(self, inputs, labels, task):
        self.opt.zero_grad()
        sample_idx = np.random.randint(inputs.shape[0], size=math.ceil(0.5 * inputs.shape[0]))
        sample_inputs = inputs[sample_idx]
        sample_labels = labels[sample_idx]

        self.buffer.add_data(examples=sample_inputs, labels=sample_labels)
        self.calc_buffer_dist_matrix()

    def calc_dist(self, buf_inputs):
        differences = buf_inputs.unsqueeze(1) - buf_inputs.unsqueeze(0)
        squared_diff = differences ** 2
        squared_distances = squared_diff.sum(dim=2)
        distances = torch.sqrt(squared_distances + 1e-8)

        return distances

    def calc_buffer_dist_matrix(self):
        all_buf_data = self.buffer.get_all_data()
        all_buf_examples = all_buf_data[0]
        all_buf_examples_list = []
        all_buf_labels = all_buf_data[1]

        with torch.no_grad():
            for example in all_buf_examples:
                output = self.net(data=example.unsqueeze(0))['cls_token']
                all_buf_examples_list.append(output)
            all_buf_features = torch.cat(all_buf_examples_list, dim=0)

        differences = all_buf_features.unsqueeze(1) - all_buf_features.unsqueeze(0)
        squared_diff = differences ** 2
        squared_distances = squared_diff.sum(dim=2)
        distances = torch.sqrt(squared_distances + 1e-8)

        self.buffer.set_dist_matrix(dist_matrix=distances)
