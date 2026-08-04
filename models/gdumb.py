# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.optim import lr_scheduler
from utils.buffer import Buffer
import torch
import numpy as np
from utils.optim import build_optimizer
from pathlib import Path
from tqdm import tqdm

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--maxlr', type=float, default=5e-2,
                        help='Penalty weight.')
    parser.add_argument('--minlr', type=float, default=5e-4,
                        help='Penalty weight.')
    parser.add_argument('--fitting_epochs', type=int, default=256,
                        help='Penalty weight.')
    parser.add_argument('--cutmix_alpha', type=float, default=None,
                        help='Penalty weight.')
    add_experiment_args(parser)
    return parser

def fit_buffer(self, epochs, dataset):
    # Imported lazily to avoid a model-discovery import cycle.
    from utils.training import (
        early_stopping_from_args,
        evaluate_val,
        load_checkpoint,
    )

    optimizer = build_optimizer(
        self.net.parameters(), self.args, lr=self.args.maxlr
    )
    self.opt = optimizer
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=self.args.minlr
    )
    fold = int(self.args.fold)
    results_dir = Path("checkpoints") / self.args.exp_desc / f"fold_{fold}"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / ".gdumb_best.pt"
    early_stopping = early_stopping_from_args(self.args)
    epoch_bar = tqdm(
        range(epochs),
        desc=f"fold {fold} GDumb fit",
        leave=False,
        disable=bool(getattr(self.args, "non_verbose", False)),
    )
    for epoch in epoch_bar:
        order = np.random.permutation(len(self.buffer.examples))
        loss = torch.tensor(0.0, device=self.device)
        epoch_loss = 0.0
        batch_bar = tqdm(
            order,
            total=len(order),
            desc=f"epoch {epoch + 1}/{epochs}",
            leave=False,
            disable=bool(getattr(self.args, "non_verbose", False)),
        )
        for index in batch_bar:
            optimizer.zero_grad()
            bag = self.buffer.examples[index]
            label = self.buffer.labels[index]
            logits = self.net(bag)[0]
            loss = self.loss(logits, label)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        scheduler.step()
        average_loss = epoch_loss / max(len(order), 1)
        epoch_bar.set_postfix(loss=f"{average_loss:.4f}", refresh=False)
        if not bool(getattr(self.args, "non_verbose", False)):
            tqdm.write(
                f"[train] fold={fold} GDumb epoch={epoch + 1}/{epochs} "
                f"avg_loss={average_loss:.4f} updates={len(order)}"
            )
        if evaluate_val(
            self,
            dataset,
            dataset.N_TASKS - 1,
            epoch,
            checkpoint_path,
            fold,
            early_stopping,
        ):
            break
    load_checkpoint(self, checkpoint_path, dataset, fold)
    checkpoint_path.unlink(missing_ok=True)

class GDumb(ContinualModel):
    NAME = 'gdumb'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(GDumb, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        self.buffer.add_data(examples=[features, coords, patch_size],
                             labels=labels)
        return 0

    def end_task(self, dataset):
        # new model
        self.task += 1
        if not (self.task == dataset.N_TASKS):
            return
        self.net = dataset.get_backbone().to(self.device)
        # GDumb's actual optimization happens only here, so monitor validation
        # over the complete task stream rather than the last task alone.
        dataset.get_joint_data_loaders(self.args.fold)
        fit_buffer(self, self.args.fitting_epochs, dataset)
