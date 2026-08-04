"""Lifelong Whole-Slide Retrieval adapted to ConSlide's WSI stream.

The immutable upstream core is kept under
``third_party/upstream/lwsr``.  Runtime code deliberately does not import that
snapshot; this module adapts its pair, classification and distance-consistency
objectives to variable-length 768-D TITAN/FEATHER bags.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from models.utils.wsi_replay import ReplayBag, VariableBagReservoir, unpack_prepared_batch
from utils.args import add_experiment_args, add_management_args


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Lifelong Whole-Slide Retrieval")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--buffer_size", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=4)
    parser.add_argument("--bags_per_update", type=int, default=4)
    parser.add_argument("--buffer_max_patches", type=int, default=400)
    parser.add_argument("--pair_loss_weight", type=float, default=1.0)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--dc_loss_weight", type=float, default=0.01)
    return parser


class Lwsr(ContinualModel):
    """LWSR with CPU reservoir replay for variable-length WSI bags."""

    NAME = "lwsr"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("titan", "feather")
    REQUIRES_TRAINABLE_BACKBONE = True
    REQUIRED_FEATURE_DIM = 768
    CHECKPOINT_VERSION = 1

    def __init__(self, backbone, loss, args, transform):
        self._validate_configuration(backbone, args)
        super().__init__(backbone, loss, args, transform)

        seed = getattr(args, "seed", 0)
        self.num_classes = int(args.num_classes)
        self.minibatch_size = max(1, int(getattr(args, "minibatch_size", 4) or 4))
        self.bags_per_update = max(1, int(getattr(args, "bags_per_update", 4) or 4))
        self.pair_loss_weight = float(getattr(args, "pair_loss_weight", 1.0))
        self.ce_loss_weight = float(getattr(args, "ce_loss_weight", 1.0))
        self.dc_loss_weight = float(getattr(args, "dc_loss_weight", 0.01))
        if min(self.pair_loss_weight, self.ce_loss_weight, self.dc_loss_weight) < 0:
            raise ValueError("LWSR loss weights must be non-negative")

        self.buffer = VariableBagReservoir(
            int(getattr(args, "buffer_size", 10)),
            max_patches=int(getattr(args, "buffer_max_patches", 400)),
            seed=0 if seed is None else int(seed),
            feature_dim=768,
        )
        self.current_task = 0
        self.previous_dist_matrix: Optional[torch.Tensor] = None

    @classmethod
    def _validate_configuration(cls, backbone, args) -> None:
        name = str(getattr(args, "backbone", "")).lower()
        if name not in cls.SUPPORTED_BACKBONES:
            raise ValueError(
                f"LWSR supports only {cls.SUPPORTED_BACKBONES}, got backbone={name!r}"
            )
        if int(getattr(args, "feature_dim", 768)) != 768:
            raise ValueError("LWSR requires 768-D TITAN/FEATHER patch features")
        if bool(getattr(args, "backbone_freeze", False)):
            raise ValueError("LWSR requires a trainable slide backbone")
        if int(getattr(args, "num_classes", 0)) <= 0:
            raise ValueError("LWSR requires a positive global class count")
        if int(getattr(args, "buffer_size", 10)) < 0:
            raise ValueError("LWSR buffer_size must be non-negative")
        if not callable(getattr(backbone, "forward_with_embedding", None)):
            raise TypeError("LWSR backbone must implement forward_with_embedding")

    @staticmethod
    def pair_loss(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Upstream LWSR similarity objective without the hard-coded 8 classes."""

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("LWSR embeddings must have shape [B,D] with B > 0")
        labels = labels.long().reshape(-1)
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("LWSR labels and embeddings have different batch sizes")
        if labels.min().item() < 0 or labels.max().item() >= int(num_classes):
            raise ValueError("LWSR label is outside the global classifier range")

        one_hot = F.one_hot(labels, num_classes=int(num_classes)).float()
        label_similarity = 2.0 * one_hot.matmul(one_hot.t()) - 1.0
        feature_dim = embeddings.shape[1]
        similarity_error = embeddings.float().matmul(embeddings.float().t())
        similarity_error = similarity_error / feature_dim - label_similarity
        batch_size = embeddings.shape[0]
        denominator = batch_size * (batch_size - 1 if batch_size > 1 else batch_size)
        quantization = torch.abs(torch.abs(embeddings.float()) - 1.0).mean()
        return similarity_error.square().sum() / denominator + 0.3 * quantization

    @staticmethod
    def distance_consistency_loss(
        previous_dist_matrix: Optional[torch.Tensor],
        current_embeddings: torch.Tensor,
        replay_indices: torch.Tensor,
    ) -> torch.Tensor:
        if previous_dist_matrix is None or replay_indices.numel() == 0:
            return current_embeddings.sum() * 0.0
        if current_embeddings.ndim != 2:
            raise ValueError("Replay embeddings must have shape [B,D]")
        indices = replay_indices.detach().to(device="cpu", dtype=torch.long)
        if indices.min().item() < 0 or indices.max().item() >= previous_dist_matrix.shape[0]:
            raise IndexError("LWSR replay index is outside the saved distance matrix")
        reference = previous_dist_matrix.index_select(0, indices).index_select(1, indices)
        current = torch.cdist(current_embeddings.float(), current_embeddings.float(), p=2)
        return F.mse_loss(current, reference.to(current.device, dtype=current.dtype))

    def _forward_embedding(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.net.forward_with_embedding(features, coords, patch_size)
        if not isinstance(output, dict):
            raise TypeError("forward_with_embedding must return a dictionary")
        logits = output.get("logits")
        embedding = output.get("embedding")
        if not torch.is_tensor(logits) or not torch.is_tensor(embedding):
            raise ValueError("forward_with_embedding must return tensor logits and embedding")
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if logits.ndim != 2 or logits.shape != (1, self.num_classes):
            raise ValueError(
                f"LWSR expects logits [1,{self.num_classes}], got {tuple(logits.shape)}"
            )
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(f"LWSR expects one slide embedding, got {tuple(embedding.shape)}")
        if not torch.isfinite(logits).all() or not torch.isfinite(embedding).all():
            raise FloatingPointError("LWSR backbone returned non-finite values")
        return logits, embedding

    def _forward_batches(
        self,
        batches: Sequence[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits: List[torch.Tensor] = []
        embeddings: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for raw_batch in batches:
            features, coords, patch_size, label = unpack_prepared_batch(raw_batch)
            batch_logits, batch_embedding = self._forward_embedding(
                features, coords, patch_size
            )
            logits.append(batch_logits)
            embeddings.append(batch_embedding)
            labels.append(label)
        if not logits:
            raise ValueError("LWSR observe_many received no WSI bags")
        return torch.cat(logits), torch.cat(embeddings), torch.cat(labels)

    @staticmethod
    def _replay_tuple(item: ReplayBag) -> Tuple[torch.Tensor, ...]:
        return item.features, item.coords, item.patch_size, item.label

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("LWSR does not define a separate SSL phase")
        if task is not None:
            self.current_task = int(task)
        self.net.train()
        self.opt.zero_grad(set_to_none=True)

        current_logits, current_embeddings, current_labels = self._forward_batches(batches)
        all_logits = current_logits
        all_embeddings = current_embeddings
        all_labels = current_labels
        replay_embeddings = current_embeddings.new_empty((0, current_embeddings.shape[1]))
        replay_indices = torch.empty(0, dtype=torch.long)

        replay = []
        if self.current_task > 0:
            replay = self.buffer.sample(self.minibatch_size, self.device)
        if replay:
            replay_logits, replay_embeddings, replay_labels = self._forward_batches(
                [self._replay_tuple(item) for item in replay]
            )
            all_logits = torch.cat((all_logits, replay_logits), dim=0)
            all_embeddings = torch.cat((all_embeddings, replay_embeddings), dim=0)
            all_labels = torch.cat((all_labels, replay_labels), dim=0)
            replay_indices = torch.tensor([item.index for item in replay], dtype=torch.long)

        loss_pair = self.pair_loss(all_embeddings, all_labels, self.num_classes)
        loss_ce = F.cross_entropy(all_logits.float(), all_labels.long())
        loss_dcr = self.distance_consistency_loss(
            self.previous_dist_matrix,
            replay_embeddings,
            replay_indices,
        )
        loss = (
            self.pair_loss_weight * loss_pair
            + self.ce_loss_weight * loss_ce
            + self.dc_loss_weight * loss_dcr
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("LWSR produced a non-finite loss")
        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "loss_pair": float(loss_pair.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_dcr": float(loss_dcr.detach().cpu()),
            "replay_bags": float(len(replay)),
            "buffer_size": float(len(self.buffer)),
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def begin_task(self, dataset) -> None:
        # get_data_loaders advances current_task before this hook is called.
        self.current_task = max(0, int(getattr(dataset, "current_task", 1)) - 1)

    def save_buffer(self, features, coords, patch_size, labels, task=None) -> int:
        if task is not None:
            self.current_task = int(task)
        return self.buffer.add(features, coords, patch_size, labels)

    def _refresh_distance_matrix(self) -> None:
        if self.buffer.is_empty():
            self.previous_dist_matrix = None
            return
        was_training = self.net.training
        self.net.eval()
        embeddings = []
        with torch.no_grad():
            for item in self.buffer.all(self.device):
                _, embedding = self._forward_embedding(
                    item.features, item.coords, item.patch_size
                )
                embeddings.append(embedding)
        stacked = torch.cat(embeddings, dim=0)
        self.previous_dist_matrix = torch.cdist(
            stacked.float(), stacked.float(), p=2
        ).detach().cpu()
        self.net.train(was_training)

    def end_task(self, dataset=None) -> None:
        # The framework populates the reservoir from the completed train split
        # after reloading the best epoch and before calling this hook.
        self._refresh_distance_matrix()

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "buffer_size": self.buffer.capacity,
            "minibatch_size": self.minibatch_size,
            "bags_per_update": self.bags_per_update,
            "buffer_max_patches": self.buffer.max_patches,
            "pair_loss_weight": self.pair_loss_weight,
            "ce_loss_weight": self.ce_loss_weight,
            "dc_loss_weight": self.dc_loss_weight,
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "backbone": str(getattr(self.args, "backbone", "")).lower(),
            "feature_dim": 768,
            "current_task": self.current_task,
            "config": self._checkpoint_config(),
            "buffer": self.buffer.state_dict(),
            "previous_dist_matrix": (
                None
                if self.previous_dist_matrix is None
                else self.previous_dist_matrix.detach().cpu().clone()
            ),
        }

    def load_checkpoint_state(self, state: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("LWSR checkpoint state must be a dictionary")
        expected = {
            "version": self.CHECKPOINT_VERSION,
            "method": self.NAME,
            "backbone": str(getattr(self.args, "backbone", "")).lower(),
            "feature_dim": 768,
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(
                    f"LWSR checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )
        expected_config = self._checkpoint_config()
        if strict and state.get("config") != expected_config:
            raise ValueError(
                "LWSR checkpoint hyperparameters do not match this run: "
                f"saved={state.get('config')!r}, expected={expected_config!r}"
            )
        self.buffer.load_state_dict(state.get("buffer"), strict=strict)
        matrix = state.get("previous_dist_matrix")
        if matrix is not None:
            if not torch.is_tensor(matrix) or matrix.shape != (len(self.buffer), len(self.buffer)):
                raise ValueError("LWSR checkpoint has an invalid distance matrix")
            if not torch.isfinite(matrix).all():
                raise ValueError("LWSR checkpoint distance matrix is non-finite")
            matrix = matrix.detach().cpu().float().clone()
        self.previous_dist_matrix = matrix
        self.current_task = int(state.get("current_task", 0))


# Preserve the upstream spelling for direct research-code imports while the
# ConSlide discovery convention continues to use ``Lwsr``.
LWSR = Lwsr
