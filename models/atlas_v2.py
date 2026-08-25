"""ATLAS-v2: an explicit additive continual-learning ladder for FEATHER.

The implementation intentionally does not import ATLAS-MIL components.  Its
only mechanisms are the explicitly selected LoRA strategy, full-bag experience
replay, slide-level prototypes, replay-based prototype realignment, and the
optional prompt/NCE branch declared by the selected registry setting.
"""

from __future__ import annotations

import hashlib
import json
import os
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone.generic_mil import build_mil_backbone
from backbone.pretrained_mil import (
    FEATHER_MODEL_ID, FEATHER_REVISION, TITAN_MODEL_ID, TITAN_REVISION,
    _resolve_snapshot,
)
from configs.qpmil_vl_prompts import prompt_schema_hash, resolve_class_prompts
from models.qpmil_vl import build_class_features
from models.atlas_distribution import (
    DISTRIBUTION_MODES, DISTRIBUTION_STATE_VERSION, FrozenDistributionHead,
)
from models.utils.continual_model import ContinualModel
from models.utils.owlora import OWLoRAAdapter
from models.utils.wsi_replay import unpack_prepared_batch
from utils.args import add_experiment_args, add_management_args
from utils.optim import build_optimizer


CHECKPOINT_VERSION = 1
REPLAY_CAPACITY = 30


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ATLAS-v2 continual WSI learning")
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--buffer_size", type=int, default=REPLAY_CAPACITY)
    parser.add_argument("--minibatch_size", type=int, default=1)
    parser.add_argument("--bags_per_update", type=int, default=1)
    parser.add_argument("--atlasv2_lora_rank", type=int, default=8)
    parser.add_argument("--atlasv2_lora_alpha", type=float, default=8.0)
    parser.add_argument("--atlasv2_lora_merge_scale", type=float, default=1.0)
    parser.add_argument("--atlasv2_svd_energy", type=float, default=0.99)
    parser.add_argument("--atlasv2_comel_svd_energy", type=float, default=0.99)
    parser.add_argument("--atlasv2_comel_orthogonal_weight", type=float, default=1.0)
    parser.add_argument("--atlasv2_prompt_fusion", type=float, default=0.5)
    parser.add_argument("--atlasv2_prompt_ce_weight", type=float, default=1.0)
    parser.add_argument("--atlasv2_nce_temperature", type=float, default=0.07)
    parser.add_argument("--atlasv2_nce_weight", type=float, default=1.0)
    parser.add_argument(
        "--atlasv2_distribution_mode", type=str, default="legacy",
        choices=("legacy", *DISTRIBUTION_MODES),
    )
    parser.add_argument("--atlasv2_distribution_alpha", type=float, default=0.1)
    parser.add_argument("--atlasv2_distribution_rho", type=float, default=0.5)
    parser.add_argument("--atlasv2_distribution_rank", type=int, default=8)
    parser.add_argument("--atlasv2_distribution_clusters", type=int, default=3)
    parser.add_argument("--atlasv2_distribution_tau_multi", type=float, default=0.1)
    parser.add_argument("--atlasv2_distribution_beta", type=float, default=0.25)
    parser.add_argument("--atlasv2_distribution_tau_task", type=float, default=0.1)
    parser.add_argument("--atlasv2_pt_steps", type=int, default=200)
    parser.add_argument("--atlasv2_pt_lr", type=float, default=0.05)
    parser.add_argument("--atlasv2_pt_samples_per_class", type=int, default=32)
    parser.add_argument("--atlasv2_pt_temperature", type=float, default=0.1)
    parser.add_argument("--atlasv2_pt_anchor_weight", type=float, default=0.1)
    parser.add_argument("--atlasv2_pt_margin_weight", type=float, default=0.0)
    parser.add_argument("--atlasv2_pt_margin", type=float, default=0.2)
    parser.add_argument("--atlasv2_ranpac_dim", type=int, default=2000)
    parser.add_argument("--atlasv2_ranpac_ridge", type=float, default=1.0)
    parser.add_argument("--atlasv2_text_model_id", type=str, default=TITAN_MODEL_ID)
    parser.add_argument("--atlasv2_text_revision", type=str, default=TITAN_REVISION)
    for option, help_text in (
        ("atlasv2_lora", "Enable standard LoRA adapters."),
        (
            "atlasv2_svd_orthogonal",
            "Track merged LoRA subspaces by SVD and project future updates orthogonally.",
        ),
        (
            "atlasv2_comel_owlora",
            "Use cumulative task adapters with CoMEL OWLoRA regularization/projection.",
        ),
        ("atlasv2_replay", "Enable full-feature-bag experience replay."),
        ("atlasv2_prototype", "Use continual slide prototypes at inference."),
        (
            "atlasv2_prototype_lda",
            "Fit a train-only OAS-shrinkage LDA metric over frozen prototypes.",
        ),
        ("atlasv2_realign", "Refresh old prototypes from replay at task boundaries."),
        ("atlasv2_prompt", "Enable the explicit semantic prompt branch."),
        ("atlasv2_nce", "Add slide-to-class-prompt InfoNCE."),
        ("atlasv2_train_classifier", "Train the linear classifier with CE."),
    ):
        parser.add_argument(
            "--" + option,
            action=BooleanOptionalAction,
            default=option == "atlasv2_train_classifier",
            help=help_text,
        )
    return parser


def validate_args(args) -> None:
    if str(getattr(args, "backbone", "")).lower() != "feather":
        raise ValueError("ATLAS-v2 requires the pretrained FEATHER backbone")
    if int(getattr(args, "feature_dim", 768)) != 768:
        raise ValueError("ATLAS-v2 requires 768-D CONCH patch features")
    if bool(getattr(args, "backbone_freeze", False)):
        raise ValueError("ATLAS-v2 owns FEATHER freezing; omit --backbone_freeze")
    if int(getattr(args, "backbone_max_patches", 0) or 0) != 0:
        raise ValueError("ATLAS-v2 requires backbone_max_patches=0 (full bags)")

    lora = bool(getattr(args, "atlasv2_lora", False))
    svd_orthogonal = bool(getattr(args, "atlasv2_svd_orthogonal", False))
    comel_owlora = bool(getattr(args, "atlasv2_comel_owlora", False))
    replay = bool(getattr(args, "atlasv2_replay", False))
    prototype = bool(getattr(args, "atlasv2_prototype", False))
    prototype_lda = bool(getattr(args, "atlasv2_prototype_lda", False))
    distribution_mode = str(getattr(args, "atlasv2_distribution_mode", "legacy"))
    realign = bool(getattr(args, "atlasv2_realign", False))
    prompt = bool(getattr(args, "atlasv2_prompt", False))
    nce = bool(getattr(args, "atlasv2_nce", False))
    if replay and int(getattr(args, "buffer_size", 0)) != REPLAY_CAPACITY:
        raise ValueError("ATLAS-v2 replay budget is exactly 30 WSIs total")
    if not replay and int(getattr(args, "buffer_size", 0)) not in (0, REPLAY_CAPACITY):
        raise ValueError("ATLAS-v2 non-replay settings use no replay memory")
    if realign and not (lora and replay and prototype):
        raise ValueError("ATLAS-v2 realignment requires LoRA, replay, and prototypes")
    if prompt and not (lora and replay and prototype and realign):
        raise ValueError("ATLAS-v2 prompt extension requires the realigned core")
    if nce and not prompt:
        raise ValueError("ATLAS-v2 NCE requires the prompt branch")
    if prototype_lda and not prototype:
        raise ValueError("ATLAS-v2 train-only LDA requires prototypes")
    if prototype_lda and any(
        (lora, replay, realign, prompt, nce,
         bool(getattr(args, "atlasv2_train_classifier", True)))
    ):
        raise ValueError(
            "ATLAS-v2 train-only LDA is restricted to the frozen-prototype "
            "control without LoRA/replay/realignment/prompt/NCE/linear CE"
        )
    if distribution_mode != "legacy":
        if distribution_mode not in DISTRIBUTION_MODES:
            raise ValueError(f"Unknown ATLAS-v2 distribution mode {distribution_mode!r}")
        if not prototype or prototype_lda:
            raise ValueError("ATLAS-v2 distribution settings require prototypes and exclude legacy OAS-LDA")
        if any((lora, replay, realign, prompt, nce, bool(getattr(args, "atlasv2_train_classifier", True)))):
            raise ValueError("ATLAS-v2 distribution suite requires frozen FEATHER without replay/PETL/linear CE")
        if int(getattr(args, "atlasv2_distribution_rank", 0)) not in (2, 4, 8):
            raise ValueError("ATLAS-v2 distribution rank must be one of {2,4,8}")
        if int(getattr(args, "atlasv2_distribution_clusters", 0)) not in (2, 3):
            raise ValueError("ATLAS-v2 distribution clusters must be one of {2,3}")
        if int(getattr(args, "atlasv2_pt_steps", 0)) <= 0:
            raise ValueError("ATLAS-v2 PT steps must be positive")
        if int(getattr(args, "atlasv2_ranpac_dim", 0)) <= 0:
            raise ValueError("ATLAS-v2 RanPAC projection dimension must be positive")
        for name in (
            "atlasv2_distribution_alpha", "atlasv2_distribution_tau_multi",
            "atlasv2_distribution_tau_task", "atlasv2_pt_lr",
            "atlasv2_ranpac_ridge",
        ):
            if float(getattr(args, name, 0.0)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(getattr(args, "atlasv2_distribution_rho", -1.0)) <= 1.0:
            raise ValueError("atlasv2_distribution_rho must be in [0,1]")
        if float(getattr(args, "atlasv2_distribution_beta", -1.0)) < 0.0:
            raise ValueError("atlasv2_distribution_beta must be non-negative")
        if int(getattr(args, "atlasv2_pt_samples_per_class", 0)) <= 0:
            raise ValueError("atlasv2_pt_samples_per_class must be positive")
    if svd_orthogonal and not lora:
        raise ValueError("ATLAS-v2 SVD-orthogonal adaptation requires LoRA")
    if comel_owlora and not lora:
        raise ValueError("ATLAS-v2 CoMEL OWLoRA adaptation requires LoRA")
    if comel_owlora and svd_orthogonal:
        raise ValueError("ATLAS-v2 CoMEL OWLoRA and SVD-orthogonal LoRA are exclusive")
    if int(getattr(args, "atlasv2_lora_rank", 0)) <= 0:
        raise ValueError("atlasv2_lora_rank must be positive")
    for name in (
        "atlasv2_lora_alpha", "atlasv2_lora_merge_scale",
        "atlasv2_prompt_ce_weight", "atlasv2_nce_temperature",
        "atlasv2_nce_weight",
    ):
        if float(getattr(args, name, 0.0)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    fusion = float(getattr(args, "atlasv2_prompt_fusion", 0.5))
    if prompt and fusion != 0.5:
        raise ValueError("ATLAS-v2 prompt fusion is fixed at 0.5")
    if int(getattr(args, "minibatch_size", 1)) <= 0:
        raise ValueError("ATLAS-v2 minibatch_size must be positive")
    svd_energy = float(getattr(args, "atlasv2_svd_energy", 0.99))
    if not 0.0 < svd_energy <= 1.0:
        raise ValueError("atlasv2_svd_energy must be in (0, 1]")
    comel_energy = float(getattr(args, "atlasv2_comel_svd_energy", 0.99))
    if not 0.0 < comel_energy < 1.0:
        raise ValueError("atlasv2_comel_svd_energy must be in (0, 1)")
    if float(getattr(args, "atlasv2_comel_orthogonal_weight", 1.0)) < 0.0:
        raise ValueError("atlasv2_comel_orthogonal_weight must be non-negative")


class StandardLoRALinear(nn.Module):
    """Frozen dense layer plus the standard ``B(A(x)) * alpha/r`` update."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or rank > min(linear.in_features, linear.out_features):
            raise ValueError("LoRA rank is incompatible with the wrapped Linear")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
            if linear.bias is not None else None
        )
        self.lora_a = nn.Parameter(linear.weight.new_empty(self.rank, self.in_features))
        self.lora_b = nn.Parameter(linear.weight.new_zeros(self.out_features, self.rank))
        self.reset_adapter()
        self.train(linear.training)

    def reset_adapter(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            self.lora_b.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        source = inputs.to(self.lora_a.dtype)
        update = F.linear(F.linear(source, self.lora_a), self.lora_b)
        return base + (self.scaling * update).to(base.dtype)

    @torch.no_grad()
    def merge(self, scale: float = 1.0) -> None:
        self.weight.add_(
            (float(scale) * self.scaling * (self.lora_b.float() @ self.lora_a.float()))
            .to(self.weight)
        )
        self.reset_adapter()


class SVDOrthogonalLoRALinear(StandardLoRALinear):
    """Standard LoRA constrained to new output subspaces across tasks.

    The forward path hard-projects ``B`` onto the orthogonal complement of the
    left-singular basis accumulated from earlier merged updates. At a task
    boundary, SVD extracts the smallest left subspace that explains the
    configured energy of the projected update. The learned projected update is
    merged exactly; SVD is used for subspace tracking, not rank compression.
    """

    def __init__(
        self, linear: nn.Linear, rank: int, alpha: float, *,
        max_basis_rank: int, energy_threshold: float,
    ) -> None:
        super().__init__(linear, rank, alpha)
        self.max_basis_rank = min(int(max_basis_rank), self.out_features)
        self.energy_threshold = float(energy_threshold)
        if self.max_basis_rank <= 0:
            raise ValueError("SVD-orthogonal LoRA basis capacity must be positive")
        if not 0.0 < self.energy_threshold <= 1.0:
            raise ValueError("SVD-orthogonal LoRA energy must be in (0, 1]")
        self.register_buffer(
            "svd_basis", self.weight.new_zeros(self.out_features, self.max_basis_rank)
        )
        self.register_buffer("svd_basis_rank", torch.zeros((), dtype=torch.long))

    def historical_basis(self) -> torch.Tensor:
        return self.svd_basis[:, : int(self.svd_basis_rank)]

    def projected_lora_b(self) -> torch.Tensor:
        basis = self.historical_basis().to(self.lora_b)
        if basis.numel() == 0:
            return self.lora_b
        return self.lora_b - basis @ (basis.t() @ self.lora_b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        source = inputs.to(self.lora_a.dtype)
        update = F.linear(F.linear(source, self.lora_a), self.projected_lora_b())
        return base + (self.scaling * update).to(base.dtype)

    @torch.no_grad()
    def merge(self, scale: float = 1.0) -> None:
        projected_b = self.projected_lora_b().float()
        delta = self.scaling * (projected_b @ self.lora_a.detach().float())
        if not torch.isfinite(delta).all():
            raise FloatingPointError("SVD-orthogonal LoRA produced a non-finite update")
        self.weight.add_((float(scale) * delta).to(self.weight))

        energy = delta.square().sum()
        if float(energy) > torch.finfo(delta.dtype).eps:
            u, singular, _ = torch.linalg.svd(delta, full_matrices=False)
            ratios = singular.square().cumsum(0) / singular.square().sum()
            matches = torch.nonzero(
                ratios >= self.energy_threshold, as_tuple=False
            )
            retained = (
                int(matches[0].item() + 1)
                if matches.numel() else int(singular.numel())
            )
            old = self.historical_basis().float()
            candidates = torch.cat((old, u[:, :retained]), dim=1)
            basis, _ = torch.linalg.qr(candidates, mode="reduced")
            kept = min(int(basis.shape[1]), self.max_basis_rank)
            self.svd_basis.zero_()
            self.svd_basis[:, :kept].copy_(basis[:, :kept].to(self.svd_basis))
            self.svd_basis_rank.fill_(kept)
        self.reset_adapter()


class CoMELOWLoRALinear(nn.Module):
    """Frozen SVD-truncated Linear plus cumulative CoMEL OWLoRA adapters.

    Adapter zero in ``reference`` is a frozen source-style subspace used only
    by gradient projection.  A distinct weighted adapter is preallocated for
    every task, while only the active task adapter is trainable and included
    together with completed adapters in the forward path.
    """

    def __init__(
        self, linear: nn.Linear, rank: int, n_tasks: int, energy_threshold: float,
    ) -> None:
        super().__init__()
        if int(rank) <= 0 or int(n_tasks) <= 0:
            raise ValueError("CoMEL OWLoRA rank and number of tasks must be positive")
        if not 0.0 < float(energy_threshold) < 1.0:
            raise ValueError("CoMEL OWLoRA SVD energy must be in (0, 1)")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rank = int(rank)
        self.n_tasks = int(n_tasks)
        self.energy_threshold = float(energy_threshold)

        weight = linear.weight.detach().float()
        if not torch.isfinite(weight).all():
            raise FloatingPointError("Cannot initialize CoMEL OWLoRA from non-finite weights")
        u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        squared = singular.square()
        total = squared.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            raise ValueError("Cannot initialize CoMEL OWLoRA from a zero-energy weight")
        ratios = squared.cumsum(0) / total
        matches = torch.nonzero(ratios > self.energy_threshold, as_tuple=False)
        retained = int(matches[0].item() + 1) if matches.numel() else int(singular.numel())
        truncated = (u[:, :retained] * singular[:retained]) @ vh[:retained]

        self.weight = nn.Parameter(truncated.to(linear.weight), requires_grad=False)
        self.bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
            if linear.bias is not None else None
        )
        self.reference = OWLoRAAdapter(
            self.in_features, self.out_features, retained,
            device=linear.weight.device, dtype=linear.weight.dtype,
        )
        self.task_adapters = nn.ModuleList([
            OWLoRAAdapter(
                self.in_features, self.out_features, self.rank,
                device=linear.weight.device, dtype=linear.weight.dtype,
            )
            for _ in range(self.n_tasks)
        ])
        self.register_buffer("active_task", torch.zeros((), dtype=torch.long))
        self.set_task(0)
        self.train(linear.training)

    def set_task(self, task: int) -> None:
        task = int(task)
        if not 0 <= task < self.n_tasks:
            raise ValueError(f"CoMEL OWLoRA task {task} is outside 0..{self.n_tasks - 1}")
        self.active_task.fill_(task)
        self.reference.requires_grad_(False)
        for index, adapter in enumerate(self.task_adapters):
            adapter.requires_grad_(index == task)

    def current_adapter(self) -> OWLoRAAdapter:
        return self.task_adapters[int(self.active_task.item())]

    def historical_adapters(self) -> Tuple[OWLoRAAdapter, ...]:
        task = int(self.active_task.item())
        return (self.reference, *tuple(self.task_adapters[:task]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight, self.bias)
        stop = int(self.active_task.item()) + 1
        for adapter in self.task_adapters[:stop]:
            output = output + adapter(inputs)
        return output

    def orthogonality_penalty(self) -> torch.Tensor:
        current = self.current_adapter()
        down, up = current.down.weight, current.up.weight
        identity = torch.eye(current.rank, device=down.device, dtype=down.dtype)
        return (
            (down @ down.t() - identity).square().sum()
            + (up.t() @ up - identity).square().sum()
        ) / float(current.rank ** 2)

    @torch.no_grad()
    def project_current_gradients(self) -> None:
        current = self.current_adapter()
        historical = self.historical_adapters()
        down_grad = current.down.weight.grad
        if down_grad is not None:
            projection = torch.zeros_like(down_grad)
            for old in historical:
                old_down = old.down.weight.detach()
                projection.add_((down_grad @ old_down.t()) @ old_down)
            down_grad.sub_(projection)
        up_grad = current.up.weight.grad
        if up_grad is not None:
            projection = torch.zeros_like(up_grad)
            for old in historical:
                old_up = old.up.weight.detach()
                projection.add_(old_up @ (old_up.t() @ up_grad))
            up_grad.sub_(projection)


def _linear_references(
    root: nn.Module, excluded: Iterable[nn.Module]
) -> Dict[int, Tuple[nn.Linear, List[Tuple[nn.Module, str, str]]]]:
    excluded_ids = {id(module) for module in excluded}
    found: MutableMapping[int, Tuple[nn.Linear, List[Tuple[nn.Module, str, str]]]] = {}

    def visit(parent: nn.Module, prefix: str, ancestors: set[int]) -> None:
        for name, child in parent._modules.items():
            if child is None or id(child) in excluded_ids:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if type(child) is nn.Linear and child.out_features > 1:
                if id(child) not in found:
                    found[id(child)] = (child, [])
                found[id(child)][1].append((parent, name, path))
            else:
                if id(child) in ancestors:
                    raise ValueError(f"Module cycle while attaching LoRA at {path}")
                visit(child, path, ancestors | {id(child)})

    visit(root, "backbone.model", {id(root)})
    return dict(found)


def attach_standard_lora(
    root: nn.Module, classifier: nn.Module, rank: int, alpha: float, *,
    svd_orthogonal: bool = False, n_tasks: int = 1,
    svd_energy: float = 0.99,
) -> "OrderedDict[str, StandardLoRALinear]":
    attached: Dict[int, Tuple[str, StandardLoRALinear]] = {}
    for linear, aliases in _linear_references(root, (classifier,)).values():
        if rank > min(linear.in_features, linear.out_features):
            continue
        wrapped = (
            SVDOrthogonalLoRALinear(
                linear, rank, alpha,
                max_basis_rank=int(rank) * int(n_tasks),
                energy_threshold=float(svd_energy),
            )
            if svd_orthogonal
            else StandardLoRALinear(linear, rank, alpha)
        )
        for parent, name, path in aliases:
            setattr(parent, name, wrapped)
            if id(wrapped) not in attached or path < attached[id(wrapped)][0]:
                attached[id(wrapped)] = (path, wrapped)
    if not attached:
        raise ValueError("ATLAS-v2 found no FEATHER Linear eligible for LoRA")
    return OrderedDict(sorted(attached.values()))


def attach_comel_owlora(
    root: nn.Module, classifier: nn.Module, rank: int, n_tasks: int,
    energy_threshold: float,
) -> "OrderedDict[str, CoMELOWLoRALinear]":
    """Attach the CoMEL rank convention while preserving shared aliases."""

    attached: Dict[int, Tuple[str, CoMELOWLoRALinear]] = {}
    for linear, aliases in _linear_references(root, (classifier,)).values():
        adapter_rank = (
            3 * int(rank)
            if any("qkv" in path.lower() for _, _, path in aliases)
            else int(rank)
        )
        wrapped = CoMELOWLoRALinear(
            linear, adapter_rank, int(n_tasks), float(energy_threshold)
        )
        for parent, name, path in aliases:
            setattr(parent, name, wrapped)
            if id(wrapped) not in attached or path < attached[id(wrapped)][0]:
                attached[id(wrapped)] = (path, wrapped)
    if not attached:
        raise ValueError("ATLAS-v2 found no FEATHER Linear eligible for CoMEL OWLoRA")
    return OrderedDict(sorted(attached.values()))


@dataclass
class FullBagEntry:
    features: torch.Tensor
    coords: torch.Tensor
    patch_size: torch.Tensor
    label: torch.Tensor
    origin_task: int
    priority: float

    def to(self, device: torch.device | str) -> "FullBagEntry":
        return FullBagEntry(
            self.features.to(device), self.coords.to(device), self.patch_size.to(device),
            self.label.to(device), self.origin_task, self.priority,
        )


class FullBagReplayBuffer:
    """Seeded class-balanced reservoir containing at most 30 complete WSIs."""

    def __init__(self, capacity: int, feature_dim: int, num_classes: int, seed: int):
        if int(capacity) != REPLAY_CAPACITY:
            raise ValueError("FullBagReplayBuffer capacity must equal 30")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.entries: List[FullBagEntry] = []
        self.seen_classes: List[int] = []
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + 2203)

    def __len__(self) -> int:
        return len(self.entries)

    def _targets(self, available: Mapping[int, int]) -> Dict[int, int]:
        classes = sorted(self.seen_classes)
        target = {label: 0 for label in classes}
        remaining = min(self.capacity, sum(int(available.get(c, 0)) for c in classes))
        # Deterministic water filling gives the lowest global class IDs the
        # remainder whenever multiple classes can accept another exemplar.
        while remaining:
            candidates = [
                label for label in classes if target[label] < int(available.get(label, 0))
            ]
            if not candidates:
                break
            minimum = min(target[label] for label in candidates)
            candidates = [label for label in candidates if target[label] == minimum]
            for label in candidates:
                if remaining == 0:
                    break
                target[label] += 1
                remaining -= 1
        return target

    def add(self, entry: FullBagEntry, seen_classes: Sequence[int]) -> None:
        label = int(entry.label.item())
        self.seen_classes = sorted({int(value) for value in seen_classes})
        if label not in self.seen_classes:
            raise ValueError("Replay entry label is not a seen class")
        candidates = [*self.entries, entry]
        available: Dict[int, int] = defaultdict(int)
        for candidate in candidates:
            available[int(candidate.label.item())] += 1
        targets = self._targets(available)
        kept: List[FullBagEntry] = []
        for class_id in self.seen_classes:
            values = [
                candidate for candidate in candidates
                if int(candidate.label.item()) == class_id
            ]
            values.sort(key=lambda value: value.priority)
            kept.extend(values[: targets[class_id]])
        self.entries = sorted(
            kept, key=lambda value: (int(value.label.item()), value.priority)
        )
        if len(self.entries) > self.capacity:
            raise AssertionError("Replay capacity invariant was violated")

    def make_entry(self, features, coords, patch_size, label, origin_task: int) -> FullBagEntry:
        features = torch.as_tensor(features).detach().cpu().float().clone()
        coords = torch.as_tensor(coords).detach().cpu().long().clone()
        patch_size = torch.as_tensor(patch_size).detach().cpu().long().reshape(())
        label = torch.as_tensor(label).detach().cpu().long().reshape(1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Replay features must be one full [N,D] WSI bag")
        if coords.shape != (features.shape[0], 2) or features.shape[0] == 0:
            raise ValueError("Replay coordinates must match the complete WSI bag")
        return FullBagEntry(
            features, coords, patch_size, label, int(origin_task),
            float(torch.rand((), generator=self.generator)),
        )

    def sample(self, count: int, device: torch.device | str) -> List[FullBagEntry]:
        if not self.entries or count <= 0:
            return []
        by_class: Dict[int, List[int]] = defaultdict(list)
        for index, entry in enumerate(self.entries):
            by_class[int(entry.label.item())].append(index)
        labels = sorted(by_class)
        label_order = torch.randperm(len(labels), generator=self.generator).tolist()
        chosen: List[int] = []
        cursor = 0
        while len(chosen) < min(int(count), len(self.entries)):
            label = labels[label_order[cursor % len(labels)]]
            choices = [index for index in by_class[label] if index not in chosen]
            if choices:
                offset = int(torch.randint(len(choices), (), generator=self.generator))
                chosen.append(choices[offset])
            cursor += 1
            if cursor > len(self.entries) * len(labels):
                break
        return [self.entries[index].to(device) for index in chosen]

    def by_label(self, label: int, device: torch.device | str) -> List[FullBagEntry]:
        return [entry.to(device) for entry in self.entries if int(entry.label.item()) == int(label)]

    def accounting(self) -> Dict[str, float | int]:
        rows = sum(int(entry.features.shape[0]) for entry in self.entries)
        byte_count = sum(
            entry.features.numel() * entry.features.element_size()
            + entry.coords.numel() * entry.coords.element_size()
            + entry.patch_size.numel() * entry.patch_size.element_size()
            + entry.label.numel() * entry.label.element_size()
            for entry in self.entries
        )
        return {
            "retained_wsis": len(self.entries),
            "retained_patch_rows": rows,
            "replay_memory_mib": byte_count / (1024.0 ** 2),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "seen_classes": list(self.seen_classes),
            "generator_state": self.generator.get_state().clone(),
            "entries": [
                {
                    "features": entry.features.clone(), "coords": entry.coords.clone(),
                    "patch_size": entry.patch_size.clone(), "label": entry.label.clone(),
                    "origin_task": entry.origin_task, "priority": entry.priority,
                }
                for entry in self.entries
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("ATLAS-v2 replay checkpoint capacity mismatch")
        if int(state.get("feature_dim", -1)) != self.feature_dim:
            raise ValueError("ATLAS-v2 replay checkpoint feature dimension mismatch")
        if int(state.get("num_classes", -1)) != self.num_classes:
            raise ValueError("ATLAS-v2 replay checkpoint class count mismatch")
        self.seen_classes = [int(value) for value in state.get("seen_classes", [])]
        self.entries = []
        for raw in state.get("entries", []):
            entry = self.make_entry(
                raw["features"], raw["coords"], raw["patch_size"], raw["label"],
                int(raw["origin_task"]),
            )
            entry.priority = float(raw["priority"])
            self.entries.append(entry)
        if len(self.entries) > self.capacity:
            raise ValueError("ATLAS-v2 replay checkpoint exceeds capacity")
        self.generator.set_state(torch.as_tensor(state["generator_state"], dtype=torch.uint8).cpu())


class AtlasV2Network(nn.Module):
    supports_ssl = False

    def __init__(
        self, backbone: nn.Module, num_classes: int, embedding_dim: int,
        prototype: bool, prompt_anchors: torch.Tensor | None, prompt_fusion: float,
        prototype_lda: bool = False,
        distribution_head: FrozenDistributionHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = backbone.get_classifier()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.prototype_enabled = bool(prototype)
        self.prototype_lda_enabled = bool(prototype_lda)
        self.distribution_head = distribution_head
        self.distribution_enabled = distribution_head is not None
        if self.prototype_lda_enabled and not self.prototype_enabled:
            raise ValueError("ATLAS-v2 LDA metric requires a prototype bank")
        self.prompt_enabled = prompt_anchors is not None
        self.prompt_fusion = float(prompt_fusion)
        # Strong compatibility contract: new distribution buffers must never
        # enter the legacy NCM/OAS state-dict keyspace.
        if self.prototype_enabled and not self.distribution_enabled:
            self.register_buffer("prototype_bank", torch.zeros(num_classes, embedding_dim))
            self.register_buffer("prototype_valid", torch.zeros(num_classes, dtype=torch.bool))
        if self.prototype_lda_enabled:
            # These are sufficient statistics of training embeddings only.
            # They are finalized at task boundaries and are read-only at test.
            self.register_buffer("lda_means", torch.zeros(num_classes, embedding_dim))
            self.register_buffer("lda_counts", torch.zeros(num_classes, dtype=torch.long))
            self.register_buffer("lda_scatter", torch.zeros(embedding_dim, embedding_dim))
            self.register_buffer("lda_degrees", torch.zeros((), dtype=torch.long))
            self.register_buffer("lda_weight", torch.zeros(num_classes, embedding_dim))
            self.register_buffer("lda_bias", torch.zeros(num_classes))
            self.register_buffer("lda_shrinkage", torch.ones(()))
            self.register_buffer("lda_fitted", torch.zeros((), dtype=torch.bool))
        if self.prompt_enabled:
            anchors = F.normalize(torch.as_tensor(prompt_anchors).detach().float(), dim=1)
            self.register_buffer("prompt_anchors", anchors)
            self.prompt_projector = nn.Linear(anchors.shape[1], embedding_dim, bias=False)

    def encode(self, features, coords=None, patch_size_level0=None) -> torch.Tensor:
        output = self.backbone.forward_with_embedding(features, coords, patch_size_level0)
        embedding = output.get("embedding") if isinstance(output, dict) else None
        if not torch.is_tensor(embedding) or embedding.shape != (1, self.embedding_dim):
            raise ValueError(f"ATLAS-v2 expects slide embedding [1,{self.embedding_dim}]")
        return embedding.float()

    def linear_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding)

    def prompt_vectors(self) -> torch.Tensor:
        if not self.prompt_enabled:
            raise RuntimeError("ATLAS-v2 prompt branch is disabled")
        return F.normalize(self.prompt_projector(self.prompt_anchors), dim=1, eps=1.0e-6)

    def prompt_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embedding, dim=1, eps=1.0e-6)
        return z @ self.prompt_vectors().t()

    def inference_logits(
        self, embedding: torch.Tensor, seen_classes: int, use_prototype: bool = True
    ) -> torch.Tensor:
        # Task-epoch validation happens before current prototypes may legally be
        # built. It therefore follows the trainable linear CE path. Post-task
        # benchmark evaluation explicitly enables prototype inference.
        if not self.prototype_enabled or not use_prototype:
            logits = self.linear_logits(embedding)
        elif self.distribution_enabled:
            logits = self.distribution_head.scores(embedding, seen_classes)
        elif self.prototype_lda_enabled:
            if not bool(self.lda_fitted):
                raise RuntimeError("ATLAS-v2 train-only LDA has not been fitted")
            z = F.normalize(embedding, dim=1, eps=1.0e-6)
            logits = F.linear(z, self.lda_weight, self.lda_bias)
            logits = logits.masked_fill(~self.prototype_valid.unsqueeze(0), float("-inf"))
        else:
            z = F.normalize(embedding, dim=1, eps=1.0e-6)
            prototypes = F.normalize(self.prototype_bank, dim=1, eps=1.0e-6)
            logits = z @ prototypes.t()
            logits = logits.masked_fill(~self.prototype_valid.unsqueeze(0), float("-inf"))
            if self.prompt_enabled:
                logits = (
                    (1.0 - self.prompt_fusion) * logits
                    + self.prompt_fusion * self.prompt_logits(embedding)
                )
        if seen_classes < self.num_classes:
            logits = logits.clone()
            logits[:, seen_classes:] = float("-inf")
        return logits

    def forward_with_embedding(
        self, features, coords=None, patch_size_level0=None, seen_classes=None,
        use_prototype=True,
    ):
        embedding = self.encode(features, coords, patch_size_level0)
        task_scores = None
        if self.distribution_enabled and bool(use_prototype):
            logits, task_scores = self.distribution_head.scores_and_task_scores(
                embedding,
                self.num_classes if seen_classes is None else int(seen_classes),
            )
        else:
            logits = self.inference_logits(
                embedding, self.num_classes if seen_classes is None else int(seen_classes),
                bool(use_prototype),
            )
        return {
            "embedding": embedding,
            "logits": logits,
            "soft_task_scores": task_scores,
        }

    def forward(self, features, coords=None, patch_size_level0=None, **kwargs):
        if isinstance(features, (list, tuple)):
            values = features
            features = values[0]
            coords = values[1] if len(values) > 1 else coords
            patch_size_level0 = values[2] if len(values) > 2 else patch_size_level0
        output = self.forward_with_embedding(features, coords, patch_size_level0, **kwargs)
        logits = output["logits"]
        attention = torch.full(
            (1, features.shape[0]), 1.0 / features.shape[0],
            device=features.device, dtype=features.dtype,
        )
        return logits, logits.softmax(1), logits.argmax(1), attention, logits.sum() * 0.0

    @torch.no_grad()
    def set_prototype(self, label: int, embeddings: torch.Tensor) -> None:
        values = F.normalize(embeddings.detach().float().reshape(-1, self.embedding_dim), dim=1)
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("Cannot create a prototype from empty/non-finite embeddings")
        prototype = F.normalize(values.mean(dim=0), dim=0)
        self.prototype_bank[int(label)].copy_(prototype.to(self.prototype_bank))
        self.prototype_valid[int(label)] = True

    @torch.no_grad()
    def update_lda_statistics(self, label: int, embeddings: torch.Tensor) -> None:
        """Accumulate one class's train-only normalized mean and scatter."""

        if not self.prototype_lda_enabled:
            raise RuntimeError("ATLAS-v2 train-only LDA is disabled")
        label = int(label)
        if int(self.lda_counts[label]) != 0:
            raise RuntimeError(f"ATLAS-v2 LDA class {label} was already finalized")
        values = F.normalize(
            embeddings.detach().float().reshape(-1, self.embedding_dim), dim=1,
            eps=1.0e-6,
        ).to(self.lda_scatter)
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("Cannot fit LDA from empty/non-finite train embeddings")
        mean = values.mean(0)
        centered = values - mean
        self.lda_means[label].copy_(mean)
        self.lda_counts[label] = int(values.shape[0])
        self.lda_scatter.add_(centered.t() @ centered)
        self.lda_degrees.add_(max(int(values.shape[0]) - 1, 0))

    @torch.no_grad()
    def fit_lda(self, seen_classes: int) -> None:
        """Fit an equal-prior OAS-shrinkage LDA classifier from train statistics."""

        if not self.prototype_lda_enabled:
            raise RuntimeError("ATLAS-v2 train-only LDA is disabled")
        seen_classes = int(seen_classes)
        if seen_classes <= 0 or bool((self.lda_counts[:seen_classes] <= 0).any()):
            raise RuntimeError("ATLAS-v2 LDA is missing a seen class's train statistics")

        dimensions = self.embedding_dim
        degrees = int(self.lda_degrees.item())
        empirical = self.lda_scatter.float() / float(max(degrees, 1))
        mu = empirical.trace() / float(dimensions)
        epsilon = torch.finfo(empirical.dtype).eps
        if degrees == 0 or not torch.isfinite(mu) or float(mu) <= epsilon:
            mu = empirical.new_tensor(1.0)
            shrinkage = empirical.new_tensor(1.0)
        else:
            alpha = empirical.square().mean()
            numerator = alpha + mu.square()
            denominator = float(degrees + 1) * (
                alpha - mu.square() / float(dimensions)
            )
            shrinkage = (
                empirical.new_tensor(1.0)
                if not torch.isfinite(denominator) or float(denominator) <= epsilon
                else (numerator / denominator).clamp(0.0, 1.0)
            )
        covariance = (1.0 - shrinkage) * empirical
        covariance.diagonal().add_(shrinkage * mu)
        covariance.diagonal().add_(torch.clamp(mu * 1.0e-6, min=1.0e-6))
        means = self.lda_means[:seen_classes].float()
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if int(info.item()) == 0:
            precision_means = torch.cholesky_solve(means.t(), cholesky).t()
        else:
            precision_means = means @ torch.linalg.pinv(covariance)
        biases = -0.5 * (means * precision_means).sum(1)
        if not torch.isfinite(precision_means).all() or not torch.isfinite(biases).all():
            raise FloatingPointError("ATLAS-v2 LDA produced non-finite parameters")
        self.lda_weight.zero_()
        self.lda_bias.zero_()
        self.lda_weight[:seen_classes].copy_(precision_means.to(self.lda_weight))
        self.lda_bias[:seen_classes].copy_(biases.to(self.lda_bias))
        self.lda_shrinkage.copy_(shrinkage.to(self.lda_shrinkage))
        self.lda_fitted.fill_(True)


class AtlasV2(ContinualModel):
    NAME = "atlas_v2"
    COMPATIBILITY = ["class-il", "task-il"]
    SUPPORTED_BACKBONES = ("feather",)
    REQUIRED_FEATURE_DIM = 768
    CHECKPOINT_INCLUDE_OPTIMIZER = False
    CHECKPOINT_VERSION = CHECKPOINT_VERSION

    def __init__(self, backbone, loss, args: Namespace, transform, prompt_anchors=None):
        validate_args(args)
        if not callable(getattr(backbone, "forward_with_embedding", None)):
            raise TypeError("ATLAS-v2 FEATHER must expose forward_with_embedding()")
        classifier = backbone.get_classifier()
        if not isinstance(classifier, nn.Linear):
            raise TypeError("ATLAS-v2 requires a linear FEATHER classifier")
        self.lora_enabled = bool(args.atlasv2_lora)
        self.svd_orthogonal_enabled = bool(
            getattr(args, "atlasv2_svd_orthogonal", False)
        )
        self.comel_owlora_enabled = bool(
            getattr(args, "atlasv2_comel_owlora", False)
        )
        self.replay_enabled = bool(args.atlasv2_replay)
        self.prototype_enabled = bool(args.atlasv2_prototype)
        self.prototype_lda_enabled = bool(
            getattr(args, "atlasv2_prototype_lda", False)
        )
        self.distribution_mode = str(
            getattr(args, "atlasv2_distribution_mode", "legacy")
        )
        self.distribution_enabled = self.distribution_mode != "legacy"
        self.realign_enabled = bool(args.atlasv2_realign)
        self.prompt_enabled = bool(args.atlasv2_prompt)
        self.nce_enabled = bool(args.atlasv2_nce)
        self.train_classifier = bool(args.atlasv2_train_classifier)

        lora_modules = {}
        if self.lora_enabled:
            lora_modules = (
                attach_comel_owlora(
                    backbone.model, classifier, int(args.atlasv2_lora_rank),
                    int(args.n_tasks),
                    float(getattr(args, "atlasv2_comel_svd_energy", 0.99)),
                )
                if self.comel_owlora_enabled
                else attach_standard_lora(
                    backbone.model, classifier, int(args.atlasv2_lora_rank),
                    float(args.atlasv2_lora_alpha),
                    svd_orthogonal=self.svd_orthogonal_enabled,
                    n_tasks=int(args.n_tasks),
                    svd_energy=float(getattr(args, "atlasv2_svd_energy", 0.99)),
                )
            )
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        if self.comel_owlora_enabled:
            for module in lora_modules.values():
                module.set_task(0)
        else:
            for module in lora_modules.values():
                module.lora_a.requires_grad_(True)
                module.lora_b.requires_grad_(True)
        classifier.requires_grad_(self.train_classifier)
        if self.prompt_enabled and prompt_anchors is None:
            raise ValueError("ATLAS-v2 prompt setting requires fixed text embeddings")
        if not self.prompt_enabled and prompt_anchors is not None:
            raise ValueError("Prompt anchors must be absent when the prompt branch is off")

        class_task = []
        for task, count in enumerate(args.task_num_classes):
            class_task.extend([task] * int(count))
        distribution_head = (
            FrozenDistributionHead(
                int(args.num_classes), int(classifier.in_features), class_task,
                self.distribution_mode,
                seed=int(getattr(args, "seed", 0) or 0) + 1009 * int(getattr(args, "fold", 0) or 0),
                alpha=float(args.atlasv2_distribution_alpha),
                rho=float(args.atlasv2_distribution_rho),
                rank=int(args.atlasv2_distribution_rank),
                clusters=int(args.atlasv2_distribution_clusters),
                tau_multi=float(args.atlasv2_distribution_tau_multi),
                beta=float(args.atlasv2_distribution_beta),
                tau_task=float(args.atlasv2_distribution_tau_task),
                ranpac_dim=int(args.atlasv2_ranpac_dim),
                ranpac_ridge=float(args.atlasv2_ranpac_ridge),
            )
            if self.distribution_enabled else None
        )
        network = AtlasV2Network(
            backbone, int(args.num_classes), int(classifier.in_features),
            self.prototype_enabled, prompt_anchors,
            float(args.atlasv2_prompt_fusion),
            prototype_lda=self.prototype_lda_enabled,
            distribution_head=distribution_head,
        )
        super().__init__(network, loss, args, transform)
        object.__setattr__(self, "_lora_modules", lora_modules)
        self.num_classes = int(args.num_classes)
        self.feature_dim = int(args.feature_dim)
        self.embedding_dim = int(classifier.in_features)
        if (
            self.distribution_enabled
            and str(getattr(args, "backbone_model_id", "")) == FEATHER_MODEL_ID
            and str(getattr(args, "backbone_revision", "")) == FEATHER_REVISION
            and self.embedding_dim != 512
        ):
            raise ValueError(
                "Pinned FEATHER must expose runtime slide_embedding_dim=512, "
                f"got {self.embedding_dim}"
            )
        self.task_num_classes = tuple(int(v) for v in args.task_num_classes)
        self.class_offsets = tuple(int(v) for v in args.class_offsets)
        self.task_order = tuple(str(v) for v in args.task_order)
        self.n_tasks = int(args.n_tasks)
        self._validate_layout()
        seed = int(getattr(args, "seed", 0) or 0) + 1009 * int(getattr(args, "fold", 0) or 0)
        self.memory = (
            FullBagReplayBuffer(REPLAY_CAPACITY, self.feature_dim, self.num_classes, seed)
            if self.replay_enabled else None
        )
        self.minibatch_size = int(args.minibatch_size)
        self.prompt_ce_weight = float(args.atlasv2_prompt_ce_weight)
        self.nce_temperature = float(args.atlasv2_nce_temperature)
        self.nce_weight = float(args.atlasv2_nce_weight)
        self.merge_scale = float(args.atlasv2_lora_merge_scale)
        self.comel_orthogonal_weight = float(
            getattr(args, "atlasv2_comel_orthogonal_weight", 1.0)
        )
        self.prompt_hash = (
            prompt_schema_hash(self.task_order, self.task_num_classes)
            if self.prompt_enabled else None
        )
        self.current_task = 0
        self.completed_tasks = 0
        self.old_class_count = 0
        self.seen_class_count = self.task_num_classes[0]
        self.memory_history: List[Dict[str, float | int]] = []
        self.calibration_history: List[Dict[str, Any]] = []
        self._reset_optimizer()

    @property
    def lora_modules(self) -> Mapping[str, nn.Module]:
        return self.__dict__["_lora_modules"]

    def _validate_layout(self) -> None:
        if not (
            len(self.task_num_classes) == len(self.class_offsets)
            == len(self.task_order) == self.n_tasks
        ):
            raise ValueError("ATLAS-v2 task metadata lengths differ")
        expected = 0
        for offset, count in zip(self.class_offsets, self.task_num_classes):
            if offset != expected or count <= 0:
                raise ValueError("ATLAS-v2 requires contiguous positive task classes")
            expected += count
        if expected != self.num_classes:
            raise ValueError("ATLAS-v2 task classes do not cover the global classifier")

    def _reset_optimizer(self) -> None:
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        # The frozen-prototype baseline performs no gradient training.  AdamW
        # still needs a parameter list for the shared training/checkpoint API.
        parameters = trainable or [self.net.classifier.weight]
        self.opt = build_optimizer(parameters, self.args)

    def _bounds(self, task: int) -> Tuple[int, int]:
        start = self.class_offsets[int(task)]
        return start, start + self.task_num_classes[int(task)]

    def begin_task(self, dataset) -> None:
        task = int(dataset.current_task) - 1
        if task != self.completed_tasks or not 0 <= task < self.n_tasks:
            raise RuntimeError("ATLAS-v2 tasks must be learned sequentially")
        self.current_task = task
        self.old_class_count, self.seen_class_count = self._bounds(task)
        if self.comel_owlora_enabled:
            for module in self.lora_modules.values():
                module.set_task(task)
        self._reset_optimizer()

    def _unpack(self, batch):
        return unpack_prepared_batch(batch, feature_dim=self.feature_dim)

    def _embedding_and_label(self, batch):
        features, coords, patch_size, label = self._unpack(batch)
        embedding = self.net.encode(features, coords, patch_size)
        return embedding, label.long().reshape(-1)

    def _semantic_loss(self, embedding: torch.Tensor, label: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = self.net.prompt_logits(embedding)[:, : self.seen_class_count] / float(temperature)
        return F.cross_entropy(logits, label)

    def observe_many(self, batches, task=None, ssl=False) -> Dict[str, float]:
        if ssl:
            raise ValueError("ATLAS-v2 has no SSL phase")
        if not batches:
            raise ValueError("ATLAS-v2 observe_many requires current bags")
        if task is not None and int(task) != self.current_task:
            raise RuntimeError("ATLAS-v2 received a non-active task")
        current = [self._embedding_and_label(batch) for batch in batches]
        for _, label in current:
            if not self.old_class_count <= int(label.item()) < self.seen_class_count:
                raise ValueError("Current label is outside the active task")
        # Exact 1:1 protocol: request at most one replay WSI per current WSI.
        # The buffer returns fewer only when it contains fewer exemplars.
        replay_count = len(batches)
        replay_entries = (
            self.memory.sample(replay_count, self.device)
            if self.replay_enabled and self.old_class_count > 0 else []
        )
        replay = [
            self._embedding_and_label((e.features, e.coords, e.patch_size, e.label))
            for e in replay_entries
        ]
        samples = [*current, *replay]
        zero = samples[0][0].sum() * 0.0
        linear_losses = [
            F.cross_entropy(
                self.net.linear_logits(embedding)[:, : self.seen_class_count], label
            )
            for embedding, label in samples
        ] if self.train_classifier else []
        prompt_losses = [
            self._semantic_loss(embedding, label, 1.0)
            for embedding, label in samples
        ] if self.prompt_enabled else []
        nce_losses = [
            self._semantic_loss(embedding, label, self.nce_temperature)
            for embedding, label in samples
        ] if self.nce_enabled else []
        loss_linear = torch.stack(linear_losses).mean() if linear_losses else zero
        loss_prompt = torch.stack(prompt_losses).mean() if prompt_losses else zero
        loss_nce = torch.stack(nce_losses).mean() if nce_losses else zero
        loss_comel = (
            torch.stack([
                module.orthogonality_penalty()
                for module in self.lora_modules.values()
            ]).sum()
            if self.comel_owlora_enabled else zero
        )
        loss = (
            loss_linear + self.prompt_ce_weight * loss_prompt
            + self.nce_weight * loss_nce
            + self.comel_orthogonal_weight * loss_comel
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("ATLAS-v2 produced a non-finite loss")
        self.opt.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
            if self.comel_owlora_enabled:
                for module in self.lora_modules.values():
                    module.project_current_gradients()
            self.opt.step()
        return {
            "loss": float(loss.detach()),
            "loss_cls": float(loss_linear.detach()),
            "loss_prompt": float(loss_prompt.detach()),
            "loss_atlas_nce": float(loss_nce.detach()),
            "loss_comel_orthogonal": float(loss_comel.detach()),
            "replay_bags": float(len(replay_entries)),
            "buffer_size": float(len(self.memory) if self.memory is not None else 0),
        }

    def observe(self, features, coords, patch_size, labels, task=None, ssl=False):
        return self.observe_many(
            [(features, coords, patch_size, labels)], task=task, ssl=ssl
        )

    def save_buffer(self, features, coords, patch_size, labels, task=None) -> int:
        if not self.replay_enabled:
            return -1
        if task is not None and int(task) != self.current_task:
            raise RuntimeError("ATLAS-v2 buffer received a non-active task")
        features, coords, patch_size, label = self._unpack(
            (features, coords, patch_size, labels)
        )
        seen = range(self.seen_class_count)
        self.memory.add(
            self.memory.make_entry(features, coords, patch_size, label, self.current_task),
            seen,
        )
        return 1

    @torch.no_grad()
    def _encode_loader_object(self, source_loader) -> Dict[int, List[torch.Tensor]]:
        loader = DataLoader(
            source_loader.dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=source_loader.collate_fn,
        )
        values: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for batch in loader:
            features, coords, patch_size = self.prepare_inputs(
                batch.features, batch.coords, batch.patch_size_level0, training=False
            )
            label = int(batch.labels.reshape(-1)[0])
            values[label].append(self.net.encode(features, coords, patch_size).cpu())
        return values

    @torch.no_grad()
    def _encode_loader(self, dataset) -> Dict[int, List[torch.Tensor]]:
        return self._encode_loader_object(dataset.train_loader)

    @torch.no_grad()
    def _validation_cache(self, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        loaders = getattr(dataset, "val_loaders", None)
        if not isinstance(loaders, list) or len(loaders) <= self.current_task:
            raise RuntimeError("ATLAS distribution calibration requires retained validation loaders")
        values, labels = [], []
        for loader in loaders[: self.current_task + 1]:
            encoded = self._encode_loader_object(loader)
            for label in sorted(encoded):
                class_values = torch.cat(encoded[label])
                values.append(class_values)
                labels.append(torch.full((class_values.shape[0],), label, dtype=torch.long))
        if not values:
            raise RuntimeError("ATLAS distribution calibration cache is empty")
        return torch.cat(values).to(self.device), torch.cat(labels).to(self.device)

    def _calibrate_distribution(self, dataset, current) -> None:
        head = self.net.distribution_head
        validation_raw, labels = self._validation_cache(dataset)
        validation_norm = F.normalize(validation_raw, dim=1, eps=1.0e-8)
        stages = {
            "diag": ("diag",),
            "diag_shrink": ("diag_shrink",),
            "lowrank": ("lowrank",),
            "task_centroid": ("diag_shrink", "task_centroid"),
            "task_lme": ("diag_shrink", "task_lme"),
            "multi": ("multi",),
            "pt_only": ("diag_shrink",),
            "atlas_tf": ("lowrank", "multi", "atlas_tf"),
            "atlas_pt": ("lowrank", "multi", "atlas_tf"),
            "ranpac": ("ranpac",),
        }[self.distribution_mode]
        selected = []
        for stage in stages:
            cache = validation_raw if stage == "ranpac" else validation_norm
            if stage == "multi" and self.current_task == 0:
                current_values = {
                    label: torch.cat(current[label]).to(self.device)
                    for label in range(*self._bounds(self.current_task))
                }
                original_mode = head.mode
                best_key, best = None, None
                for index, (clusters, tau) in enumerate(
                    product((2, 3), (0.05, 0.1, 0.2))
                ):
                    head.refit_subprototypes(current_values, clusters)
                    head.selected_clusters.fill_(clusters)
                    head.selected_tau_multi.fill_(tau)
                    head.mode = "multi"
                    predictions = head.scores(cache).argmax(1)
                    recalls = [
                        (predictions[labels == label] == label).float().mean()
                        for label in torch.unique(labels)
                    ]
                    metric = float(torch.stack(recalls).mean())
                    key = (metric, -index)
                    if best_key is None or key > best_key:
                        best_key, best = key, (clusters, tau)
                clusters, tau = best
                head.refit_subprototypes(current_values, clusters)
                head.selected_clusters.fill_(clusters)
                head.selected_tau_multi.fill_(tau)
                head.mode = original_mode
                selected.append({
                    "stage": "multi", "clusters": clusters,
                    "tau_multi": tau, "validation_bacc": best_key[0],
                    "candidate_count": 6,
                })
            elif stage == "multi":
                selected.append({
                    "stage": "multi", "locked_from_task": 0,
                    "clusters": int(head.selected_clusters),
                    "tau_multi": float(head.selected_tau_multi),
                    "candidate_count": 0,
                })
            else:
                selected.append({"stage": stage, **head.select(cache, labels, stage)})
        self.calibration_history.append({
            "distribution_state_version": DISTRIBUTION_STATE_VERSION,
            "fold": int(getattr(self.args, "fold", 0) or 0),
            "task": self.current_task,
            "split": "validation",
            "contains_test_cache": False,
            "runtime_slide_embedding_dim": self.embedding_dim,
            "stages": selected,
        })

    def _calibrate_prototype_tuning(self, dataset, current) -> None:
        head = self.net.distribution_head
        validation_raw, labels = self._validation_cache(dataset)
        validation_norm = F.normalize(validation_raw, dim=1, eps=1.0e-8)
        base_offset = head.class_offset.detach().clone()
        best_key, best_offset, best_hp = None, None, None
        candidates = [
            (temperature, anchor, margin)
            for temperature in (0.05, 0.1, 0.2)
            for anchor in (0.01, 0.1, 1.0)
            for margin in (0.0, 0.1)
        ]
        current_tensors = {
            label: torch.cat(current[label]).to(self.device)
            for label in range(*self._bounds(self.current_task))
        }
        for index, (temperature, anchor, margin_weight) in enumerate(candidates):
            with torch.no_grad():
                head.class_offset.copy_(base_offset)
            head.tune_offsets(
                current_tensors,
                task=self.current_task,
                steps=int(self.args.atlasv2_pt_steps),
                learning_rate=float(self.args.atlasv2_pt_lr),
                samples_per_class=int(self.args.atlasv2_pt_samples_per_class),
                temperature=temperature,
                anchor_weight=anchor,
                margin_weight=margin_weight,
                margin=float(self.args.atlasv2_pt_margin),
            )
            with torch.no_grad():
                metric = float(
                    sum(
                        (head.scores(validation_norm).argmax(1)[labels == label] == label)
                        .float().mean()
                        for label in torch.unique(labels)
                    ) / len(torch.unique(labels))
                )
            key = (metric, -index)
            if best_key is None or key > best_key:
                best_key = key
                best_offset = head.class_offset.detach().clone()
                best_hp = {
                    "ce_temperature": temperature,
                    "anchor_weight": anchor,
                    "margin_weight": margin_weight,
                }
        with torch.no_grad():
            head.class_offset.copy_(best_offset)
        self.calibration_history[-1]["stages"].append({
            "stage": "prototype_tuning",
            **best_hp,
            "validation_bacc": float(best_key[0]),
            "candidate_count": len(candidates),
        })

    @torch.no_grad()
    def _realign_old_prototypes(self) -> None:
        for label in range(self.old_class_count):
            entries = self.memory.by_label(label, self.device)
            if not entries:
                raise RuntimeError(f"No replay exemplar available to realign class {label}")
            embeddings = [
                self.net.encode(entry.features, entry.coords, entry.patch_size).cpu()
                for entry in entries
            ]
            self.net.set_prototype(label, torch.cat(embeddings))

    def end_task(self, dataset=None) -> None:
        if dataset is None:
            raise RuntimeError("ATLAS-v2 end_task requires the current train dataset")
        was_training = self.net.training
        self.net.eval()
        try:
            if self.lora_enabled and not self.comel_owlora_enabled:
                for module in self.lora_modules.values():
                    module.merge(self.merge_scale)
            if self.prototype_enabled:
                current = self._encode_loader(dataset)
                if self.distribution_enabled:
                    head = self.net.distribution_head
                    start, stop = self._bounds(self.current_task)
                    for label in range(start, stop):
                        if not current[label]:
                            raise RuntimeError(f"Current train split has no class {label}")
                        embeddings = torch.cat(current[label]).to(self.device)
                        head.fit_class(label, F.normalize(embeddings, dim=1, eps=1.0e-8))
                    if self.distribution_mode == "ranpac":
                        raw = torch.cat([torch.cat(current[label]) for label in range(start, stop)]).to(self.device)
                        labels = torch.cat([
                            torch.full((len(current[label]),), label, dtype=torch.long)
                            for label in range(start, stop)
                        ]).to(self.device)
                        head.update_ranpac(raw, labels)
                    self._calibrate_distribution(dataset, current)
                    if self.distribution_mode in {"pt_only", "atlas_pt"}:
                        self._calibrate_prototype_tuning(dataset, current)
                elif self.realign_enabled and self.old_class_count:
                    self._realign_old_prototypes()
                if not self.distribution_enabled:
                    start, stop = self._bounds(self.current_task)
                    for label in range(start, stop):
                        if not current[label]:
                            raise RuntimeError(f"Current train split has no class {label}")
                        embeddings = torch.cat(current[label])
                        self.net.set_prototype(label, embeddings)
                        if self.prototype_lda_enabled:
                            self.net.update_lda_statistics(label, embeddings)
                    if self.prototype_lda_enabled:
                        self.net.fit_lda(stop)
        finally:
            self.net.train(was_training)
        self.completed_tasks = self.current_task + 1
        accounting = (
            self.memory.accounting() if self.memory is not None
            else {"retained_wsis": 0, "retained_patch_rows": 0, "replay_memory_mib": 0.0}
        )
        self.memory_history.append({"task": self.current_task, **accounting})
        print(
            f"[atlas_v2] finalized task {self.current_task}: "
            f"retained_wsis={accounting['retained_wsis']} "
            f"patch_rows={accounting['retained_patch_rows']} "
            f"memory_mib={accounting['replay_memory_mib']:.2f}"
        )

    def forward(self, x, coords=None, patch_size_level0=None):
        use_prototype = self.prototype_enabled and self.completed_tasks > self.current_task
        with self.autocast_context():
            if isinstance(x, (list, tuple)):
                values = x
                return self.net(
                    values[0], values[1], values[2], seen_classes=self.seen_class_count,
                    use_prototype=use_prototype,
                )
            return self.net(
                x, coords, patch_size_level0, seen_classes=self.seen_class_count,
                use_prototype=use_prototype,
            )

    def forward_with_distribution_diagnostics(
        self, x, coords=None, patch_size_level0=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-encode evaluation path returning oracle-free soft task scores."""

        use_prototype = self.prototype_enabled and self.completed_tasks > self.current_task
        if isinstance(x, (list, tuple)):
            values = x
            x, coords, patch_size_level0 = values[0], values[1], values[2]
        with self.autocast_context():
            output = self.net.forward_with_embedding(
                x, coords, patch_size_level0,
                seen_classes=self.seen_class_count,
                use_prototype=use_prototype,
            )
        return output["logits"], output["soft_task_scores"]

    def _config(self) -> Dict[str, Any]:
        config = {
            "version": CHECKPOINT_VERSION,
            "backbone": "feather",
            "lora": self.lora_enabled, "replay": self.replay_enabled,
            "prototype": self.prototype_enabled, "realign": self.realign_enabled,
            "prompt": self.prompt_enabled, "nce": self.nce_enabled,
            "train_classifier": self.train_classifier,
            "buffer_size": REPLAY_CAPACITY if self.replay_enabled else 0,
            "replay_bag": "full" if self.replay_enabled else None,
            "lora_rank": int(self.args.atlasv2_lora_rank),
            "lora_alpha": float(self.args.atlasv2_lora_alpha),
            "lora_modules": list(self.lora_modules),
            "prompt_fusion": float(self.args.atlasv2_prompt_fusion),
            "prompt_ce_weight": self.prompt_ce_weight,
            "nce_temperature": self.nce_temperature,
            "nce_weight": self.nce_weight,
            "ablation_id": getattr(self.args, "ablation_id", None),
            "ablation_config_hash": getattr(self.args, "ablation_config_hash", None),
        }
        # Preserve checkpoint/run metadata for pre-existing settings. Only
        # opt-in LoRA strategy/geometry extensions receive additional keys.
        if self.svd_orthogonal_enabled:
            config["svd_orthogonal"] = True
            config["svd_energy"] = float(self.args.atlasv2_svd_energy)
        if self.comel_owlora_enabled:
            config["lora_strategy"] = "comel_owlora"
            config["comel_svd_energy"] = float(self.args.atlasv2_comel_svd_energy)
            config["comel_orthogonal_weight"] = self.comel_orthogonal_weight
        if self.prototype_lda_enabled:
            config["prototype_metric"] = "oas_shrinkage_lda"
            config["prototype_fit_data"] = "train_only"
            config["test_time_adaptation"] = False
        if self.distribution_enabled:
            head = self.net.distribution_head
            config.update({
                "distribution_state_version": DISTRIBUTION_STATE_VERSION,
                "distribution_mode": self.distribution_mode,
                "distribution_fit_data": "train_only_normalized_slide_embeddings",
                "runtime_slide_embedding_dim": self.embedding_dim,
                "expected_pinned_feather_embedding_dim": 512,
                "raw_embedding_path": self.distribution_mode == "ranpac",
                "test_time_adaptation": False,
                "alpha": float(head.selected_alpha),
                "rho": float(head.selected_rho),
                "rank": int(head.selected_rank),
                "clusters": int(head.selected_clusters),
                "tau_multi": float(head.selected_tau_multi),
                "beta": float(head.selected_beta),
                "tau_task": float(head.selected_tau_task),
            })
            if self.distribution_mode == "ranpac":
                config.update({
                    "ranpac_projection_dim": int(head.selected_ranpac_dim),
                    "ranpac_ridge": float(head.selected_ranpac_ridge),
                    "ranpac_projection_grid": [2000, 5000, 10000],
                    "ranpac_feature_path": "embedding_raw",
                })
        return config

    def get_run_metadata(self) -> Dict[str, Any]:
        config = self._config()
        accounting = (
            self.memory.accounting() if self.memory is not None
            else {"retained_wsis": 0, "retained_patch_rows": 0, "replay_memory_mib": 0.0}
        )
        return {
            "atlas_v2_config": config,
            "atlas_v2_config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "replay_memory_accounting": accounting,
            "prototype_metric_accounting": (
                {
                    "metric": "oas_shrinkage_lda",
                    "train_samples": int(self.net.lda_counts.sum().item()),
                    "within_class_degrees": int(self.net.lda_degrees.item()),
                    "oas_shrinkage": float(self.net.lda_shrinkage.item()),
                    "test_time_adaptation": False,
                }
                if self.prototype_lda_enabled else None
            ),
            "distribution_accounting": (
                {
                    **self.net.distribution_head.diagnostics(),
                    "mode": self.distribution_mode,
                    "stored_slide_embeddings": 0,
                    "test_time_adaptation": False,
                }
                if self.distribution_enabled else None
            ),
            "calibration_manifest": (
                list(self.calibration_history) if self.distribution_enabled else None
            ),
        }

    def get_calibration_manifest(self) -> Dict[str, Any] | None:
        if not self.distribution_enabled:
            return None
        return {
            "distribution_state_version": DISTRIBUTION_STATE_VERSION,
            "ablation_id": getattr(self.args, "ablation_id", None),
            "fold": int(getattr(self.args, "fold", 0) or 0),
            "runtime_slide_embedding_dim": self.embedding_dim,
            "expected_pinned_feather_embedding_dim": 512,
            "backbone_model_id": getattr(self.args, "backbone_model_id", None),
            "backbone_revision": getattr(self.args, "backbone_revision", None),
            "selection_protocol": "fold-specific isolated validation; staged non-Cartesian DAG",
            "contains_train_embeddings": False,
            "contains_validation_embeddings": False,
            "contains_test_embeddings": False,
            "history": list(self.calibration_history),
        }

    def get_task_diagnostics(self) -> Dict[str, Any] | None:
        if not self.distribution_enabled:
            return None
        return {
            "task": self.current_task,
            **self.net.distribution_head.diagnostics(),
            "selected_hyperparameters": json.dumps(
                self.calibration_history[-1]["stages"] if self.calibration_history else [],
                sort_keys=True,
            ),
        }

    def get_checkpoint_state(self) -> Dict[str, Any]:
        state = {
            "version": CHECKPOINT_VERSION, "method": self.NAME,
            "config": self._config(), "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets),
            "current_task": self.current_task, "completed_tasks": self.completed_tasks,
            "old_class_count": self.old_class_count,
            "seen_class_count": self.seen_class_count,
            "prompt_hash": self.prompt_hash,
            "memory_history": [dict(row) for row in self.memory_history],
            "memory": self.memory.state_dict() if self.memory is not None else None,
        }
        if self.distribution_enabled:
            state["calibration_history"] = list(self.calibration_history)
        return state

    def load_checkpoint_state(self, state: Mapping[str, Any], strict: bool = True) -> None:
        expected = {
            "version": CHECKPOINT_VERSION, "method": self.NAME,
            "config": self._config(), "task_order": list(self.task_order),
            "task_num_classes": list(self.task_num_classes),
            "class_offsets": list(self.class_offsets), "prompt_hash": self.prompt_hash,
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(f"ATLAS-v2 checkpoint mismatch for {key}")
        completed = int(state.get("completed_tasks", -1))
        task = int(state.get("current_task", -1))
        if not 0 <= task < self.n_tasks or completed not in {task, task + 1}:
            raise ValueError("ATLAS-v2 checkpoint task state is invalid")
        old_count, seen_count = self._bounds(task)
        if strict and (
            int(state.get("old_class_count", -1)) != old_count
            or int(state.get("seen_class_count", -1)) != seen_count
        ):
            raise ValueError("ATLAS-v2 checkpoint class boundaries are invalid")
        if self.memory is not None:
            if state.get("memory") is None:
                raise ValueError("ATLAS-v2 replay checkpoint is missing its buffer")
            self.memory.load_state_dict(state["memory"])
        elif state.get("memory") is not None:
            raise ValueError("ATLAS-v2 no-replay checkpoint contains a buffer")
        self.current_task = task
        self.completed_tasks = completed
        self.old_class_count = int(state["old_class_count"])
        self.seen_class_count = int(state["seen_class_count"])
        history = [dict(row) for row in state.get("memory_history", [])]
        if strict and len(history) != completed:
            raise ValueError("ATLAS-v2 checkpoint memory history is incomplete")
        self.memory_history = history
        if self.distribution_enabled:
            calibration = [dict(row) for row in state.get("calibration_history", [])]
            if strict and len(calibration) != completed:
                raise ValueError("ATLAS-v2 distribution checkpoint calibration history is incomplete")
            self.calibration_history = calibration
        if self.comel_owlora_enabled:
            for module in self.lora_modules.values():
                module.set_task(task)
        self._reset_optimizer()


def _load_titan_text(args: Namespace, device: torch.device):
    from transformers import AutoModel

    model_id = str(args.atlasv2_text_model_id)
    revision = str(args.atlasv2_text_revision)
    _resolve_snapshot(
        model_id, revision, getattr(args, "backbone_cache_dir", None),
        bool(getattr(args, "backbone_allow_download", False)),
    )
    allow_download = bool(getattr(args, "backbone_allow_download", False))
    kwargs = {
        "revision": revision, "trust_remote_code": True,
        "local_files_only": not allow_download,
    }
    if getattr(args, "backbone_cache_dir", None) is not None:
        kwargs["cache_dir"] = args.backbone_cache_dir
    if allow_download and os.environ.get("HF_TOKEN"):
        kwargs["token"] = os.environ["HF_TOKEN"]
    model = AutoModel.from_pretrained(model_id, **kwargs)
    model.to(device).eval()
    return model


def build_model_from_components(args, loss, transform, backbone, prompt_anchors=None) -> AtlasV2:
    return AtlasV2(backbone, loss, args, transform, prompt_anchors)


def build_model(args: Namespace, loss, transform) -> AtlasV2:
    validate_args(args)
    backbone = build_mil_backbone(args, int(args.num_classes))
    anchors = None
    if bool(args.atlasv2_prompt):
        prompts = resolve_class_prompts(args.task_order, args.task_num_classes)
        device = torch.device("cpu")
        text_model = _load_titan_text(args, device)
        anchors = build_class_features(text_model, prompts, device).detach().cpu()
        del text_model
    return build_model_from_components(args, loss, transform, backbone, anchors)


ATLASV2 = AtlasV2
