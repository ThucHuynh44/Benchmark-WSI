"""Fixed-rank LoRA layers for ATLAS-MIL semantic continual merging."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtlasLoRALinear(nn.Linear):
    """Frozen Linear plus one merged and one active fixed-rank update.

    The merged factors are buffers.  Only the active factors receive gradients,
    so the number of parameters is independent of the number of tasks.
    """

    def __init__(self, *args, rank: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("ATLAS LoRA rank must be positive")
        if self.rank > min(self.in_features, self.out_features):
            raise ValueError(
                "ATLAS LoRA rank cannot exceed the smaller Linear dimension"
            )
        self.register_buffer(
            "merged_up",
            torch.zeros(self.out_features, self.rank, device=self.weight.device, dtype=self.weight.dtype),
        )
        self.register_buffer(
            "merged_down",
            torch.zeros(self.rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype),
        )
        self.active_down = nn.Parameter(
            torch.empty(self.rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.active_up = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.reset_active()
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rank: int) -> "AtlasLoRALinear":
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            rank=rank,
        )
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        wrapped.train(linear.training)
        return wrapped

    def reset_active(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.active_down, std=1.0 / float(self.rank))
            self.active_up.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = F.linear(hidden_states, self.weight, self.bias)
        dtype = self.active_down.dtype
        source = hidden_states.to(dtype)
        merged = F.linear(F.linear(source, self.merged_down), self.merged_up)
        active = F.linear(F.linear(source, self.active_down), self.active_up)
        return base + (merged + active).to(base.dtype)

    def merged_delta(self) -> torch.Tensor:
        return self.merged_up.float() @ self.merged_down.float()

    def active_delta(self) -> torch.Tensor:
        return self.active_up.float() @ self.active_down.float()

    def _merged_left_basis(self) -> torch.Tensor:
        columns = self.merged_up.float()
        norms = torch.linalg.vector_norm(columns, dim=0)
        valid = norms > torch.finfo(columns.dtype).eps
        if not bool(valid.any()):
            return columns[:, :0]
        basis, _ = torch.linalg.qr(columns[:, valid], mode="reduced")
        return basis

    @staticmethod
    def _compress_factors(
        left: torch.Tensor,
        right: torch.Tensor,
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Truncate ``left @ right`` through a small factor-space SVD."""
        q_left, r_left = torch.linalg.qr(left, mode="reduced")
        q_right, r_right = torch.linalg.qr(right.t(), mode="reduced")
        core = r_left @ r_right.t()
        u, singular, vh = torch.linalg.svd(core, full_matrices=False)
        kept = min(int(rank), int(singular.numel()))
        root = singular[:kept].clamp_min(0).sqrt()
        up = (q_left @ u[:, :kept]) * root.unsqueeze(0)
        down = root.unsqueeze(1) * (vh[:kept] @ q_right.t())
        if kept < int(rank):
            up = torch.cat(
                [up, up.new_zeros(up.shape[0], int(rank) - kept)], dim=1
            )
            down = torch.cat(
                [down, down.new_zeros(int(rank) - kept, down.shape[1])], dim=0
            )
        return up, down

    @torch.no_grad()
    def merge_active(self, *, rho: float, scale: float = 1.0) -> None:
        rho = float(rho)
        scale = float(scale)
        if not 0.0 <= rho <= 1.0:
            raise ValueError("ATLAS semantic merge rho must be in [0, 1]")
        if scale <= 0.0:
            raise ValueError("ATLAS LoRA merge scale must be positive")

        old_up = self.merged_up.float()
        old_down = self.merged_down.float()
        active_up = self.active_up.detach().float()
        active_down = self.active_down.detach().float()
        basis = self._merged_left_basis()
        if basis.numel():
            active_up = active_up - (1.0 - rho) * basis @ (basis.t() @ active_up)

        left = torch.cat((old_up, active_up * scale), dim=1)
        right = torch.cat((old_down, active_down), dim=0)
        merged_up, merged_down = self._compress_factors(left, right, self.rank)
        if not torch.isfinite(merged_up).all() or not torch.isfinite(merged_down).all():
            raise FloatingPointError("ATLAS LoRA merge produced NaN or Inf")
        self.merged_up.copy_(merged_up.to(self.merged_up))
        self.merged_down.copy_(merged_down.to(self.merged_down))
        self.reset_active()


def _collect_linear_references(
    root: nn.Module,
    *,
    root_name: str,
    excluded: Iterable[nn.Module],
) -> Dict[int, Tuple[nn.Linear, list[Tuple[nn.Module, str, str]]]]:
    excluded_ids = {id(module) for module in excluded}
    references: MutableMapping[
        int, Tuple[nn.Linear, list[Tuple[nn.Module, str, str]]]
    ] = {}

    def visit(parent: nn.Module, prefix: str, ancestors: set[int]) -> None:
        for child_name, child in parent._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{child_name}" if prefix else child_name
            if id(child) in excluded_ids:
                continue
            if type(child) is nn.Linear and int(child.out_features) > 1:
                entry = references.get(id(child))
                reference = (parent, child_name, path)
                if entry is None:
                    references[id(child)] = (child, [reference])
                else:
                    entry[1].append(reference)
                continue
            if id(child) in ancestors:
                raise ValueError(f"Module cycle detected while adapting {path!r}")
            visit(child, path, ancestors | {id(child)})

    visit(root, root_name, {id(root)})
    return dict(references)


def attach_atlas_lora(
    root: nn.Module,
    *,
    root_name: str,
    classifier: nn.Module,
    rank: int,
) -> "OrderedDict[str, AtlasLoRALinear]":
    """Replace eligible Linear modules while preserving shared aliases."""
    references = _collect_linear_references(
        root, root_name=root_name, excluded=(classifier,)
    )
    if not references:
        raise ValueError(f"ATLAS found no eligible Linear modules below {root_name!r}")
    canonical: Dict[int, Tuple[str, AtlasLoRALinear]] = {}
    for linear, aliases in references.values():
        if int(rank) > min(linear.in_features, linear.out_features):
            continue
        wrapped = AtlasLoRALinear.from_linear(linear, rank=int(rank))
        for parent, child_name, path in aliases:
            setattr(parent, child_name, wrapped)
            if id(wrapped) not in canonical or path < canonical[id(wrapped)][0]:
                canonical[id(wrapped)] = (path, wrapped)
    if not canonical:
        raise ValueError("ATLAS LoRA rank is too large for every eligible Linear")
    return OrderedDict(sorted(canonical.values()))


def merge_atlas_lora(
    modules: Mapping[str, AtlasLoRALinear],
    *,
    rho: float,
    scale: float,
) -> None:
    for module in modules.values():
        module.merge_active(rho=rho, scale=scale)

