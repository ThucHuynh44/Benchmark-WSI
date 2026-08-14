"""Dynamic OWLoRA layers adapted from the CoMEL continual-bag method."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OWLoRAAdapter(nn.Module):
    """Weighted low-rank residual used by OWLoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"OWLoRA rank must be positive, got {rank}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = rank
        self.down = nn.Linear(
            self.in_features, rank, bias=False, device=device, dtype=dtype
        )
        self.up = nn.Linear(
            rank, self.out_features, bias=False, device=device, dtype=dtype
        )
        self.scales = nn.Parameter(torch.ones(rank, device=device, dtype=dtype))
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        adapter_dtype = self.down.weight.dtype
        down = self.down(hidden_states.to(adapter_dtype))
        return self.up(self.scales * down).to(original_dtype)


class OWLoRALinear(nn.Linear):
    """A Linear layer whose task adapters are materialized incrementally."""

    def __init__(self, *args, adapter_rank: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adapter_rank = int(adapter_rank)
        if self.adapter_rank <= 0:
            raise ValueError(
                f"OWLoRA adapter_rank must be positive, got {self.adapter_rank}"
            )
        self.lora_layers = nn.ModuleList()

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, adapter_rank: int) -> "OWLoRALinear":
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            adapter_rank=adapter_rank,
        )
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        wrapped.weight.requires_grad_(linear.weight.requires_grad)
        if wrapped.bias is not None and linear.bias is not None:
            wrapped.bias.requires_grad_(linear.bias.requires_grad)
        wrapped.train(linear.training)
        return wrapped

    @property
    def has_reference(self) -> bool:
        return len(self.lora_layers) > 0

    @property
    def task_adapter_count(self) -> int:
        return max(0, len(self.lora_layers) - 1)

    def _new_adapter(self, rank: int) -> OWLoRAAdapter:
        return OWLoRAAdapter(
            self.in_features,
            self.out_features,
            rank,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def add_reference(self, rank: int) -> OWLoRAAdapter:
        if self.lora_layers:
            raise RuntimeError("OWLoRA reference must be the first LoRA layer")
        layer = self._new_adapter(rank)
        layer.requires_grad_(False)
        self.lora_layers.append(layer)
        return layer

    def add_task_adapter(self, rank: int | None = None) -> OWLoRAAdapter:
        if not self.lora_layers:
            raise RuntimeError("OWLoRA task adapter requires an initial reference")
        layer = self._new_adapter(self.adapter_rank if rank is None else int(rank))
        self.lora_layers.append(layer)
        return layer

    def rebuild_adapters(self, ranks: Sequence[int]) -> None:
        ranks = [int(rank) for rank in ranks]
        if ranks and len(ranks) < 2:
            raise ValueError(
                "An OWLoRA checkpoint cannot contain a reference without a task adapter"
            )
        current = [int(layer.rank) for layer in self.lora_layers]
        if current == ranks:
            return
        self.lora_layers = nn.ModuleList()
        if ranks:
            self.add_reference(ranks[0])
            for rank in ranks[1:]:
                self.add_task_adapter(rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = F.linear(hidden_states, self.weight, self.bias)
        # Layer zero is the frozen OWLoRA reference and never contributes to
        # the forward path.  Every materialized task adapter is cumulative.
        for adapter in self.lora_layers[1:]:
            output = output + adapter(hidden_states)
        return output


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


def attach_owlora(
    root: nn.Module,
    *,
    root_name: str,
    classifier: nn.Module,
    rank: int,
) -> "OrderedDict[str, OWLoRALinear]":
    """Replace eligible Linear objects and preserve any shared aliases."""

    rank = int(rank)
    if rank <= 0:
        raise ValueError(f"OWLoRA rank must be positive, got {rank}")
    references = _collect_linear_references(
        root, root_name=root_name, excluded=(classifier,)
    )
    if not references:
        raise ValueError(
            f"OWLoRA found no eligible nn.Linear modules below {root_name!r}"
        )

    wrapped_by_path: Dict[str, OWLoRALinear] = {}
    for linear, aliases in references.values():
        canonical = min(path for _, _, path in aliases)
        adapter_rank = (
            3 * rank
            if any("qkv" in path.lower() for _, _, path in aliases)
            else rank
        )
        wrapped = OWLoRALinear.from_linear(linear, adapter_rank=adapter_rank)
        for parent, child_name, path in aliases:
            setattr(parent, child_name, wrapped)
            wrapped_by_path[path] = wrapped

    # One canonical entry per unique wrapped object keeps expansion and
    # checkpoint metadata deterministic even when the source model shares it.
    canonical_items = {}
    for path, module in wrapped_by_path.items():
        canonical_items.setdefault(id(module), (path, module))
        if path < canonical_items[id(module)][0]:
            canonical_items[id(module)] = (path, module)
    return OrderedDict(
        sorted((path, module) for path, module in canonical_items.values())
    )


def _truncated_weight(
    module: OWLoRALinear, energy_threshold: float
) -> Tuple[torch.Tensor, int]:
    weight = module.weight.detach().float()
    if not torch.isfinite(weight).all():
        raise FloatingPointError("Cannot initialize OWLoRA from non-finite weights")
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    squared = singular.square()
    total = squared.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        raise ValueError("Cannot initialize OWLoRA from a zero-energy Linear weight")
    ratios = squared.cumsum(dim=0) / total
    matches = torch.nonzero(ratios > float(energy_threshold), as_tuple=False)
    retained = int(matches[0].item() + 1) if matches.numel() else int(singular.numel())
    reconstructed = (u[:, :retained] * singular[:retained]) @ vh[:retained]
    return reconstructed.to(dtype=module.weight.dtype), retained


def initialize_references(
    modules: Mapping[str, OWLoRALinear], energy_threshold: float
) -> Dict[str, int]:
    """Truncate every base weight, then add the frozen source-style reference."""

    energy_threshold = float(energy_threshold)
    if not 0.0 < energy_threshold < 1.0:
        raise ValueError("OWLoRA SVD energy threshold must be in (0, 1)")
    prepared = {
        path: _truncated_weight(module, energy_threshold)
        for path, module in modules.items()
    }
    if any(module.lora_layers for module in modules.values()):
        raise RuntimeError("OWLoRA references have already been initialized")
    retained_ranks = {}
    with torch.no_grad():
        for path, module in modules.items():
            reconstructed, retained = prepared[path]
            module.weight.copy_(reconstructed)
            module.add_reference(retained)
            retained_ranks[path] = retained
    return retained_ranks


def expand_owlora(modules: Mapping[str, OWLoRALinear]) -> None:
    counts = {module.task_adapter_count for module in modules.values()}
    if len(counts) != 1:
        raise RuntimeError(f"Inconsistent OWLoRA adapter counts: {sorted(counts)}")
    if not all(module.has_reference for module in modules.values()):
        raise RuntimeError("OWLoRA expansion requires references on every module")
    for module in modules.values():
        module.add_task_adapter()


def orthogonality_penalty(
    modules: Mapping[str, OWLoRALinear],
) -> torch.Tensor:
    first = next(iter(modules.values()))
    penalty = first.weight.sum() * 0.0
    for module in modules.values():
        if module.task_adapter_count == 0:
            continue
        current = module.lora_layers[-1]
        down = current.down.weight
        up = current.up.weight
        identity = torch.eye(current.rank, device=down.device, dtype=down.dtype)
        penalty = penalty + (
            (down @ down.t() - identity).square().sum()
            + (up.t() @ up - identity).square().sum()
        ) / float(current.rank**2)
    return penalty


def project_current_gradients(
    modules: Mapping[str, OWLoRALinear],
) -> None:
    """Apply CoMEL's OWLoRA projection without hidden-dimension squares."""

    for module in modules.values():
        if module.task_adapter_count == 0:
            raise RuntimeError("OWLoRA has no current task adapter to project")
        current = module.lora_layers[-1]
        historical = module.lora_layers[:-1]
        down_grad = current.down.weight.grad
        up_grad = current.up.weight.grad
        if down_grad is not None:
            projected = torch.zeros_like(down_grad)
            for old in historical:
                old_down = old.down.weight.detach()
                projected.add_((down_grad @ old_down.t()) @ old_down)
            down_grad.sub_(projected)
        if up_grad is not None:
            projected = torch.zeros_like(up_grad)
            for old in historical:
                old_up = old.up.weight.detach()
                projected.add_(old_up @ (old_up.t() @ up_grad))
            up_grad.sub_(projected)


def reconstruct_adapter_layout_from_state_dict(
    modules: Mapping[str, OWLoRALinear],
    state_dict: Mapping[str, torch.Tensor],
    *,
    model_prefix: str = "net.",
) -> Dict[str, list[int]]:
    """Materialize dynamic adapters by inspecting a checkpoint state dict."""

    layouts: Dict[str, list[int]] = {}
    for path, module in modules.items():
        prefix = f"{model_prefix}{path}.lora_layers."
        entries: MutableMapping[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            index_text, separator, parameter_name = remainder.partition(".")
            if not separator or not index_text.isdigit():
                raise ValueError(f"Malformed OWLoRA checkpoint key {key!r}")
            entries[int(index_text)][parameter_name] = value

        if not entries:
            module.rebuild_adapters([])
            layouts[path] = []
            continue
        indices = sorted(entries)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Non-contiguous OWLoRA adapter indices for {path!r}: {indices}"
            )
        ranks = []
        for index in indices:
            tensors = entries[index]
            required = {"down.weight", "up.weight", "scales"}
            if set(tensors) != required:
                raise ValueError(
                    f"Incomplete OWLoRA adapter {path}[{index}]: "
                    f"expected {sorted(required)}, got {sorted(tensors)}"
                )
            down, up, scales = (
                tensors["down.weight"], tensors["up.weight"], tensors["scales"]
            )
            rank = int(down.shape[0])
            expected_shapes = {
                "down.weight": (rank, module.in_features),
                "up.weight": (module.out_features, rank),
                "scales": (rank,),
            }
            actual = {
                "down.weight": tuple(down.shape),
                "up.weight": tuple(up.shape),
                "scales": tuple(scales.shape),
            }
            if actual != expected_shapes:
                raise ValueError(
                    f"OWLoRA shape mismatch for {path}[{index}]: "
                    f"expected {expected_shapes}, got {actual}"
                )
            if index > 0 and rank != module.adapter_rank:
                raise ValueError(
                    f"OWLoRA task rank mismatch for {path}[{index}]: "
                    f"expected {module.adapter_rank}, got {rank}"
                )
            ranks.append(rank)
        module.rebuild_adapters(ranks)
        layouts[path] = ranks
    return layouts
