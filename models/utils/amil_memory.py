"""Class-balanced pseudo-bag memory used by AMIL.

The pool performs reservoir admission at WSI level and stores only selected
patches.  Cached attention/logit targets are refreshed atomically after every
task, so an incomplete refresh can never be replayed or checkpointed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PseudoBag:
    """One CPU-backed AMIL replay item."""

    features: torch.Tensor
    coords: torch.Tensor
    patch_size: torch.Tensor
    label: torch.Tensor
    origin_task_id: int
    target_seen_class_count: int = 0
    target_attention: Optional[torch.Tensor] = None
    target_logits: Optional[torch.Tensor] = None
    target_snapshot_task: Optional[int] = None

    def to(self, device: torch.device | str) -> "PseudoBag":
        return PseudoBag(
            features=self.features.to(device, non_blocking=True),
            coords=self.coords.to(device, non_blocking=True),
            patch_size=self.patch_size.to(device, non_blocking=True),
            label=self.label.to(device, non_blocking=True),
            origin_task_id=self.origin_task_id,
            target_seen_class_count=self.target_seen_class_count,
            target_attention=(
                None
                if self.target_attention is None
                else self.target_attention.to(device, non_blocking=True)
            ),
            target_logits=(
                None
                if self.target_logits is None
                else self.target_logits.to(device, non_blocking=True)
            ),
            target_snapshot_task=self.target_snapshot_task,
        )


@dataclass(frozen=True)
class ReservoirDecision:
    """A single admission decision that must be committed before the next one."""

    label: int
    index: int
    serial: int


def _attention_vector(attention: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(attention):
        raise TypeError("MaxMinRand attention must be a tensor")
    if attention.ndim == 2 and attention.shape[0] == 1:
        attention = attention.squeeze(0)
    if attention.ndim != 1 or attention.numel() == 0:
        raise ValueError(
            "MaxMinRand attention must have shape [N] or [1,N] with N > 0"
        )
    attention = attention.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(attention).all():
        raise ValueError("MaxMinRand attention contains NaN or Inf")
    return attention


def maxminrand_select(
    attention: torch.Tensor,
    k: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select disjoint random/high/low-attention indices without replacement."""

    if int(k) <= 0:
        raise ValueError("MaxMinRand k must be positive")
    scores = _attention_vector(attention)
    patch_count = int(scores.numel())
    selected_count = min(patch_count, int(k))
    random_count = selected_count // 2
    max_count = selected_count // 4
    min_count = selected_count - random_count - max_count

    order = torch.argsort(scores, stable=True)
    minimum = order[:min_count]
    maximum = order[patch_count - max_count :] if max_count else order[:0]

    used = torch.zeros(patch_count, dtype=torch.bool)
    used[minimum] = True
    used[maximum] = True
    remaining = torch.arange(patch_count, dtype=torch.long)[~used]
    if random_count > int(remaining.numel()):
        raise RuntimeError("MaxMinRand quota exceeds the remaining patch pool")
    if random_count:
        permutation = torch.randperm(
            int(remaining.numel()), generator=generator, device="cpu"
        )
        random_indices = remaining.index_select(0, permutation[:random_count])
    else:
        random_indices = remaining[:0]

    selected = torch.cat((minimum, maximum, random_indices)).sort().values
    if selected.numel() != selected_count or torch.unique(selected).numel() != selected_count:
        raise RuntimeError("MaxMinRand produced duplicate or missing indices")
    return selected


class PseudoBagMemoryPool:
    """Seeded class-balanced WSI reservoir with atomic cached-target refresh."""

    STATE_VERSION = 1

    def __init__(
        self,
        capacity: int,
        *,
        pmp_k: int,
        seed: int,
        num_classes: int = 27,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("AMIL memory capacity must be positive")
        if int(pmp_k) <= 0:
            raise ValueError("AMIL pmp_k must be positive")
        if int(num_classes) <= 0:
            raise ValueError("AMIL num_classes must be positive")
        self.capacity = int(capacity)
        self.pmp_k = int(pmp_k)
        self.num_classes = int(num_classes)

        base_seed = int(seed)
        self.selection_generator = torch.Generator(device="cpu")
        self.selection_generator.manual_seed(base_seed + 11)
        self._reservoir_generator = torch.Generator(device="cpu")
        self._reservoir_generator.manual_seed(base_seed + 23)
        self._replay_generator = torch.Generator(device="cpu")
        self._replay_generator.manual_seed(base_seed + 37)
        priority_generator = torch.Generator(device="cpu")
        priority_generator.manual_seed(base_seed + 53)
        self._class_priority = torch.randperm(
            self.num_classes, generator=priority_generator
        ).long()

        self._entries: Dict[int, List[PseudoBag]] = {
            label: [] for label in range(self.num_classes)
        }
        self._seen_count = torch.zeros(self.num_classes, dtype=torch.long)
        self._quotas = torch.zeros(self.num_classes, dtype=torch.long)
        self._seen_class_count = 0
        self.target_snapshot_task: Optional[int] = None
        self.refresh_required = False
        self._active_update_task: Optional[int] = None
        self._pending_decision: Optional[ReservoirDecision] = None
        self._decision_serial = 0

    def __len__(self) -> int:
        return sum(len(items) for items in self._entries.values())

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(
            label
            for label in range(self.num_classes)
            for _ in self._entries[label]
        )

    @property
    def seen_count(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._seen_count.tolist())

    @property
    def quotas(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._quotas.tolist())

    @property
    def class_priority(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self._class_priority.tolist())

    def label_counts(self) -> torch.Tensor:
        return torch.tensor(
            [len(self._entries[label]) for label in range(self.num_classes)],
            dtype=torch.long,
        )

    def _quota_vector(self, seen_class_count: int) -> torch.Tensor:
        seen_class_count = int(seen_class_count)
        if seen_class_count <= 0 or seen_class_count > self.num_classes:
            raise ValueError(
                "AMIL seen_class_count must be within the global class range"
            )
        quotas = torch.zeros(self.num_classes, dtype=torch.long)
        base, remainder = divmod(self.capacity, seen_class_count)
        quotas[:seen_class_count] = base
        if remainder:
            rank = {
                int(label): index
                for index, label in enumerate(self._class_priority.tolist())
            }
            ordered_seen = sorted(range(seen_class_count), key=rank.__getitem__)
            quotas[torch.tensor(ordered_seen[:remainder], dtype=torch.long)] += 1
        return quotas

    def start_update(self, seen_class_count: int, *, task_id: int) -> None:
        if self.refresh_required or self._active_update_task is not None:
            raise RuntimeError("AMIL memory already has an active task update")
        if self._pending_decision is not None:
            raise RuntimeError("AMIL memory has an uncommitted reservoir decision")
        seen_class_count = int(seen_class_count)
        if seen_class_count < self._seen_class_count:
            raise ValueError("AMIL seen classes cannot decrease")
        new_quotas = self._quota_vector(seen_class_count)

        rebalanced: Dict[int, List[PseudoBag]] = {}
        for label in range(self.num_classes):
            items = self._entries[label]
            quota = int(new_quotas[label])
            if len(items) > quota:
                chosen = torch.randperm(
                    len(items), generator=self._reservoir_generator
                )[:quota].sort().values.tolist()
                items = [items[index] for index in chosen]
            rebalanced[label] = list(items)

        self._entries = rebalanced
        self._quotas = new_quotas
        self._seen_class_count = seen_class_count
        self._active_update_task = int(task_id)
        self.refresh_required = True

    def consider(self, label: int) -> Optional[ReservoirDecision]:
        if not self.refresh_required or self._active_update_task is None:
            raise RuntimeError("AMIL reservoir admission requires start_update()")
        if self._pending_decision is not None:
            raise RuntimeError("Commit the previous AMIL reservoir decision first")
        label = int(label)
        if label < 0 or label >= self._seen_class_count:
            raise ValueError("AMIL candidate label is outside the seen-class prefix")

        self._seen_count[label] += 1
        seen = int(self._seen_count[label])
        quota = int(self._quotas[label])
        items = self._entries[label]
        if quota == 0:
            return None
        if len(items) < quota:
            index = len(items)
        else:
            candidate = int(
                torch.randint(
                    0,
                    seen,
                    (1,),
                    generator=self._reservoir_generator,
                ).item()
            )
            if candidate >= quota:
                return None
            index = candidate

        decision = ReservoirDecision(label, index, self._decision_serial)
        self._decision_serial += 1
        self._pending_decision = decision
        return decision

    def _cpu_entry(self, entry: PseudoBag, *, require_targets: bool) -> PseudoBag:
        if not isinstance(entry, PseudoBag):
            raise TypeError("AMIL memory accepts only PseudoBag entries")
        features = entry.features
        coords = entry.coords
        label = entry.label
        if not all(torch.is_tensor(value) for value in (features, coords, label)):
            raise TypeError("AMIL pseudo-bag features, coords and label must be tensors")
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("AMIL pseudo-bag features must have shape [N,D], N > 0")
        if features.shape[0] > self.pmp_k:
            raise ValueError("AMIL pseudo-bag exceeds pmp_k")
        if coords.ndim != 2 or coords.shape != (features.shape[0], 2):
            raise ValueError("AMIL pseudo-bag coordinates do not match its features")
        if not torch.isfinite(features).all():
            raise ValueError("AMIL pseudo-bag features contain NaN or Inf")
        patch_size = torch.as_tensor(entry.patch_size, dtype=torch.long)
        if patch_size.numel() != 1 or int(patch_size.item()) <= 0:
            raise ValueError("AMIL pseudo-bag patch_size must be one positive integer")
        label = label.long().reshape(-1)
        if label.numel() != 1:
            raise ValueError("AMIL pseudo-bag must have one label")
        label_value = int(label.item())
        if label_value < 0 or label_value >= self.num_classes:
            raise ValueError("AMIL pseudo-bag label is outside the global class range")
        if int(entry.origin_task_id) < 0:
            raise ValueError("AMIL pseudo-bag origin_task_id must be non-negative")

        attention = entry.target_attention
        logits = entry.target_logits
        if require_targets and (attention is None or logits is None):
            raise RuntimeError("AMIL pseudo-bag is missing cached targets")
        if (attention is None) != (logits is None):
            raise ValueError("AMIL pseudo-bag cached targets must be both present or absent")
        if attention is not None:
            if attention.shape != (1, features.shape[0]):
                raise ValueError("AMIL cached attention does not match pseudo-bag patches")
            if not torch.isfinite(attention).all() or torch.any(attention < 0):
                raise ValueError("AMIL cached attention must be finite and non-negative")
            total = attention.float().sum(dim=1, keepdim=True)
            if torch.any(total <= 0):
                raise ValueError("AMIL cached attention must have positive mass")
            attention = attention.detach().to(device="cpu", dtype=torch.float32).clone()
            attention = attention / attention.sum(dim=1, keepdim=True)
            if logits.shape != (1, self.num_classes) or not torch.isfinite(logits).all():
                raise ValueError("AMIL cached logits have an invalid shape or value")
            logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()
            target_count = int(entry.target_seen_class_count)
            if target_count <= 0 or target_count > self.num_classes:
                raise ValueError("AMIL cached target class count is invalid")
            if entry.target_snapshot_task is None:
                raise ValueError("AMIL cached targets require a snapshot task")
        else:
            target_count = 0
            if require_targets:
                raise RuntimeError("AMIL pseudo-bag is missing cached targets")

        return PseudoBag(
            features=features.detach().to(device="cpu", dtype=torch.float32).clone(),
            coords=coords.detach().to(device="cpu", dtype=torch.long).clone(),
            patch_size=patch_size.detach().to(device="cpu").reshape(()).clone(),
            label=label.detach().to(device="cpu").clone(),
            origin_task_id=int(entry.origin_task_id),
            target_seen_class_count=target_count,
            target_attention=attention,
            target_logits=logits,
            target_snapshot_task=entry.target_snapshot_task,
        )

    def commit(self, decision: ReservoirDecision, entry: PseudoBag) -> int:
        if decision != self._pending_decision:
            raise ValueError("AMIL reservoir decision is stale or was not issued by this pool")
        if int(entry.label.reshape(-1)[0].item()) != decision.label:
            raise ValueError("AMIL reservoir decision label does not match pseudo-bag label")
        if int(entry.origin_task_id) != self._active_update_task:
            raise ValueError("AMIL pseudo-bag origin task does not match the active update")
        stored = self._cpu_entry(entry, require_targets=False)
        if stored.target_attention is not None:
            raise ValueError("New AMIL pseudo-bags must be committed before target refresh")
        items = self._entries[decision.label]
        if decision.index < 0 or decision.index > len(items):
            raise ValueError("AMIL reservoir decision index is invalid")
        if decision.index == len(items):
            if len(items) >= int(self._quotas[decision.label]):
                raise RuntimeError("AMIL reservoir append would exceed its class quota")
            items.append(stored)
        else:
            items[decision.index] = stored
        self._pending_decision = None
        return decision.index

    def all(
        self,
        device: torch.device | str,
        *,
        require_targets: bool = True,
    ) -> List[PseudoBag]:
        if require_targets and self.refresh_required:
            raise RuntimeError("AMIL cached targets must be refreshed before replay")
        entries = [
            entry
            for label in range(self.num_classes)
            for entry in self._entries[label]
        ]
        if require_targets:
            for entry in entries:
                self._cpu_entry(entry, require_targets=True)
        return [entry.to(device) for entry in entries]

    def sample(self, size: int, device: torch.device | str) -> List[PseudoBag]:
        entries = self.all(device="cpu", require_targets=True)
        if int(size) <= 0 or not entries:
            return []
        count = min(int(size), len(entries))
        selected = torch.randperm(
            len(entries), generator=self._replay_generator
        )[:count].tolist()
        return [entries[index].to(device) for index in selected]

    def refresh_targets(
        self,
        targets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        target_seen_class_count: int,
        target_snapshot_task: int,
    ) -> None:
        if not self.refresh_required or self._active_update_task is None:
            raise RuntimeError("AMIL target refresh requires an active memory update")
        if self._pending_decision is not None:
            raise RuntimeError("AMIL cannot refresh with an uncommitted reservoir decision")
        if int(target_snapshot_task) != self._active_update_task:
            raise ValueError("AMIL target snapshot task does not match the active update")
        target_seen_class_count = int(target_seen_class_count)
        if target_seen_class_count != self._seen_class_count:
            raise ValueError("AMIL target class count does not match the active quotas")

        flat_entries = [
            entry
            for label in range(self.num_classes)
            for entry in self._entries[label]
        ]
        if len(targets) != len(flat_entries):
            raise ValueError("AMIL target refresh count does not match memory size")

        refreshed_flat: List[PseudoBag] = []
        for entry, raw_target in zip(flat_entries, targets):
            if not isinstance(raw_target, (tuple, list)) or len(raw_target) != 2:
                raise TypeError("AMIL refresh targets must be (attention, logits) pairs")
            attention, logits = raw_target
            candidate = replace(
                entry,
                target_seen_class_count=target_seen_class_count,
                target_attention=attention,
                target_logits=logits,
                target_snapshot_task=int(target_snapshot_task),
            )
            refreshed_flat.append(self._cpu_entry(candidate, require_targets=True))

        refreshed: Dict[int, List[PseudoBag]] = {
            label: [] for label in range(self.num_classes)
        }
        for entry in refreshed_flat:
            refreshed[int(entry.label.item())].append(entry)
        self._entries = refreshed
        self.target_snapshot_task = int(target_snapshot_task)
        self.refresh_required = False
        self._active_update_task = None

    def state_dict(self) -> Dict[str, Any]:
        if self.refresh_required or self._active_update_task is not None:
            raise RuntimeError("AMIL cannot serialize an unfinished target refresh")
        if self._pending_decision is not None:
            raise RuntimeError("AMIL cannot serialize an uncommitted reservoir decision")
        entries = self.all(device="cpu", require_targets=True)
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "pmp_k": self.pmp_k,
            "num_classes": self.num_classes,
            "seen_class_count": self._seen_class_count,
            "seen_count": self._seen_count.clone(),
            "quotas": self._quotas.clone(),
            "class_priority": self._class_priority.clone(),
            "target_snapshot_task": self.target_snapshot_task,
            "decision_serial": self._decision_serial,
            "selection_generator_state": self.selection_generator.get_state().clone(),
            "reservoir_generator_state": self._reservoir_generator.get_state().clone(),
            "replay_generator_state": self._replay_generator.get_state().clone(),
            "entries": [
                {
                    "features": entry.features.clone(),
                    "coords": entry.coords.clone(),
                    "patch_size": entry.patch_size.clone(),
                    "label": entry.label.clone(),
                    "origin_task_id": entry.origin_task_id,
                    "target_seen_class_count": entry.target_seen_class_count,
                    "target_attention": entry.target_attention.clone(),
                    "target_logits": entry.target_logits.clone(),
                    "target_snapshot_task": entry.target_snapshot_task,
                }
                for entry in entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any], *, strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("AMIL memory checkpoint must be a dictionary")
        expected = {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "pmp_k": self.pmp_k,
            "num_classes": self.num_classes,
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(
                    f"AMIL memory checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )

        seen_class_count = int(state.get("seen_class_count", 0))
        if seen_class_count < 0 or seen_class_count > self.num_classes:
            raise ValueError("AMIL memory checkpoint has invalid seen classes")
        tensors: Dict[str, torch.Tensor] = {}
        for name in ("seen_count", "quotas", "class_priority"):
            value = state.get(name)
            if not torch.is_tensor(value) or value.shape != (self.num_classes,):
                raise ValueError(f"AMIL memory checkpoint has invalid {name}")
            tensors[name] = value.detach().cpu().long().clone()
        if torch.any(tensors["seen_count"] < 0) or torch.any(tensors["quotas"] < 0):
            raise ValueError("AMIL memory checkpoint has negative counters or quotas")
        if sorted(tensors["class_priority"].tolist()) != list(range(self.num_classes)):
            raise ValueError("AMIL memory checkpoint class priority is not a permutation")

        original_priority = self._class_priority
        self._class_priority = tensors["class_priority"]
        try:
            expected_quotas = (
                torch.zeros(self.num_classes, dtype=torch.long)
                if seen_class_count == 0
                else self._quota_vector(seen_class_count)
            )
        finally:
            self._class_priority = original_priority
        if strict and not torch.equal(tensors["quotas"], expected_quotas):
            raise ValueError("AMIL memory checkpoint class quotas are inconsistent")

        raw_entries = state.get("entries")
        if not isinstance(raw_entries, list) or len(raw_entries) > self.capacity:
            raise ValueError("AMIL memory checkpoint has an invalid entries list")
        restored: Dict[int, List[PseudoBag]] = {
            label: [] for label in range(self.num_classes)
        }
        snapshot_task = state.get("target_snapshot_task")
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise ValueError("AMIL memory checkpoint contains a malformed entry")
            entry = PseudoBag(
                features=raw.get("features"),
                coords=raw.get("coords"),
                patch_size=raw.get("patch_size"),
                label=raw.get("label"),
                origin_task_id=int(raw.get("origin_task_id", -1)),
                target_seen_class_count=int(raw.get("target_seen_class_count", 0)),
                target_attention=raw.get("target_attention"),
                target_logits=raw.get("target_logits"),
                target_snapshot_task=raw.get("target_snapshot_task"),
            )
            entry = self._cpu_entry(entry, require_targets=True)
            label = int(entry.label.item())
            if label >= seen_class_count:
                raise ValueError("AMIL memory entry label is outside saved seen classes")
            if entry.target_snapshot_task != snapshot_task:
                raise ValueError("AMIL memory entries disagree on target snapshot task")
            if entry.target_seen_class_count != seen_class_count:
                raise ValueError(
                    "AMIL memory cached target class count disagrees with its "
                    "saved seen-class prefix"
                )
            restored[label].append(entry)

        for label, items in restored.items():
            if len(items) > int(tensors["quotas"][label]):
                raise ValueError("AMIL memory checkpoint exceeds a class quota")
            if int(tensors["seen_count"][label]) < len(items):
                raise ValueError("AMIL seen_count is smaller than retained entries")
        if raw_entries and snapshot_task is None:
            raise ValueError("AMIL non-empty memory requires a target snapshot task")

        generator_states = []
        for name in (
            "selection_generator_state",
            "reservoir_generator_state",
            "replay_generator_state",
        ):
            value = state.get(name)
            if not torch.is_tensor(value):
                raise ValueError(f"AMIL memory checkpoint is missing {name}")
            generator_states.append(value.detach().cpu())

        self._entries = restored
        self._seen_count = tensors["seen_count"]
        self._quotas = tensors["quotas"]
        self._class_priority = tensors["class_priority"]
        self._seen_class_count = seen_class_count
        self.target_snapshot_task = snapshot_task
        self._decision_serial = int(state.get("decision_serial", 0))
        self.selection_generator.set_state(generator_states[0])
        self._reservoir_generator.set_state(generator_states[1])
        self._replay_generator.set_state(generator_states[2])
        self.refresh_required = False
        self._active_update_task = None
        self._pending_decision = None
