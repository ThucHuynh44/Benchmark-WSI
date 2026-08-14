"""Class-balanced pseudo-bag memory for ATLAS-MIL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from models.utils.amil_memory import ReservoirDecision


@dataclass(frozen=True)
class AtlasReplayBag:
    features: torch.Tensor
    coords: torch.Tensor
    patch_size: torch.Tensor
    label: torch.Tensor
    origin_task_id: int
    target_attention: Optional[torch.Tensor] = None
    target_embedding: Optional[torch.Tensor] = None
    target_snapshot_task: Optional[int] = None

    def to(self, device: torch.device | str) -> "AtlasReplayBag":
        return AtlasReplayBag(
            features=self.features.to(device, non_blocking=True),
            coords=self.coords.to(device, non_blocking=True),
            patch_size=self.patch_size.to(device, non_blocking=True),
            label=self.label.to(device, non_blocking=True),
            origin_task_id=self.origin_task_id,
            target_attention=(
                None if self.target_attention is None
                else self.target_attention.to(device, non_blocking=True)
            ),
            target_embedding=(
                None if self.target_embedding is None
                else self.target_embedding.to(device, non_blocking=True)
            ),
            target_snapshot_task=self.target_snapshot_task,
        )


class AtlasMemoryPool:
    """Seeded class-balanced reservoir with atomic teacher-target refresh."""

    STATE_VERSION = 1

    def __init__(
        self,
        capacity: int,
        *,
        pmp_k: int,
        feature_dim: int,
        embedding_dim: int,
        num_classes: int,
        seed: int,
    ) -> None:
        for name, value in {
            "capacity": capacity,
            "pmp_k": pmp_k,
            "feature_dim": feature_dim,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"ATLAS memory {name} must be positive")
        self.capacity = int(capacity)
        self.pmp_k = int(pmp_k)
        self.feature_dim = int(feature_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)

        base_seed = int(seed)
        self.selection_generator = torch.Generator(device="cpu")
        self.selection_generator.manual_seed(base_seed + 101)
        self._reservoir_generator = torch.Generator(device="cpu")
        self._reservoir_generator.manual_seed(base_seed + 103)
        self._replay_generator = torch.Generator(device="cpu")
        self._replay_generator.manual_seed(base_seed + 107)
        priority_generator = torch.Generator(device="cpu")
        priority_generator.manual_seed(base_seed + 109)
        self._class_priority = torch.randperm(
            self.num_classes, generator=priority_generator
        ).long()

        self._entries: Dict[int, List[AtlasReplayBag]] = {
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
        return sum(len(entries) for entries in self._entries.values())

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(
            label for label in range(self.num_classes) for _ in self._entries[label]
        )

    def label_counts(self) -> torch.Tensor:
        return torch.tensor(
            [len(self._entries[label]) for label in range(self.num_classes)],
            dtype=torch.long,
        )

    def _quota_vector(self, seen_class_count: int) -> torch.Tensor:
        seen_class_count = int(seen_class_count)
        if seen_class_count <= 0 or seen_class_count > self.num_classes:
            raise ValueError("ATLAS seen_class_count is outside the global range")
        quotas = torch.zeros(self.num_classes, dtype=torch.long)
        base, remainder = divmod(self.capacity, seen_class_count)
        quotas[:seen_class_count] = base
        if remainder:
            priority = {
                int(label): index
                for index, label in enumerate(self._class_priority.tolist())
            }
            ordered = sorted(range(seen_class_count), key=priority.__getitem__)
            quotas[torch.tensor(ordered[:remainder], dtype=torch.long)] += 1
        return quotas

    def start_update(self, seen_class_count: int, *, task_id: int) -> None:
        if self.refresh_required or self._active_update_task is not None:
            raise RuntimeError("ATLAS memory already has an active update")
        if self._pending_decision is not None:
            raise RuntimeError("ATLAS memory has an uncommitted decision")
        seen_class_count = int(seen_class_count)
        if seen_class_count < self._seen_class_count:
            raise ValueError("ATLAS seen classes cannot decrease")
        quotas = self._quota_vector(seen_class_count)
        for label in range(self.num_classes):
            entries = self._entries[label]
            quota = int(quotas[label])
            if len(entries) > quota:
                keep = torch.randperm(
                    len(entries), generator=self._reservoir_generator
                )[:quota].sort().values.tolist()
                self._entries[label] = [entries[index] for index in keep]
        self._quotas = quotas
        self._seen_class_count = seen_class_count
        self._active_update_task = int(task_id)
        self.refresh_required = True

    def consider(self, label: int) -> Optional[ReservoirDecision]:
        if not self.refresh_required or self._active_update_task is None:
            raise RuntimeError("ATLAS reservoir admission requires start_update()")
        if self._pending_decision is not None:
            raise RuntimeError("Commit the previous ATLAS decision first")
        label = int(label)
        if label < 0 or label >= self._seen_class_count:
            raise ValueError("ATLAS candidate label is outside the seen prefix")
        self._seen_count[label] += 1
        quota = int(self._quotas[label])
        if quota == 0:
            return None
        entries = self._entries[label]
        seen = int(self._seen_count[label])
        if len(entries) < quota:
            index = len(entries)
        else:
            index = int(
                torch.randint(0, seen, (1,), generator=self._reservoir_generator).item()
            )
            if index >= quota:
                return None
        decision = ReservoirDecision(label, index, self._decision_serial)
        self._decision_serial += 1
        self._pending_decision = decision
        return decision

    def _cpu_entry(
        self, entry: AtlasReplayBag, *, require_targets: bool
    ) -> AtlasReplayBag:
        if not isinstance(entry, AtlasReplayBag):
            raise TypeError("ATLAS memory accepts only AtlasReplayBag entries")
        features = torch.as_tensor(entry.features)
        coords = torch.as_tensor(entry.coords)
        label = torch.as_tensor(entry.label).long().reshape(-1)
        patch_size = torch.as_tensor(entry.patch_size, dtype=torch.long).reshape(-1)
        if features.ndim != 2 or tuple(features.shape[1:]) != (self.feature_dim,):
            raise ValueError("ATLAS replay features have an invalid shape")
        if features.shape[0] == 0 or features.shape[0] > self.pmp_k:
            raise ValueError("ATLAS replay patch count is outside [1,pmp_k]")
        if coords.shape != (features.shape[0], 2):
            raise ValueError("ATLAS replay coordinates do not match features")
        if label.numel() != 1 or not 0 <= int(label.item()) < self.num_classes:
            raise ValueError("ATLAS replay label is invalid")
        if patch_size.numel() != 1 or int(patch_size.item()) <= 0:
            raise ValueError("ATLAS replay patch size is invalid")
        if int(entry.origin_task_id) < 0 or not torch.isfinite(features).all():
            raise ValueError("ATLAS replay entry has invalid task/features")

        attention = entry.target_attention
        embedding = entry.target_embedding
        if require_targets and (attention is None or embedding is None):
            raise RuntimeError("ATLAS replay entry is missing cached targets")
        if (attention is None) != (embedding is None):
            raise ValueError("ATLAS cached targets must be both present or absent")
        if attention is not None:
            attention = torch.as_tensor(attention, dtype=torch.float32)
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
            if attention.shape != (1, features.shape[0]):
                raise ValueError("ATLAS cached attention has an invalid shape")
            if embedding.shape != (1, self.embedding_dim):
                raise ValueError("ATLAS cached embedding has an invalid shape")
            if (
                not torch.isfinite(attention).all()
                or not torch.isfinite(embedding).all()
                or torch.any(attention < 0)
                or torch.any(attention.sum(dim=1) <= 0)
            ):
                raise ValueError("ATLAS cached targets contain invalid values")
            attention = attention / attention.sum(dim=1, keepdim=True)
            if entry.target_snapshot_task is None:
                raise ValueError("ATLAS cached targets require a snapshot task")

        return AtlasReplayBag(
            features=features.detach().cpu().float().clone(),
            coords=coords.detach().cpu().long().clone(),
            patch_size=patch_size.detach().cpu().reshape(()).clone(),
            label=label.detach().cpu().clone(),
            origin_task_id=int(entry.origin_task_id),
            target_attention=(None if attention is None else attention.detach().cpu().clone()),
            target_embedding=(None if embedding is None else embedding.detach().cpu().clone()),
            target_snapshot_task=(
                None if entry.target_snapshot_task is None
                else int(entry.target_snapshot_task)
            ),
        )

    def commit(self, decision: ReservoirDecision, entry: AtlasReplayBag) -> int:
        if decision != self._pending_decision:
            raise ValueError("ATLAS reservoir decision is stale")
        if int(entry.label.reshape(-1)[0]) != decision.label:
            raise ValueError("ATLAS decision label does not match entry")
        if int(entry.origin_task_id) != self._active_update_task:
            raise ValueError("ATLAS entry task does not match active update")
        stored = self._cpu_entry(entry, require_targets=False)
        if stored.target_attention is not None:
            raise ValueError("New ATLAS entries cannot contain cached targets")
        entries = self._entries[decision.label]
        if decision.index == len(entries):
            entries.append(stored)
        elif 0 <= decision.index < len(entries):
            entries[decision.index] = stored
        else:
            raise ValueError("ATLAS reservoir decision index is invalid")
        self._pending_decision = None
        return decision.index

    def all(
        self, device: torch.device | str, *, require_targets: bool = True
    ) -> List[AtlasReplayBag]:
        if require_targets and self.refresh_required:
            raise RuntimeError("ATLAS cached targets require refresh")
        entries = [
            entry
            for label in range(self.num_classes)
            for entry in self._entries[label]
        ]
        if require_targets:
            for entry in entries:
                self._cpu_entry(entry, require_targets=True)
        return [entry.to(device) for entry in entries]

    def sample(self, size: int, device: torch.device | str) -> List[AtlasReplayBag]:
        entries = self.all("cpu", require_targets=True)
        if int(size) <= 0 or not entries:
            return []
        count = min(int(size), len(entries))
        indices = torch.randperm(
            len(entries), generator=self._replay_generator
        )[:count].tolist()
        return [entries[index].to(device) for index in indices]

    def embeddings_for_label(
        self, label: int, device: torch.device | str
    ) -> List[torch.Tensor]:
        if self.refresh_required:
            raise RuntimeError("ATLAS memory positives require refreshed targets")
        values = []
        for entry in self._entries[int(label)]:
            checked = self._cpu_entry(entry, require_targets=True)
            values.append(checked.target_embedding.to(device))
        return values

    def refresh_targets(
        self,
        targets: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        target_snapshot_task: int,
    ) -> None:
        if not self.refresh_required or self._active_update_task is None:
            raise RuntimeError("ATLAS target refresh requires an active update")
        if self._pending_decision is not None:
            raise RuntimeError("ATLAS cannot refresh an uncommitted decision")
        if int(target_snapshot_task) != self._active_update_task:
            raise ValueError("ATLAS target snapshot does not match active task")
        flat = [
            entry
            for label in range(self.num_classes)
            for entry in self._entries[label]
        ]
        if len(flat) != len(targets):
            raise ValueError("ATLAS refresh target count does not match memory")
        refreshed: Dict[int, List[AtlasReplayBag]] = {
            label: [] for label in range(self.num_classes)
        }
        for entry, (attention, embedding) in zip(flat, targets):
            candidate = replace(
                entry,
                target_attention=attention,
                target_embedding=embedding,
                target_snapshot_task=int(target_snapshot_task),
            )
            checked = self._cpu_entry(candidate, require_targets=True)
            refreshed[int(checked.label.item())].append(checked)
        self._entries = refreshed
        self.target_snapshot_task = int(target_snapshot_task)
        self.refresh_required = False
        self._active_update_task = None

    def state_dict(self) -> Dict[str, Any]:
        if self.refresh_required or self._active_update_task is not None:
            raise RuntimeError("ATLAS cannot serialize an unfinished refresh")
        entries = self.all("cpu", require_targets=True)
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "pmp_k": self.pmp_k,
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
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
                    "target_attention": entry.target_attention.clone(),
                    "target_embedding": entry.target_embedding.clone(),
                    "target_snapshot_task": entry.target_snapshot_task,
                }
                for entry in entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any], *, strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("ATLAS memory checkpoint must be a dictionary")
        expected = {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "pmp_k": self.pmp_k,
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
        }
        for key, value in expected.items():
            if strict and state.get(key) != value:
                raise ValueError(
                    f"ATLAS memory checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )
        seen = int(state.get("seen_class_count", 0))
        if seen < 0 or seen > self.num_classes:
            raise ValueError("ATLAS memory checkpoint has invalid seen classes")
        # Framework checkpoints are loaded with map_location=model.device, which
        # also moves method-owned bookkeeping tensors to CUDA.  ATLAS memory is
        # deliberately CPU-backed, so normalize counters before comparing them
        # with freshly-created CPU quota tensors or restoring CPU generators.
        priority = torch.as_tensor(
            state.get("class_priority"), dtype=torch.long
        ).detach().cpu()
        seen_count = torch.as_tensor(
            state.get("seen_count"), dtype=torch.long
        ).detach().cpu()
        quotas = torch.as_tensor(
            state.get("quotas"), dtype=torch.long
        ).detach().cpu()
        for name, value in {"priority": priority, "seen_count": seen_count, "quotas": quotas}.items():
            if value.shape != (self.num_classes,):
                raise ValueError(f"ATLAS memory checkpoint has invalid {name}")
        if sorted(priority.tolist()) != list(range(self.num_classes)):
            raise ValueError("ATLAS memory class priority is not a permutation")
        if torch.any(seen_count < 0) or torch.any(quotas < 0):
            raise ValueError("ATLAS memory counters and quotas must be non-negative")

        self._class_priority = priority.clone()
        expected_quotas = (
            torch.zeros(self.num_classes, dtype=torch.long)
            if seen == 0
            else self._quota_vector(seen)
        )
        if strict and not torch.equal(quotas, expected_quotas):
            raise ValueError("ATLAS memory checkpoint quotas are inconsistent")
        self._seen_count = seen_count.clone()
        self._quotas = quotas.clone()
        self._seen_class_count = seen
        self._entries = {label: [] for label in range(self.num_classes)}
        for raw in state.get("entries", []):
            checked = self._cpu_entry(AtlasReplayBag(**raw), require_targets=True)
            self._entries[int(checked.label.item())].append(checked)
        if len(self) > self.capacity or torch.any(self.label_counts() > self._quotas):
            raise ValueError("ATLAS memory checkpoint exceeds its quotas")
        self.target_snapshot_task = state.get("target_snapshot_task")
        if any(
            entry.target_snapshot_task != self.target_snapshot_task
            for entries in self._entries.values()
            for entry in entries
        ):
            raise ValueError("ATLAS memory entries have inconsistent snapshot tasks")
        self._decision_serial = int(state.get("decision_serial", 0))
        for generator, key in (
            (self.selection_generator, "selection_generator_state"),
            (self._reservoir_generator, "reservoir_generator_state"),
            (self._replay_generator, "replay_generator_state"),
        ):
            generator.set_state(torch.as_tensor(state[key], dtype=torch.uint8).cpu())
        self.refresh_required = False
        self._active_update_task = None
        self._pending_decision = None
