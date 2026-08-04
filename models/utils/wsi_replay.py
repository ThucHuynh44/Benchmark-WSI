"""CPU replay storage for variable-length whole-slide feature bags.

This module is intentionally independent from the tensor-shaped buffers in
``utils.buffer``.  A WSI bag has a variable number of patches and must keep its
coordinates and level-0 patch size together, so padding or stacking bags in a
single tensor would silently change the pretrained backbone input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ReplayBag:
    """One replay item, stored on CPU and identified by its reservoir slot."""

    index: int
    features: torch.Tensor
    coords: torch.Tensor
    patch_size: torch.Tensor
    label: torch.Tensor

    def to(self, device: torch.device) -> "ReplayBag":
        return ReplayBag(
            index=self.index,
            features=self.features.to(device, non_blocking=True),
            coords=self.coords.to(device, non_blocking=True),
            patch_size=self.patch_size.to(device, non_blocking=True),
            label=self.label.to(device, non_blocking=True),
        )


def unpack_prepared_batch(
    batch: Sequence[torch.Tensor],
    *,
    feature_dim: int = 768,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the grouped-training tuple used by LWSR and MICIL.

    The training loop has already sampled the bag and transferred it to the
    model device.  This function only normalizes harmless singleton dimensions
    and fails early on malformed data.
    """

    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise TypeError(
            "A grouped WSI batch must be a 4-tuple "
            "(features, coords, patch_size, labels)."
        )
    features, coords, patch_size, labels = batch
    if not all(torch.is_tensor(value) for value in (features, coords, labels)):
        raise TypeError("features, coords and labels must be tensors")

    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    if features.ndim != 2 or features.shape[1] != int(feature_dim):
        raise ValueError(
            f"Expected WSI features [N,{int(feature_dim)}], got {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError("A WSI feature bag cannot be empty")
    if coords.ndim != 2 or tuple(coords.shape) != (features.shape[0], 2):
        raise ValueError(
            "Expected coordinates [N,2] matching the feature bag, "
            f"got {tuple(coords.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError("WSI features contain NaN or infinite values")

    patch_size = torch.as_tensor(patch_size, dtype=torch.long, device=features.device)
    if patch_size.numel() != 1 or int(patch_size.detach().cpu().item()) <= 0:
        raise ValueError("patch_size must contain one positive integer")
    patch_size = patch_size.reshape(())

    labels = labels.to(device=features.device, dtype=torch.long).reshape(-1)
    if labels.numel() != 1:
        raise ValueError(f"Each WSI bag must have one label, got shape {tuple(labels.shape)}")
    return features.float(), coords.long(), patch_size, labels


class VariableBagReservoir:
    """Seeded reservoir for CPU-only, variable-length WSI bags."""

    STATE_VERSION = 1

    def __init__(
        self,
        capacity: int,
        *,
        max_patches: int,
        seed: int,
        feature_dim: int = 768,
    ) -> None:
        if int(capacity) < 0:
            raise ValueError("Replay capacity must be non-negative")
        if int(max_patches) < 0:
            raise ValueError("max_patches must be non-negative")
        self.capacity = int(capacity)
        self.max_patches = int(max_patches)
        self.feature_dim = int(feature_dim)
        self.num_seen_examples = 0
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))
        self._entries: List[ReplayBag] = []

    def __len__(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return not self._entries

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(int(entry.label.item()) for entry in self._entries)

    def _reservoir_index(self) -> int:
        if self.capacity == 0:
            return -1
        if self.num_seen_examples < self.capacity:
            return self.num_seen_examples
        candidate = int(
            torch.randint(
                low=0,
                high=self.num_seen_examples + 1,
                size=(1,),
                generator=self._generator,
            ).item()
        )
        return candidate if candidate < self.capacity else -1

    def _cpu_item(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
        label: torch.Tensor,
    ) -> ReplayBag:
        features, coords, patch_size, label = unpack_prepared_batch(
            (features, coords, patch_size, label),
            feature_dim=self.feature_dim,
        )
        features = features.detach().to(device="cpu", dtype=torch.float32).clone()
        coords = coords.detach().to(device="cpu", dtype=torch.long).clone()
        patch_size = patch_size.detach().to(device="cpu", dtype=torch.long).clone()
        label = label.detach().to(device="cpu", dtype=torch.long).clone()

        if self.max_patches and features.shape[0] > self.max_patches:
            indices = torch.randperm(
                features.shape[0], generator=self._generator
            )[: self.max_patches].sort().values
            features = features.index_select(0, indices)
            coords = coords.index_select(0, indices)
        return ReplayBag(-1, features, coords, patch_size, label)

    def add(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
        label: torch.Tensor,
    ) -> int:
        """Process one stream item and return its slot, or ``-1`` if skipped."""

        item = self._cpu_item(features, coords, patch_size, label)
        index = self._reservoir_index()
        self.num_seen_examples += 1
        if index < 0:
            return -1
        stored = ReplayBag(index, item.features, item.coords, item.patch_size, item.label)
        if index == len(self._entries):
            self._entries.append(stored)
        else:
            self._entries[index] = stored
        return index

    def sample(self, size: int, device: torch.device) -> List[ReplayBag]:
        if not self._entries or int(size) <= 0:
            return []
        count = min(int(size), len(self._entries))
        indices = torch.randperm(
            len(self._entries), generator=self._generator
        )[:count].tolist()
        return [self._entries[index].to(device) for index in indices]

    def all(self, device: torch.device) -> List[ReplayBag]:
        return [entry.to(device) for entry in self._entries]

    def label_counts(self, num_classes: int) -> torch.Tensor:
        counts = torch.zeros(int(num_classes), dtype=torch.long)
        for label in self.labels:
            if label < 0 or label >= int(num_classes):
                raise ValueError(f"Replay label {label} is outside [0,{int(num_classes) - 1}]")
            counts[label] += 1
        return counts

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "max_patches": self.max_patches,
            "feature_dim": self.feature_dim,
            "num_seen_examples": self.num_seen_examples,
            "generator_state": self._generator.get_state().clone(),
            "entries": [
                {
                    "features": entry.features.clone(),
                    "coords": entry.coords.clone(),
                    "patch_size": entry.patch_size.clone(),
                    "label": entry.label.clone(),
                }
                for entry in self._entries
            ],
        }

    def load_state_dict(self, state: Dict[str, Any], *, strict: bool = True) -> None:
        if not isinstance(state, dict):
            raise TypeError("Replay state must be a dictionary")
        expected = {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "max_patches": self.max_patches,
            "feature_dim": self.feature_dim,
        }
        for key, value in expected.items():
            if state.get(key) != value and strict:
                raise ValueError(
                    f"Replay checkpoint mismatch for {key}: "
                    f"saved={state.get(key)!r}, expected={value!r}"
                )
        entries = state.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Replay checkpoint is missing its entries list")
        if len(entries) > self.capacity:
            raise ValueError("Replay checkpoint contains more entries than the configured capacity")

        restored: List[ReplayBag] = []
        for index, raw in enumerate(entries):
            if not isinstance(raw, dict):
                raise ValueError("Replay checkpoint contains a malformed entry")
            item = self._cpu_item(
                raw.get("features"),
                raw.get("coords"),
                raw.get("patch_size"),
                raw.get("label"),
            )
            # Loading must not re-sample an already capped upstream entry.
            restored.append(
                ReplayBag(index, item.features, item.coords, item.patch_size, item.label)
            )
        num_seen = int(state.get("num_seen_examples", len(restored)))
        if num_seen < len(restored):
            raise ValueError("Replay num_seen_examples is smaller than its stored entry count")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("Replay checkpoint is missing its generator state")
        self._entries = restored
        self.num_seen_examples = num_seen
        self._generator.set_state(generator_state.detach().cpu())

