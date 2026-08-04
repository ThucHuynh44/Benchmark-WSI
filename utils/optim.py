"""Shared optimizer construction for controlled continual-learning runs."""

from __future__ import annotations

from typing import Iterable, Optional

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    args,
    *,
    lr: Optional[float] = None,
) -> torch.optim.AdamW:
    """Build the repository-wide AdamW optimizer.

    Keeping this in one place prevents method adapters from silently changing
    the optimizer and confounding continual-learning comparisons.
    """

    name = str(getattr(args, "optimizer", "adamw")).strip().lower()
    if name != "adamw":
        raise ValueError(f"Only optimizer='adamw' is supported, got {name!r}")

    learning_rate = float(getattr(args, "lr", 1.0e-5) if lr is None else lr)
    weight_decay = float(getattr(args, "optim_wd", 0.0))
    epsilon = float(getattr(args, "adam_eps", 1.0e-8))
    if learning_rate <= 0:
        raise ValueError(f"AdamW learning rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"AdamW weight decay must be non-negative, got {weight_decay}")
    if epsilon <= 0:
        raise ValueError(f"AdamW epsilon must be positive, got {epsilon}")

    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=epsilon,
    )
