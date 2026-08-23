"""Conditional compute-precision policy for WSI training."""

from __future__ import annotations

from contextlib import nullcontext

import torch


PRECISIONS = ("auto", "fp32", "fp16", "bf16")


def resolve_precision_name(args) -> str:
    requested = str(getattr(args, "precision", "auto")).lower()
    if requested not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {requested!r}")
    if requested == "auto":
        resolved = "fp16" if str(getattr(args, "backbone", "")).lower() == "gigapath" else "fp32"
    else:
        resolved = requested
    args.resolved_precision = resolved
    return resolved


def validate_precision_configuration(args) -> str:
    resolved = resolve_precision_name(args)
    if str(getattr(args, "backbone", "")).lower() == "gigapath" and resolved == "fp32":
        raise ValueError(
            "GigaPath fp32 compute is unsupported because the CUDA FlashAttention "
            "path in the pinned implementation requires FP16 or BF16"
        )
    return resolved


class PrecisionPolicy:
    """Own autocast and loss scaling without touching the legacy FP32 path."""

    def __init__(self, args, device: torch.device):
        self.name = resolve_precision_name(args)
        self.device = torch.device(device)
        self.enabled = self.name != "fp32"
        if self.enabled and self.device.type != "cuda":
            raise RuntimeError(f"{self.name} precision requires a CUDA device")
        if self.name == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 precision requires a CUDA GPU with BF16 support")
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.name]
        self.scaler = (
            torch.cuda.amp.GradScaler(enabled=True) if self.name == "fp16" else None
        )

    @property
    def uses_scaler(self) -> bool:
        return self.scaler is not None and bool(self.scaler.is_enabled())

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.dtype)

    def backward(self, loss, **kwargs) -> None:
        if self.uses_scaler:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def step(self, optimizer) -> None:
        if self.uses_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def gradient_scale(self) -> float:
        return float(self.scaler.get_scale()) if self.uses_scaler else 1.0

    def state_dict(self):
        return self.scaler.state_dict() if self.uses_scaler else None

    def load_state_dict(self, state) -> None:
        if not self.uses_scaler:
            if state not in (None, {}):
                raise ValueError("FP32/BF16 precision does not accept GradScaler state")
            return
        if state is not None:
            self.scaler.load_state_dict(state)
