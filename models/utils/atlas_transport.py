"""Deterministic feature-transport primitives for ATLAS-v3 ACL.

All matrices use the row-vector convention: a feature row ``x`` is mapped as
``x @ A``.  The helpers are deliberately stateless so fitted maps can be
discarded immediately after historical sufficient statistics are updated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class ResidualTransport:
    matrix: torch.Tensor
    delta: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    effective_rank: int
    diagnostics: Dict[str, float]

    def map(self, values: torch.Tensor, gates: torch.Tensor | None = None) -> torch.Tensor:
        rows = values.float().reshape(-1, self.matrix.shape[0])
        if gates is None:
            gates = torch.ones(rows.shape[0], device=rows.device, dtype=rows.dtype)
        gates = gates.to(rows).reshape(-1, 1)
        drift = (rows - self.source_mean.to(rows)) @ self.delta.to(rows)
        drift = drift + self.target_mean.to(rows) - self.source_mean.to(rows)
        return rows + gates * drift


def _pairs(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = source.detach().float().reshape(source.shape[0], -1)
    target = target.detach().float().reshape(target.shape[0], -1)
    if source.shape != target.shape or source.shape[0] == 0:
        raise ValueError("Transport requires non-empty paired [N,D] tensors")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Transport pairs contain NaN or Inf")
    return source, target


def _matrix_diagnostics(matrix: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = source @ matrix
    mse = F.mse_loss(prediction, target)
    singular = torch.linalg.svdvals(matrix.float())
    smallest = singular[-1].clamp_min(torch.finfo(singular.dtype).eps)
    return {
        "pair_train_mse": float(mse),
        "transport_condition_number": float(singular[0] / smallest),
        "transport_delta_norm": float(torch.linalg.matrix_norm(matrix - torch.eye(matrix.shape[0], device=matrix.device))),
    }


def fit_ldc(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Fit the learned no-bias linear projector used as the LDC control."""

    source, target = _pairs(source, target)
    dimensions = source.shape[1]
    with torch.enable_grad():
        projector = torch.nn.Linear(dimensions, dimensions, bias=False, device=source.device)
        with torch.no_grad():
            projector.weight.copy_(torch.eye(dimensions, device=source.device))
        optimizer = torch.optim.AdamW(projector.parameters(), lr=float(learning_rate), weight_decay=0.0)
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(projector(source), target)
            loss.backward()
            optimizer.step()
    # Linear stores y=x@weight.T; publish a row-convention matrix.
    matrix = projector.weight.detach().t().contiguous()
    return matrix, _matrix_diagnostics(matrix, source, target)


def fit_sldc(source: torch.Tensor, target: torch.Tensor, *, ridge: float) -> tuple[torch.Tensor, Dict[str, float]]:
    """Closed-form full ridge map regularized toward identity."""

    source, target = _pairs(source, target)
    dimensions = source.shape[1]
    identity = torch.eye(dimensions, device=source.device, dtype=source.dtype)
    scale = float(source.shape[0])
    lhs = source.t() @ source / scale + float(ridge) * identity
    rhs = source.t() @ target / scale + float(ridge) * identity
    matrix = torch.linalg.solve(lhs, rhs)
    return matrix, _matrix_diagnostics(matrix, source, target)


def fit_lowrank_residual(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rank: int,
    ridge: float,
) -> ResidualTransport:
    """Fit a centered ridge residual and truncate its matrix rank."""

    source, target = _pairs(source, target)
    dimensions = source.shape[1]
    source_mean = source.mean(0, keepdim=True)
    target_mean = target.mean(0, keepdim=True)
    centered = source - source_mean
    residual = (target - target_mean) - centered
    identity = torch.eye(dimensions, device=source.device, dtype=source.dtype)
    scale = float(source.shape[0])
    lhs = centered.t() @ centered / scale + float(ridge) * identity
    rhs = centered.t() @ residual / scale
    full_delta = torch.linalg.solve(lhs, rhs)
    available = min(int(rank), max(int(source.shape[0]) - 1, 0), dimensions)
    if available > 0:
        u, singular, vh = torch.linalg.svd(full_delta, full_matrices=False)
        tolerance = torch.finfo(singular.dtype).eps * max(full_delta.shape) * singular[0].clamp_min(1.0)
        effective = min(available, int((singular > tolerance).sum()))
        delta = (
            (u[:, :effective] * singular[:effective]) @ vh[:effective]
            if effective > 0
            else torch.zeros_like(full_delta)
        )
    else:
        effective = 0
        delta = torch.zeros_like(full_delta)
    matrix = identity + delta
    prediction = source + (source - source_mean) @ delta + target_mean - source_mean
    diagnostics = _matrix_diagnostics(matrix, source, target)
    diagnostics["pair_train_mse"] = float(F.mse_loss(prediction, target))
    diagnostics["effective_rank"] = float(effective)
    return ResidualTransport(matrix, delta, source_mean, target_mean, effective, diagnostics)


def support_basis(values: torch.Tensor, *, energy: float) -> torch.Tensor:
    values = values.detach().float().reshape(values.shape[0], -1)
    centered = values - values.mean(0, keepdim=True)
    if values.shape[0] < 2 or float(centered.square().sum()) <= torch.finfo(centered.dtype).eps:
        return centered.new_zeros((centered.shape[1], 0))
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    spectrum = singular.square()
    cumulative = spectrum.cumsum(0) / spectrum.sum().clamp_min(torch.finfo(spectrum.dtype).eps)
    count = int(torch.searchsorted(cumulative, cumulative.new_tensor(float(energy))).item()) + 1
    count = min(count, int((spectrum > torch.finfo(spectrum.dtype).eps).sum()))
    return vh[:count].t().contiguous()


def mean_coverage(points: torch.Tensor, source: torch.Tensor, *, energy: float) -> torch.Tensor:
    basis = support_basis(source, energy=energy)
    centered = points.float() - source.float().mean(0, keepdim=True)
    denominator = centered.square().sum(1).clamp_min(torch.finfo(centered.dtype).eps)
    if basis.shape[1] == 0:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    numerator = (centered @ basis.to(centered)).square().sum(1)
    return (numerator / denominator).clamp(0.0, 1.0)


def distribution_coverage(
    means: torch.Tensor,
    scatters: torch.Tensor,
    counts: torch.Tensor,
    source: torch.Tensor,
    *,
    energy: float,
) -> torch.Tensor:
    basis = support_basis(source, energy=energy)
    centered = means.float() - source.float().mean(0, keepdim=True)
    outputs = []
    for index in range(means.shape[0]):
        degrees = max(int(counts[index]) - 1, 1)
        covariance = scatters[index].float() / float(degrees)
        total = centered[index].square().sum() + covariance.trace().clamp_min(0.0)
        if basis.shape[1] == 0:
            supported = total.new_zeros(())
        else:
            q = basis.to(covariance)
            supported = (centered[index] @ q).square().sum()
            supported = supported + (q.t() @ covariance @ q).trace().clamp_min(0.0)
        outputs.append((supported / total.clamp_min(torch.finfo(total.dtype).eps)).clamp(0.0, 1.0))
    return torch.stack(outputs) if outputs else means.new_zeros((0,))


def bootstrap_gates(
    points: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    coverage: torch.Tensor,
    main: ResidualTransport,
    *,
    rank: int,
    ridge: float,
    samples: int,
    beta: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float | str]]:
    """Estimate WSI-level map disagreement and return conservative step gates."""

    source, target = _pairs(source, target)
    if source.shape[0] < 4 or int(samples) <= 0:
        uncertainty = torch.full_like(coverage, float("nan"))
        return coverage, uncertainty, {
            "bootstrap_status": "insufficient_samples",
            "bootstrap_valid_oob": 0.0,
            "bootstrap_oob_mse": "",
        }
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    mapped_main = F.normalize(main.map(points), dim=1, eps=1.0e-8)
    disagreements = []
    oob_losses = []
    n = source.shape[0]
    for _ in range(int(samples)):
        indices_cpu = torch.randint(n, (n,), generator=generator)
        indices = indices_cpu.to(source.device)
        fitted = fit_lowrank_residual(source[indices], target[indices], rank=rank, ridge=ridge)
        mapped = F.normalize(fitted.map(points), dim=1, eps=1.0e-8)
        disagreements.append(1.0 - (mapped * mapped_main).sum(1).clamp(-1.0, 1.0))
        selected = torch.zeros(n, dtype=torch.bool)
        selected[indices_cpu.unique()] = True
        oob_cpu = (~selected).nonzero(as_tuple=False).reshape(-1)
        if oob_cpu.numel():
            oob = oob_cpu.to(source.device)
            predicted = fitted.map(source[oob])
            oob_losses.append(F.mse_loss(predicted, target[oob]).detach())
    valid = len(oob_losses)
    if valid < 5:
        uncertainty = torch.full_like(coverage, float("nan"))
        return coverage, uncertainty, {
            "bootstrap_status": "insufficient_oob",
            "bootstrap_valid_oob": float(valid),
            "bootstrap_oob_mse": "",
        }
    uncertainty = torch.stack(disagreements).mean(0)
    gates = coverage * torch.exp(-float(beta) * uncertainty)
    return gates.clamp(0.0, 1.0), uncertainty, {
        "bootstrap_status": "ok",
        "bootstrap_valid_oob": float(valid),
        "bootstrap_oob_mse": float(torch.stack(oob_losses).mean()),
    }


def summarize(values: torch.Tensor, prefix: str) -> Dict[str, float | str]:
    finite = values.detach().float()[torch.isfinite(values)]
    if finite.numel() == 0:
        return {f"{prefix}_min": "", f"{prefix}_mean": "", f"{prefix}_max": ""}
    return {
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_max": float(finite.max()),
    }
