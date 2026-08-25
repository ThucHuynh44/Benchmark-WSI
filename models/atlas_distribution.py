"""Train-only distribution classifiers for the frozen ATLAS-v2 suite.

The module is deliberately independent from the legacy NCM/OAS implementation.
All persistent tensors are sufficient statistics or analytic classifier state;
slide embeddings are accepted transiently by ``fit_*`` methods and are never
registered in the module or returned by ``state_dict``.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DISTRIBUTION_STATE_VERSION = 1
MAX_LOWRANK_RANK = 8
MAX_SUB_PROTOTYPES = 3
DISTRIBUTION_MODES = (
    "diag", "diag_shrink", "lowrank", "task_centroid", "task_lme",
    "multi", "pt_only", "atlas_tf", "atlas_pt", "ranpac",
)


def _normalize(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(values, dim=dim, eps=1.0e-8)


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro recall over labels present in an isolated validation cache."""

    predictions = logits.argmax(1)
    recalls = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        recalls.append((predictions[mask] == label).float().mean())
    return float(torch.stack(recalls).mean()) if recalls else float("-inf")


def deterministic_spherical_kmeans(
    embeddings: torch.Tensor, clusters: int, iterations: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-iteration spherical k-means with deterministic farthest seeding."""

    values = _normalize(embeddings.detach().float().reshape(embeddings.shape[0], -1))
    if values.shape[0] == 0:
        raise ValueError("spherical k-means requires at least one embedding")
    clusters = max(1, min(int(clusters), int(values.shape[0])))
    selected = [0]
    while len(selected) < clusters:
        similarity = values @ values[selected].t()
        nearest_distance = 1.0 - similarity.max(1).values
        nearest_distance[selected] = -1.0
        selected.append(int(nearest_distance.argmax()))
    centers = values[selected].clone()
    assignments = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(int(iterations)):
        assignments = (values @ centers.t()).argmax(1)
        updated = []
        nearest_similarity = (values @ centers.t()).max(1).values
        for index in range(clusters):
            members = values[assignments == index]
            if members.shape[0]:
                updated.append(_normalize(members.mean(0), dim=0))
            else:
                replacement = int(nearest_similarity.argmin())
                updated.append(values[replacement])
                nearest_similarity[replacement] = 1.0
        centers = torch.stack(updated)
    assignments = (values @ centers.t()).argmax(1)
    occupancy = torch.bincount(assignments, minlength=clusters)
    return centers, occupancy


class FrozenDistributionHead(nn.Module):
    """Frozen class distributions and optional train-only prototype offsets."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        class_task: Sequence[int],
        mode: str,
        *,
        seed: int = 0,
        alpha: float = 0.1,
        rho: float = 0.5,
        rank: int = 4,
        clusters: int = 2,
        tau_multi: float = 0.1,
        beta: float = 0.25,
        tau_task: float = 0.1,
        ranpac_dim: int = 2000,
        ranpac_ridge: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in DISTRIBUTION_MODES:
            raise ValueError(f"Unknown ATLAS distribution mode {mode!r}")
        if len(class_task) != int(num_classes):
            raise ValueError("class_task must contain one task id per class")
        self.mode = str(mode)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.num_tasks = max(map(int, class_task)) + 1
        self.seed = int(seed)

        self.register_buffer("distribution_state_version", torch.tensor(DISTRIBUTION_STATE_VERSION))
        self.register_buffer("class_task", torch.as_tensor(class_task, dtype=torch.long))
        self.register_buffer("class_mean", torch.zeros(num_classes, embedding_dim))
        self.register_buffer("class_var", torch.zeros(num_classes, embedding_dim))
        self.register_buffer("class_count", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("pooled_var", torch.zeros(embedding_dim))
        self.register_buffer("lowrank_basis", torch.zeros(num_classes, MAX_LOWRANK_RANK, embedding_dim))
        self.register_buffer("lowrank_eigenvalues", torch.zeros(num_classes, MAX_LOWRANK_RANK))
        self.register_buffer("lowrank_residual", torch.zeros(num_classes))
        self.register_buffer("lowrank_rank", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("sub_prototypes", torch.zeros(num_classes, MAX_SUB_PROTOTYPES, embedding_dim))
        self.register_buffer("sub_prototype_count", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("sub_prototype_occupancy", torch.zeros(num_classes, MAX_SUB_PROTOTYPES, dtype=torch.long))
        self.register_buffer("task_centroids", torch.zeros(self.num_tasks, embedding_dim))
        self.register_buffer("task_valid", torch.zeros(self.num_tasks, dtype=torch.bool))
        self.register_buffer("variance_floor", torch.tensor(torch.finfo(torch.float32).eps))
        self.register_buffer("selected_alpha", torch.tensor(float(alpha)))
        self.register_buffer("selected_rho", torch.tensor(float(rho)))
        self.register_buffer("selected_rank", torch.tensor(int(rank), dtype=torch.long))
        self.register_buffer("selected_clusters", torch.tensor(int(clusters), dtype=torch.long))
        self.register_buffer("selected_tau_multi", torch.tensor(float(tau_multi)))
        self.register_buffer("selected_beta", torch.tensor(float(beta)))
        self.register_buffer("selected_tau_task", torch.tensor(float(tau_task)))
        self.register_buffer("selected_ranpac_ridge", torch.tensor(float(ranpac_ridge)))
        self.register_buffer("selected_ranpac_dim", torch.tensor(int(ranpac_dim)))
        self.register_buffer("candidate_count", torch.zeros((), dtype=torch.long))
        self.class_offset = nn.Parameter(torch.zeros(num_classes, embedding_dim), requires_grad=False)

        self.ranpac_dim = int(ranpac_dim) if mode == "ranpac" else 0
        if mode == "ranpac":
            generator = torch.Generator().manual_seed(self.seed + 7919)
            projection = torch.randn(embedding_dim, self.ranpac_dim, generator=generator)
            self.register_buffer("ranpac_projection", projection)
            self.register_buffer("ranpac_gram", torch.zeros(self.ranpac_dim, self.ranpac_dim))
            self.register_buffer("ranpac_targets", torch.zeros(self.ranpac_dim, num_classes))
            self.register_buffer("ranpac_weight", torch.zeros(self.ranpac_dim, num_classes))

    @property
    def seen_mask(self) -> torch.Tensor:
        return self.class_count > 0

    @torch.no_grad()
    def fit_class(self, label: int, embeddings_norm: torch.Tensor) -> None:
        label = int(label)
        values = _normalize(
            embeddings_norm.detach().float().reshape(-1, self.embedding_dim)
        ).to(self.class_mean)
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("Cannot fit an empty/non-finite class distribution")
        count = int(values.shape[0])
        mean = values.mean(0)
        centered = values - mean
        variance = centered.square().sum(0) / float(max(count - 1, 1))
        self.class_mean[label].copy_(mean)
        self.class_var[label].copy_(variance)
        self.class_count[label] = count
        self.lowrank_basis[label].zero_()
        self.lowrank_eigenvalues[label].zero_()
        self.lowrank_residual[label] = 0
        self.lowrank_rank[label] = 0
        self.sub_prototypes[label].zero_()
        self.sub_prototype_occupancy[label].zero_()

        effective_rank = min(MAX_LOWRANK_RANK, count - 1, self.embedding_dim - 1)
        if effective_rank > 0 and count > 1:
            # SVD of [N,D] avoids materializing the D-by-D covariance.
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            eigenvalues = singular_values.square() / float(count - 1)
            self.lowrank_basis[label, :effective_rank].copy_(vh[:effective_rank])
            self.lowrank_eigenvalues[label, :effective_rank].copy_(eigenvalues[:effective_rank])
            trace = variance.sum()
            residual = (trace - eigenvalues[:effective_rank].sum()) / float(
                self.embedding_dim - effective_rank
            )
            self.lowrank_residual[label] = residual.clamp_min(0.0)
            self.lowrank_rank[label] = effective_rank

        requested_clusters = min(int(self.selected_clusters), MAX_SUB_PROTOTYPES, count)
        centers, occupancy = deterministic_spherical_kmeans(values, requested_clusters)
        k = int(centers.shape[0])
        self.sub_prototypes[label, :k].copy_(centers)
        self.sub_prototype_occupancy[label, :k].copy_(occupancy)
        self.sub_prototype_count[label] = k
        self._refresh_pooled_statistics()
        self._refresh_task_centroids()

    @torch.no_grad()
    def refit_subprototypes(
        self, embeddings_by_class: Mapping[int, torch.Tensor], clusters: int,
    ) -> None:
        """Refit only current train classes for a fold-local K candidate."""

        clusters = int(clusters)
        if clusters not in (2, 3):
            raise ValueError("Multi-prototype K must be 2 or 3")
        for raw_label, embeddings in embeddings_by_class.items():
            label = int(raw_label)
            values = _normalize(
                embeddings.detach().float().reshape(-1, self.embedding_dim)
            ).to(self.class_mean)
            effective = min(clusters, int(values.shape[0]))
            centers, occupancy = deterministic_spherical_kmeans(values, effective)
            self.sub_prototypes[label].zero_()
            self.sub_prototype_occupancy[label].zero_()
            self.sub_prototypes[label, :effective].copy_(centers)
            self.sub_prototype_occupancy[label, :effective].copy_(occupancy)
            self.sub_prototype_count[label] = effective

    @torch.no_grad()
    def _refresh_pooled_statistics(self) -> None:
        degrees = (self.class_count - 1).clamp_min(0).float()
        total = degrees.sum()
        pooled = (
            (self.class_var * degrees[:, None]).sum(0) / total
            if float(total) > 0 else torch.ones_like(self.pooled_var)
        )
        positive = pooled[pooled > 0]
        scale = positive.median() if positive.numel() else pooled.new_tensor(1.0)
        floor = (scale * 1.0e-6).clamp_min(torch.finfo(pooled.dtype).eps)
        self.pooled_var.copy_(pooled.clamp_min(floor))
        self.variance_floor.copy_(floor)

    @torch.no_grad()
    def _refresh_task_centroids(self) -> None:
        for task in range(self.num_tasks):
            mask = (self.class_task == task) & self.seen_mask
            if bool(mask.any()):
                # Equal-class mean: class sample counts never enter this expression.
                centroid = _normalize(self.class_mean[mask].mean(0), dim=0)
                self.task_centroids[task].copy_(centroid)
                self.task_valid[task] = True

    def _offset_means(self) -> torch.Tensor:
        return _normalize(self.class_mean + self.class_offset)

    def cosine_scores(self, embeddings_norm: torch.Tensor) -> torch.Tensor:
        z = _normalize(embeddings_norm)
        return z @ self._offset_means().t()

    def diagonal_distance(
        self, embeddings_norm: torch.Tensor, rho: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = _normalize(embeddings_norm)
        rho = torch.as_tensor(rho, dtype=z.dtype, device=z.device)
        variance = (1.0 - rho) * self.class_var + rho * self.pooled_var
        variance = variance.clamp_min(self.variance_floor)
        normalized = variance / variance.mean(1, keepdim=True).clamp_min(self.variance_floor)
        differences = z[:, None, :] - self.class_mean[None, :, :]
        distance = (differences.square() / normalized[None, :, :]).mean(2)
        return distance, normalized

    def lowrank_distance(
        self,
        embeddings_norm: torch.Tensor,
        rho: float | torch.Tensor,
        requested_rank: int | torch.Tensor,
    ) -> torch.Tensor:
        z = _normalize(embeddings_norm)
        rho_value = float(torch.as_tensor(rho))
        requested = int(torch.as_tensor(requested_rank))
        pooled_isotropic = self.pooled_var.mean().clamp_min(self.variance_floor)
        distances = []
        for label in range(self.num_classes):
            available = int(self.lowrank_rank[label])
            rank = min(requested, available)
            difference = z - self.class_mean[label]
            residual_empirical = (
                (
                    self.class_var[label].sum()
                    - self.lowrank_eigenvalues[label, :rank].sum()
                ) / float(self.embedding_dim - rank)
                if rank else self.class_var[label].mean()
            ).clamp_min(self.variance_floor)
            base = ((1.0 - rho_value) * residual_empirical + rho_value * pooled_isotropic).clamp_min(self.variance_floor)
            if rank:
                basis = self.lowrank_basis[label, :rank]
                retained = ((1.0 - rho_value) * self.lowrank_eigenvalues[label, :rank] + rho_value * pooled_isotropic).clamp_min(self.variance_floor)
                trace = retained.sum() + float(self.embedding_dim - rank) * base
                scale = (trace / float(self.embedding_dim)).clamp_min(self.variance_floor)
                retained, normalized_base = retained / scale, base / scale
                projection = difference @ basis.t()
                projected_norm = projection.square().sum(1)
                quadratic = (projection.square() / retained).sum(1)
                orthogonal = (difference.square().sum(1) - projected_norm).clamp_min(0.0)
                quadratic = quadratic + orthogonal / normalized_base
            else:
                # Isotropic covariance becomes identity after trace normalization.
                quadratic = difference.square().sum(1)
            distances.append(quadratic / float(self.embedding_dim))
        return torch.stack(distances, dim=1)

    def multi_scores(self, embeddings_norm: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
        z = _normalize(embeddings_norm)
        temperature = torch.as_tensor(tau, dtype=z.dtype, device=z.device).clamp_min(1.0e-6)
        rows = []
        for label in range(self.num_classes):
            count = min(
                int(self.sub_prototype_count[label]), int(self.selected_clusters)
            )
            prototypes = _normalize(
                self.sub_prototypes[label, :count]
                + self.class_offset[label].unsqueeze(0)
            )
            similarity = z @ prototypes.t() if count else z.new_full((z.shape[0], 1), float("-inf"))
            rows.append(temperature * torch.logsumexp(similarity / temperature, dim=1))
        return torch.stack(rows, dim=1)

    def _task_calibrate(
        self, embeddings_norm: torch.Tensor, scores: torch.Tensor,
        beta: float, tau: float, centroid: bool,
    ) -> torch.Tensor:
        additions = scores.new_zeros(scores.shape[0], self.num_tasks)
        if centroid:
            additions = _normalize(embeddings_norm) @ self.task_centroids.t()
        else:
            temperature = max(float(tau), 1.0e-6)
            for task in range(self.num_tasks):
                mask = (self.class_task == task) & self.seen_mask
                if bool(mask.any()):
                    additions[:, task] = temperature * (
                        torch.logsumexp(scores[:, mask] / temperature, dim=1)
                        - torch.log(scores.new_tensor(float(mask.sum())))
                    )
        return scores + float(beta) * additions[:, self.class_task]

    def soft_task_scores_from_class_scores(
        self, scores: torch.Tensor, tau: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute oracle-free class-count-normalized task LME diagnostics."""

        temperature = max(float(self.selected_tau_task if tau is None else tau), 1.0e-6)
        values = scores.new_full((scores.shape[0], self.num_tasks), float("-inf"))
        for task in range(self.num_tasks):
            mask = (self.class_task == task) & self.seen_mask
            if bool(mask.any()):
                values[:, task] = temperature * (
                    torch.logsumexp(scores[:, mask] / temperature, dim=1)
                    - torch.log(scores.new_tensor(float(mask.sum())))
                )
        return values

    def _base_scores(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized embeddings and class scores before task calibration."""

        if self.mode == "ranpac":
            dimension = int(self.selected_ranpac_dim)
            raw = embeddings.float()
            hidden = F.relu(raw @ self.ranpac_projection[:, :dimension])
            return _normalize(raw), hidden @ self.ranpac_weight[:dimension]
        z = _normalize(embeddings.float())
        alpha = float(self.selected_alpha)
        rho = float(self.selected_rho)
        rank = int(self.selected_rank)
        if self.mode == "multi":
            result = self.multi_scores(z, self.selected_tau_multi)
        elif self.mode in {"lowrank", "atlas_tf", "atlas_pt"}:
            cosine = (
                self.multi_scores(z, self.selected_tau_multi)
                if self.mode in {"atlas_tf", "atlas_pt"} else self.cosine_scores(z)
            )
            result = cosine - alpha * float(self.embedding_dim) * self.lowrank_distance(z, rho, rank)
        else:
            distance, _ = self.diagonal_distance(z, rho if self.mode != "diag" else 0.0)
            result = self.cosine_scores(z) - alpha * float(self.embedding_dim) * distance
        return z, result

    def scores_and_task_scores(
        self, embeddings: torch.Tensor, seen_classes: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, base = self._base_scores(embeddings)
        task_scores = self.soft_task_scores_from_class_scores(base)
        result = base
        if self.mode == "task_centroid":
            result = self._task_calibrate(
                z, result, float(self.selected_beta),
                float(self.selected_tau_task), True,
            )
        elif self.mode in {"task_lme", "atlas_tf", "atlas_pt"}:
            result = result + float(self.selected_beta) * task_scores[:, self.class_task]
        valid = self.seen_mask
        result = result.masked_fill(~valid.unsqueeze(0), float("-inf"))
        if seen_classes is not None and int(seen_classes) < self.num_classes:
            result = result.clone()
            result[:, int(seen_classes):] = float("-inf")
        return result, task_scores

    def scores(self, embeddings: torch.Tensor, seen_classes: int | None = None) -> torch.Tensor:
        return self.scores_and_task_scores(embeddings, seen_classes)[0]

    @torch.no_grad()
    def update_ranpac(self, embeddings_raw: torch.Tensor, labels: torch.Tensor) -> None:
        if self.mode != "ranpac":
            raise RuntimeError("RanPAC state is disabled")
        hidden = F.relu(embeddings_raw.detach().float().to(self.ranpac_projection) @ self.ranpac_projection)
        targets = F.one_hot(labels.long().to(hidden.device), self.num_classes).float()
        self.ranpac_gram.add_(hidden.t() @ hidden)
        self.ranpac_targets.add_(hidden.t() @ targets)

    @torch.no_grad()
    def solve_ranpac(
        self, ridge: float | None = None, projection_dim: int | None = None,
    ) -> None:
        ridge = float(self.selected_ranpac_ridge if ridge is None else ridge)
        dimension = int(
            self.selected_ranpac_dim if projection_dim is None else projection_dim
        )
        if not 0 < dimension <= self.ranpac_dim:
            raise ValueError("RanPAC selected projection dimension is invalid")
        identity = torch.eye(dimension, device=self.ranpac_gram.device, dtype=self.ranpac_gram.dtype)
        system = self.ranpac_gram[:dimension, :dimension] + ridge * identity
        weight, info = torch.linalg.solve_ex(system, self.ranpac_targets[:dimension])
        if int(info.max()) != 0 or not torch.isfinite(weight).all():
            raise FloatingPointError(
                f"RanPAC solve failed for projection_dim={dimension}, ridge={ridge:g}"
            )
        self.ranpac_weight.zero_()
        self.ranpac_weight[:dimension].copy_(weight)
        self.selected_ranpac_ridge.fill_(ridge)
        self.selected_ranpac_dim.fill_(dimension)

    @torch.no_grad()
    def select(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        stage: str | None = None,
        *,
        split: str = "validation",
    ) -> Dict[str, float | int]:
        """Deterministic fold/task-local staged selection on validation only."""

        if str(split) != "validation":
            raise ValueError("ATLAS staged selection accepts validation caches only")
        stage = self.mode if stage is None else str(stage)
        grids: Mapping[str, Sequence[Mapping[str, float | int]]]
        alpha = (0.01, 0.03, 0.1, 0.3, 1.0)
        rho_diag = (0.25, 0.5, 0.75, 1.0)
        rho_lr = (0.25, 0.5, 0.75)
        if stage == "diag":
            candidates = [dict(alpha=a, rho=0.0) for a in alpha]
        elif stage == "diag_shrink":
            candidates = [dict(alpha=a, rho=r) for a, r in product(alpha, rho_diag)]
        elif stage == "lowrank":
            candidates = [dict(rank=k, rho=r, alpha=a) for k, r, a in product((2, 4, 8), rho_lr, alpha)]
        elif stage == "multi":
            candidates = [dict(clusters=k, tau_multi=t) for k, t in product((2, 3), (0.05, 0.1, 0.2))]
        elif stage == "task_centroid":
            candidates = [dict(beta=b) for b in (0.1, 0.25, 0.5, 1.0)]
        elif stage in {"task_lme", "atlas_tf"}:
            candidates = [dict(beta=b, tau_task=t) for b, t in product((0.1, 0.25, 0.5, 1.0), (0.05, 0.1, 0.2))]
        elif stage == "ranpac":
            dimensions = tuple(dict.fromkeys(
                min(value, self.ranpac_dim) for value in (2000, 5000, 10000)
            ))
            candidates = [
                dict(ranpac_dim=dimension, ranpac_ridge=10.0 ** exponent)
                for dimension, exponent in product(dimensions, range(-8, 9))
            ]
        else:
            return {}

        original_mode = self.mode
        selection_mode = stage
        if stage in {"task_centroid", "task_lme"}:
            selection_mode = stage
        best_key, best = None, None
        for index, candidate in enumerate(candidates):
            if "alpha" in candidate:
                self.selected_alpha.fill_(float(candidate["alpha"]))
            if "rho" in candidate:
                self.selected_rho.fill_(float(candidate["rho"]))
            if "rank" in candidate:
                self.selected_rank.fill_(int(candidate["rank"]))
            if "clusters" in candidate:
                self.selected_clusters.fill_(int(candidate["clusters"]))
            if "tau_multi" in candidate:
                self.selected_tau_multi.fill_(float(candidate["tau_multi"]))
            if "beta" in candidate:
                self.selected_beta.fill_(float(candidate["beta"]))
            if "tau_task" in candidate:
                self.selected_tau_task.fill_(float(candidate["tau_task"]))
            if stage == "ranpac":
                try:
                    self.solve_ranpac(
                        float(candidate["ranpac_ridge"]), int(candidate["ranpac_dim"])
                    )
                except FloatingPointError:
                    continue
            self.mode = selection_mode
            metric = balanced_accuracy(self.scores(embeddings), labels)
            key = (metric, -index)  # stable grid order resolves exact ties
            if best_key is None or key > best_key:
                best_key, best = key, dict(candidate)
        self.mode = original_mode
        if best is None:
            raise FloatingPointError(f"Every {stage} validation candidate failed")
        for key, value in (best or {}).items():
            buffer = getattr(self, "selected_" + key)
            buffer.fill_(value)
        if stage == "ranpac" and best is not None:
            self.solve_ranpac(
                float(best["ranpac_ridge"]), int(best["ranpac_dim"])
            )
        self.candidate_count.add_(len(candidates))
        return {**(best or {}), "validation_bacc": float(best_key[0]), "candidate_count": len(candidates)}

    def prototype_training_samples(
        self, current_embeddings: Mapping[int, torch.Tensor], *, task: int,
        samples_per_class: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic real/pseudo unit vectors without retaining them."""
        seen_labels = torch.nonzero(self.seen_mask, as_tuple=False).flatten().tolist()
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + 104729 * (int(task) + 1)
        )
        samples, targets = [], []
        for label in seen_labels:
            if label in current_embeddings:
                values = _normalize(current_embeddings[label].detach().float().cpu())
                if values.shape[0] >= samples_per_class:
                    indices = torch.randperm(values.shape[0], generator=generator)[:samples_per_class]
                else:
                    indices = torch.randint(values.shape[0], (samples_per_class,), generator=generator)
                draws = values[indices]
            else:
                mean = self.class_mean[label].detach().float().cpu()
                noise = torch.randn(samples_per_class, self.embedding_dim, generator=generator)
                if self.mode == "atlas_pt" and int(self.lowrank_rank[label]) > 0:
                    rank = min(int(self.selected_rank), int(self.lowrank_rank[label]))
                    basis = self.lowrank_basis[label, :rank].detach().float().cpu()
                    eigenvalues = self.lowrank_eigenvalues[label, :rank].detach().float().cpu()
                    residual = (
                        (
                            self.class_var[label].sum().detach().float().cpu()
                            - eigenvalues.sum()
                        ) / float(self.embedding_dim - rank)
                    ).clamp_min(self.variance_floor.cpu())
                    coefficients = torch.randn(samples_per_class, rank, generator=generator)
                    lowrank_scale = (eigenvalues - residual).clamp_min(0.0).sqrt()
                    draws = (
                        mean + noise * residual.sqrt()
                        + (coefficients * lowrank_scale) @ basis
                    )
                else:
                    std = self.class_var[label].detach().float().cpu().clamp_min(self.variance_floor.cpu()).sqrt()
                    draws = mean + noise * std
            # Statistics were fitted on the sphere; pseudo samples must return to it.
            draws = _normalize(draws).to(self.class_mean.device)
            samples.append(draws)
            targets.append(torch.full((samples_per_class,), label, dtype=torch.long, device=draws.device))
        return torch.cat(samples), torch.cat(targets)

    def tune_offsets(
        self,
        current_embeddings: Mapping[int, torch.Tensor],
        *,
        task: int,
        steps: int = 200,
        learning_rate: float = 0.05,
        samples_per_class: int = 32,
        temperature: float = 0.1,
        anchor_weight: float = 0.1,
        margin_weight: float = 0.0,
        margin: float = 0.2,
    ) -> None:
        """Tune only class offsets from current real and old pseudo features."""

        seen_labels = torch.nonzero(self.seen_mask, as_tuple=False).flatten().tolist()
        features, labels = self.prototype_training_samples(
            current_embeddings, task=task, samples_per_class=samples_per_class
        )

        self.class_offset.requires_grad_(True)
        optimizer = torch.optim.Adam([self.class_offset], lr=float(learning_rate))
        anchors = self.class_mean.detach().clone()
        original_mode = self.mode
        for _ in range(int(steps)):
            logits = self.scores(features) / float(temperature)
            ce = F.cross_entropy(logits, labels)
            anchor = (self.class_offset[self.seen_mask].square().sum(1)).mean()
            if float(margin_weight) > 0 and len(seen_labels) > 1:
                prototypes = self._offset_means()[self.seen_mask]
                similarity = prototypes @ prototypes.t()
                off_diagonal = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
                margin_loss = F.relu(similarity[off_diagonal] - (1.0 - float(margin))).mean()
            else:
                margin_loss = ce * 0.0
            loss = ce + float(anchor_weight) * anchor + float(margin_weight) * margin_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Only the offset parameter participates in this optimizer.
            optimizer.step()
        self.mode = original_mode
        self.class_offset.requires_grad_(False)
        if not torch.equal(self.class_mean, anchors):
            raise RuntimeError("Prototype tuning modified empirical class means")

    def distribution_memory_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.state_dict().values())

    def diagnostics(self) -> Dict[str, float | int]:
        mask = self.seen_mask
        traces_before = self.class_var[mask].sum(1)
        _, normalized = self.diagonal_distance(self.class_mean[mask], self.selected_rho) if bool(mask.any()) else (None, None)
        return {
            "covariance_trace_before": float(traces_before.mean()) if traces_before.numel() else 0.0,
            "covariance_trace_after": float(normalized[mask].sum(1).mean()) if normalized is not None else 0.0,
            "effective_lowrank_rank": float(self.lowrank_rank[mask].float().mean()) if bool(mask.any()) else 0.0,
            "sub_prototype_occupancy_min": int(self.sub_prototype_occupancy[self.sub_prototype_occupancy > 0].min()) if bool((self.sub_prototype_occupancy > 0).any()) else 0,
            "prototype_offset_norm": float(self.class_offset[mask].norm(dim=1).mean()) if bool(mask.any()) else 0.0,
            "distribution_memory_bytes": self.distribution_memory_bytes(),
        }


def class_to_task(class_offsets: Sequence[int], task_num_classes: Sequence[int]) -> list[int]:
    mapping = []
    for task, (offset, count) in enumerate(zip(class_offsets, task_num_classes)):
        if int(offset) != len(mapping):
            raise ValueError("Tasks must form a contiguous class layout")
        mapping.extend([task] * int(count))
    return mapping
