import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.atlas_distribution import (
    FrozenDistributionHead, deterministic_spherical_kmeans,
)
from models.atlas_v2 import AtlasV2Network


def normalized_samples(center, count, seed):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(
        torch.as_tensor(center).float() + 0.15 * torch.randn(
            count, len(center), generator=generator
        ),
        dim=1,
    )


class DistributionMetricTests(unittest.TestCase):
    def test_diagonal_scale_normalization_is_class_scale_invariant(self):
        head = FrozenDistributionHead(2, 5, [0, 1], "diag", rank=4, clusters=2)
        head.fit_class(0, normalized_samples([1, 0, 0, 0, 0], 9, 1))
        head.fit_class(1, normalized_samples([0, 1, 0, 0, 0], 9, 2))
        query = normalized_samples([1, 0, 0, 0, 0], 4, 3)
        before, _ = head.diagonal_distance(query, 0.0)
        head.class_var[0].mul_(17.0)
        after, _ = head.diagonal_distance(query, 0.0)
        self.assertTrue(torch.allclose(before[:, 0], after[:, 0], atol=1e-5))

    def test_lowrank_projection_matches_explicit_trace_normalized_inverse(self):
        head = FrozenDistributionHead(2, 5, [0, 1], "lowrank", rank=2, clusters=2)
        head.fit_class(0, normalized_samples([1, 0, 0, 0, 0], 12, 4))
        head.fit_class(1, normalized_samples([0, 1, 0, 0, 0], 12, 5))
        query = normalized_samples([0.7, 0.3, 0, 0, 0], 3, 6)
        rho, rank = 0.25, 2
        actual = head.lowrank_distance(query, rho, rank)[:, 0]

        basis = head.lowrank_basis[0, :rank]
        eigenvalues = head.lowrank_eigenvalues[0, :rank]
        residual = (
            head.class_var[0].sum() - head.lowrank_eigenvalues[0, :rank].sum()
        ) / (5 - rank)
        pooled = head.pooled_var.mean()
        covariance_lr = (
            basis.t() @ torch.diag(eigenvalues - residual) @ basis
            + residual * torch.eye(5)
        )
        covariance = (1 - rho) * covariance_lr + rho * pooled * torch.eye(5)
        covariance = covariance / (covariance.trace() / 5)
        difference = query - head.class_mean[0]
        expected = torch.einsum(
            "ni,ij,nj->n", difference, torch.linalg.inv(covariance), difference
        ) / 5
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-4))

    def test_rank_fallback_and_zero_variance_are_finite(self):
        head = FrozenDistributionHead(1, 4, [0], "lowrank", rank=8, clusters=3)
        head.fit_class(0, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        self.assertEqual(int(head.lowrank_rank[0]), 0)
        distance = head.lowrank_distance(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 0.5, 8)
        self.assertTrue(torch.isfinite(distance).all())

    def test_task_centroid_is_equal_class_mean_not_sample_weighted(self):
        head = FrozenDistributionHead(2, 4, [0, 0], "task_centroid", clusters=2)
        head.fit_class(0, normalized_samples([1, 0, 0, 0], 30, 7))
        head.fit_class(1, normalized_samples([0, 1, 0, 0], 3, 8))
        expected = F.normalize(head.class_mean[:2].mean(0), dim=0)
        self.assertTrue(torch.allclose(head.task_centroids[0], expected, atol=1e-6))


class DeterminismAndProtocolTests(unittest.TestCase):
    def test_distribution_head_does_not_enter_legacy_prototype_keyspace(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(4, 2)

            def get_classifier(self):
                return self.classifier

        legacy = AtlasV2Network(Backbone(), 2, 4, True, None, 0.5)
        legacy_keys = set(legacy.state_dict())
        self.assertIn("prototype_bank", legacy_keys)
        self.assertIn("prototype_valid", legacy_keys)
        self.assertFalse(any("class_mean" in key for key in legacy_keys))

        distribution = FrozenDistributionHead(2, 4, [0, 1], "diag", clusters=2)
        modern = AtlasV2Network(
            Backbone(), 2, 4, True, None, 0.5,
            distribution_head=distribution,
        )
        modern_keys = set(modern.state_dict())
        self.assertNotIn("prototype_bank", modern_keys)
        self.assertNotIn("prototype_valid", modern_keys)
        self.assertIn("distribution_head.class_mean", modern_keys)

    def test_spherical_kmeans_is_deterministic(self):
        values = normalized_samples([1, 0, 0, 0], 20, 9)
        first = deterministic_spherical_kmeans(values, 3)
        second = deterministic_spherical_kmeans(values, 3)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_selector_rejects_test_cache_and_has_non_cartesian_counts(self):
        head = FrozenDistributionHead(2, 4, [0, 1], "diag_shrink", clusters=2)
        x0 = normalized_samples([1, 0, 0, 0], 6, 10)
        x1 = normalized_samples([0, 1, 0, 0], 6, 11)
        head.fit_class(0, x0)
        head.fit_class(1, x1)
        values, labels = torch.cat((x0, x1)), torch.tensor([0] * 6 + [1] * 6)
        with self.assertRaisesRegex(ValueError, "validation"):
            head.select(values, labels, "diag", split="test")
        self.assertEqual(head.select(values, labels, "diag")["candidate_count"], 5)
        self.assertEqual(head.select(values, labels, "diag_shrink")["candidate_count"], 20)
        self.assertEqual(head.select(values, labels, "lowrank")["candidate_count"], 45)
        self.assertEqual(head.select(values, labels, "multi")["candidate_count"], 6)
        self.assertEqual(head.select(values, labels, "task_lme")["candidate_count"], 12)

    def test_pseudo_features_are_renormalized_and_not_persisted(self):
        head = FrozenDistributionHead(2, 4, [0, 1], "pt_only", clusters=2, seed=13)
        old = normalized_samples([1, 0, 0, 0], 7, 12)
        current = normalized_samples([0, 1, 0, 0], 7, 13)
        head.fit_class(0, old)
        head.fit_class(1, current)
        samples, _ = head.prototype_training_samples({1: current}, task=1, samples_per_class=16)
        self.assertTrue(torch.allclose(samples.norm(dim=1), torch.ones(32), atol=1e-6))
        self.assertFalse(any("embedding" in key for key in head.state_dict()))
        means_before = head.class_mean.clone()
        head.tune_offsets(
            {1: current}, task=1, steps=3, samples_per_class=8,
            learning_rate=0.05,
        )
        self.assertTrue(torch.equal(head.class_mean, means_before))
        self.assertGreater(float(head.class_offset.norm()), 0.0)
        self.assertFalse(head.class_offset.requires_grad)
        self.assertTrue(all(buffer.grad is None for buffer in head.buffers()))

    def test_inference_is_bitwise_repeatable_and_state_read_only(self):
        head = FrozenDistributionHead(2, 4, [0, 1], "atlas_tf", clusters=2)
        head.fit_class(0, normalized_samples([1, 0, 0, 0], 8, 14))
        head.fit_class(1, normalized_samples([0, 1, 0, 0], 8, 15))
        query = normalized_samples([0.5, 0.5, 0, 0], 5, 16)
        before = copy.deepcopy(head.state_dict())
        first, second = head.scores(query), head.scores(query)
        self.assertTrue(torch.equal(first, second))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]), key)

    def test_task_lme_does_not_change_within_task_ranking(self):
        head = FrozenDistributionHead(4, 4, [0, 0, 1, 1], "task_lme", clusters=2)
        centers = torch.eye(4)
        for label in range(4):
            head.fit_class(label, normalized_samples(centers[label], 7, 20 + label))
        query = normalized_samples([0.8, 0.2, 0, 0], 6, 30)
        head.mode = "diag_shrink"
        base = head.scores(query)
        head.mode = "task_lme"
        calibrated = head.scores(query)
        self.assertTrue(torch.equal(base[:, :2].argmax(1), calibrated[:, :2].argmax(1)))


class RanPACTests(unittest.TestCase):
    def test_streaming_gram_and_targets_equal_batch_computation_on_raw_features(self):
        head = FrozenDistributionHead(
            3, 4, [0, 0, 1], "ranpac", ranpac_dim=7, seed=17
        )
        raw = torch.randn(11, 4, generator=torch.Generator().manual_seed(18)) * 3.0
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        head.update_ranpac(raw[:5], labels[:5])
        head.update_ranpac(raw[5:], labels[5:])
        hidden = F.relu(raw @ head.ranpac_projection)
        targets = F.one_hot(labels, 3).float()
        self.assertTrue(torch.allclose(head.ranpac_gram, hidden.t() @ hidden))
        self.assertTrue(torch.allclose(head.ranpac_targets, hidden.t() @ targets))
        head.solve_ranpac(1.0, 7)
        expected_weight = torch.linalg.solve(
            hidden.t() @ hidden + torch.eye(7), hidden.t() @ targets
        )
        self.assertTrue(torch.allclose(head.ranpac_weight, expected_weight, atol=1e-5))
        normalized_hidden = F.relu(F.normalize(raw, dim=1) @ head.ranpac_projection)
        self.assertFalse(torch.allclose(head.ranpac_gram, normalized_hidden.t() @ normalized_hidden))
        normalized_hidden = F.relu(F.normalize(raw, dim=1) @ head.ranpac_projection)
        self.assertFalse(torch.allclose(head.ranpac_gram, normalized_hidden.t() @ normalized_hidden))


if __name__ == "__main__":
    unittest.main()
