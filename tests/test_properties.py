"""Randomised property tests for the invariants that matter most.

These sweep parameter combinations rather than checking one hand-picked case,
so edge cases (tiny clients, extreme alpha, all-same predictions) are exercised.
"""

from __future__ import annotations

import unittest

import numpy as np

from fusion.core.dawid_skene import dawid_skene, spearman
from fusion.core.gmm import split_two_components
from fusion.core.lid import lid_scores
from fusion.data.manifest import build_manifest, noise_report, validate
from fusion.data.specs import get_spec


class TestManifestProperties(unittest.TestCase):
    def test_conservation_across_parameter_sweep(self):
        spec = get_spec("cifar10")
        rng = np.random.default_rng(0)
        for trial in range(8):
            n = 4000
            labels = rng.integers(0, 10, n)
            num_clients = int(rng.choice([5, 10, 20, 50]))
            alpha = float(rng.choice([0.1, 0.5, 10.0]))
            ratio = float(rng.choice([0.05, 0.1, 0.2]))
            manifest = build_manifest(
                clean_labels=labels,
                spec=spec,
                num_clients=num_clients,
                dirichlet_alpha=alpha,
                public_ratio=ratio,
                noise_levels=[0.1, 0.5, 1.0],
                split_seed=int(rng.integers(0, 10_000)),
                noise_seed=int(rng.integers(0, 10_000)),
                min_client_samples=5,
            )
            with self.subTest(trial=trial, num_clients=num_clients, alpha=alpha):
                self.assertEqual(validate(manifest), [])
                self.assertEqual(
                    len(manifest.public_indices()) + len(np.where(manifest.split == "client")[0]),
                    n,
                )
                for row in noise_report(manifest):
                    # the official code truncates int(rate*n), so the realised ratio
                    # can undershoot by almost one sample; scale by client size
                    tolerance = 1.0 / max(row["samples"], 1) + 1e-6
                    self.assertLessEqual(
                        abs(row["realised_noise"] - row["requested_noise"]), tolerance,
                        msg=f"client {row['client_id']} with {row['samples']} samples",
                    )

    def test_every_label_flip_changes_the_class(self):
        spec = get_spec("cifar10")
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 10, 2000)
        manifest = build_manifest(
            clean_labels=labels, spec=spec, num_clients=10, dirichlet_alpha=0.5,
            public_ratio=0.1, noise_levels=[1.0], split_seed=1, noise_seed=2,
            min_client_samples=5,
        )
        changed = manifest.noisy_label != manifest.clean_label
        self.assertTrue(np.all(manifest.noisy_label[changed] != manifest.clean_label[changed]))


class TestDawidSkeneProperties(unittest.TestCase):
    def test_invariants_over_random_inputs(self):
        rng = np.random.default_rng(2)
        for trial in range(6):
            num_items = int(rng.integers(50, 300))
            annotators = int(rng.integers(2, 8))
            num_classes = int(rng.choice([3, 5, 10]))
            predictions = rng.integers(0, num_classes, (num_items, annotators))
            result = dawid_skene(predictions, num_classes=num_classes, max_iter=50)
            with self.subTest(trial=trial):
                self.assertEqual(result.confusion.shape, (annotators, num_classes, num_classes))
                self.assertTrue(np.all(result.reliability >= 0.0))
                self.assertTrue(np.all(result.reliability <= 1.0))
                self.assertTrue(np.all(result.estimated >= 0))
                self.assertTrue(np.all(result.estimated < num_classes))
                np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, rtol=1e-9)
                rows = result.confusion.sum(axis=2)
                nonzero = rows[rows > 0]
                np.testing.assert_allclose(nonzero, 1.0, rtol=1e-9)

    def test_rejects_out_of_range_predictions(self):
        with self.assertRaises(ValueError):
            dawid_skene(np.array([[0], [10]]), num_classes=10)

    def test_spearman_constant_input_is_nan(self):
        self.assertTrue(np.isnan(spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))))


class TestGMMProperties(unittest.TestCase):
    def test_handles_constant_and_bimodal(self):
        constant = split_two_components(np.zeros(20))
        self.assertEqual(constant.backend, "constant")
        self.assertFalse(np.any(constant.labels != constant.clean_component))

        rng = np.random.default_rng(3)
        values = np.concatenate([rng.normal(0.1, 0.01, 200), rng.normal(0.9, 0.01, 100)])
        split = split_two_components(values)
        self.assertLess(split.means[split.clean_component], split.means.max())
        self.assertGreater(np.mean(split.labels[:200] == split.clean_component), 0.95)


class TestLIDProperties(unittest.TestCase):
    def test_shape_and_finiteness_over_sizes(self):
        rng = np.random.default_rng(4)
        for n, k, d in [(2, 1, 3), (5, 2, 4), (40, 20, 10), (600, 20, 10)]:
            with self.subTest(n=n):
                features = rng.random((n, d)).astype(np.float32)
                scores = lid_scores(features, k=k)
                self.assertEqual(scores.shape, (n,))
                self.assertTrue(np.all(np.isfinite(scores)))

    def test_single_point_returns_zero(self):
        np.testing.assert_allclose(lid_scores(np.zeros((1, 5))), [0.0])


if __name__ == "__main__":
    unittest.main()
