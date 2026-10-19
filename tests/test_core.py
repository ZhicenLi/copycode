"""CPU tests for the parts that do not need torch.

Run from the project root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import numpy as np
import unittest

from fusion.config import Config
from fusion.core.dawid_skene import dawid_skene, spearman
from fusion.core.gmm import split_two_components
from fusion.core.lid import lid_scores
from fusion.data.manifest import build_manifest, noise_report, validate
from fusion.data.specs import get_spec


def synthetic_annotators(accuracies, num_classes=10, num_items=1500, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, num_classes, num_items)
    columns = []
    for acc in accuracies:
        keep = rng.random(num_items) < acc
        wrong = (truth + rng.integers(1, num_classes, num_items)) % num_classes
        columns.append(np.where(keep, truth, wrong))
    return np.stack(columns, axis=1)


class TestDawidSkene(unittest.TestCase):
    def test_reliability_orders_with_accuracy(self):
        accuracies = [0.95, 0.80, 0.60, 0.40, 0.20]
        result = dawid_skene(synthetic_annotators(accuracies), num_classes=10)
        self.assertGreater(spearman(result.reliability, accuracies), 0.95)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertEqual(result.confusion.shape, (5, 10, 10))

    def test_single_annotator_carries_no_information(self):
        preds = synthetic_annotators([0.95, 0.30], num_items=400)
        scores = []
        for column in range(preds.shape[1]):
            scores.append(dawid_skene(preds[:, [column]], num_classes=10).reliability[0])
        self.assertAlmostEqual(scores[0], scores[1], places=6)

    def test_confusion_rows_are_probabilities(self):
        result = dawid_skene(synthetic_annotators([0.7, 0.5, 0.3]), num_classes=10)
        sums = result.confusion.sum(axis=2)
        np.testing.assert_allclose(sums[sums > 0], 1.0, rtol=1e-9)

    def test_class_prior_variant_runs(self):
        result = dawid_skene(
            synthetic_annotators([0.7, 0.5, 0.3]), num_classes=10, use_class_prior=True
        )
        self.assertTrue(np.all(np.isfinite(result.reliability)))


class TestGMM(unittest.TestCase):
    def test_separates_two_modes(self):
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(0.2, 0.02, 300), rng.normal(0.8, 0.02, 200)])
        split = split_two_components(values)
        self.assertLess(split.means[split.clean_component], split.means.max())
        labels = split.labels
        self.assertGreater(np.mean(labels[:300] == split.clean_component), 0.95)
        self.assertLess(np.mean(labels[300:] == split.clean_component), 0.05)


class TestLID(unittest.TestCase):
    def test_scores_are_finite(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(60, 10))
        scores = lid_scores(x, k=10)
        self.assertEqual(scores.shape, (60,))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_lower_dimensional_manifold_has_lower_lid(self):
        rng = np.random.default_rng(3)
        t = np.linspace(0.0, 1.0, 80)
        line = np.stack([t] + [np.zeros_like(t)] * 7, axis=1)
        line = line + rng.normal(scale=1e-3, size=line.shape)
        ball = rng.normal(size=(80, 8))
        self.assertLess(np.mean(lid_scores(line, k=10)), np.mean(lid_scores(ball, k=10)))


class TestManifest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.labels = rng.integers(0, 10, 6000)
        self.spec = get_spec("cifar10")
        self.manifest = build_manifest(
            clean_labels=self.labels,
            spec=self.spec,
            num_clients=10,
            dirichlet_alpha=0.5,
            public_ratio=0.1,
            noise_levels=[0.1, 0.2, 0.4, 0.8],
            split_seed=42,
            noise_seed=7,
        )

    def test_conservation(self):
        self.assertEqual(validate(self.manifest, strict=True), [])
        public = self.manifest.public_indices()
        # stratified 10% of ~6000 samples, so allow rounding across the ten classes
        self.assertAlmostEqual(len(public), 600, delta=10)

    def test_clients_are_disjoint_and_cover_the_rest(self):
        client_ids = self.manifest.client_id
        self.assertTrue(np.all(client_ids[self.manifest.split == "client"] >= 0))
        self.assertTrue(np.all(client_ids[self.manifest.split == "public"] == -1))
        covered = np.concatenate(
            [self.manifest.client_indices(c) for c in range(10)]
        )
        expected = len(self.manifest.sample_id) - len(self.manifest.public_indices())
        self.assertEqual(len(np.unique(covered)), expected)

    def test_noise_is_reproducible_and_matches_request(self):
        again = build_manifest(
            clean_labels=self.labels,
            spec=self.spec,
            num_clients=10,
            dirichlet_alpha=0.5,
            public_ratio=0.1,
            noise_levels=[0.1, 0.2, 0.4, 0.8],
            split_seed=42,
            noise_seed=7,
        )
        np.testing.assert_array_equal(self.manifest.noisy_label, again.noisy_label)
        for row in noise_report(self.manifest):
            self.assertLess(abs(row["requested_noise"] - row["realised_noise"]), 0.02)

    def test_labels_never_flip_to_themselves(self):
        changed = self.manifest.noisy_label != self.manifest.clean_label
        self.assertTrue(np.all(self.manifest.noisy_label[changed] != self.manifest.clean_label[changed]))


class TestConfig(unittest.TestCase):
    def base(self, **overrides):
        raw = {
            "data": {"dataset": "cifar10", "num_clients": 10, "dirichlet_alpha": 0.5},
            "runtime": {"arms": ["fedavg"]},
        }
        raw.update(overrides)
        return raw

    def test_defaults_resolve_from_spec(self):
        cfg = Config.from_dict(self.base())
        self.assertEqual(cfg.data.num_classes, 10)
        self.assertEqual(cfg.model.fedcorr, "resnet18")
        self.assertEqual(cfg.model.fedds, "cnncifar")
        self.assertEqual(cfg.annotators_per_round(), 3)   # K=10 -> frac 0.3

    def test_annotator_floor_per_client_count(self):
        cfg = Config.from_dict(self.base())
        self.assertEqual(cfg.fedds.clients_per_round(10), 3)
        self.assertEqual(cfg.fedds.clients_per_round(50), 5)
        self.assertEqual(cfg.fedds.clients_per_round(100), 10)

    def test_wrong_frac1_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict(self.base(fedcorr={"frac1": 0.01}))

    def test_unimplemented_dataset_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            Config.from_dict(self.base(data={"dataset": "tinyimagenet", "num_clients": 10}))

    def test_empty_arms_are_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict(self.base(runtime={"arms": []}))

    def test_clients_per_round_cannot_exceed_clients(self):
        with self.assertRaises(ValueError):
            Config.from_dict(self.base(fedds={"frac": 2.0}))

    def test_weighted_average_rejects_length_mismatch(self):
        import torch

        from fusion.stages.fedds_train import weighted_average

        states = [{"w": torch.zeros(2)}, {"w": torch.ones(2)}]
        with self.assertRaises(ValueError):
            weighted_average(states, np.array([1.0]))

    def test_weighted_average_ignores_negative_weights(self):
        import torch

        from fusion.stages.fedds_train import weighted_average

        states = [{"w": torch.zeros(2)}, {"w": torch.ones(2)}]
        averaged = weighted_average(states, np.array([1.0, -5.0]))
        torch.testing.assert_close(averaged["w"], torch.zeros(2))

    def test_duplicate_arms_are_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict(self.base(runtime={"arms": ["fedds", "fedds"]}))

    def test_path_like_experiment_name_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict(self.base(runtime={"arms": ["fedds"], "exp_name": "../escape"}))


if __name__ == "__main__":
    unittest.main()
