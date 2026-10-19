"""Tests for the ground-truth diagnostics used in the paper's analysis section."""

from __future__ import annotations

import unittest

import numpy as np

from fusion.analysis import (
    correction_metrics,
    identification_metrics,
    summarise,
    true_client_noise,
)
from fusion.data.manifest import Manifest


def _manifest(clean, noisy, client_id) -> Manifest:
    clean = np.asarray(clean, dtype=np.int64)
    noisy = np.asarray(noisy, dtype=np.int64)
    client_id = np.asarray(client_id, dtype=np.int32)
    return Manifest(
        sample_id=np.arange(len(clean), dtype=np.int32),
        client_id=client_id,
        clean_label=clean,
        noisy_label=noisy,
        corrected_label=noisy.copy(),
        noise_rate=np.zeros(len(clean), dtype=np.float32),
        split=np.where(client_id < 0, "public", "client").astype("U6"),
        meta={"num_clients": int(client_id.max()) + 1, "num_classes": 10},
    )


class TestCorrectionMetrics(unittest.TestCase):
    def test_counts_fixed_and_broken(self):
        clean = list(range(10))
        noisy = list(range(10))
        noisy[7] = 0                      # one label was wrong
        corrected = list(noisy)
        corrected[7] = 7                  # fixed
        corrected[0] = 5                  # broke a correct one

        metrics = correction_metrics(_manifest(clean, noisy, [0] * 10), np.array(corrected))

        self.assertEqual(metrics["labels_changed"], 2)
        self.assertEqual(metrics["fixed"], 1)
        self.assertEqual(metrics["broken"], 1)
        self.assertEqual(metrics["noisy_labels_total"], 1)
        self.assertAlmostEqual(metrics["correction_precision"], 0.5)
        self.assertAlmostEqual(metrics["correction_recall"], 1.0)
        self.assertAlmostEqual(metrics["label_accuracy_before"], 0.9)
        self.assertAlmostEqual(metrics["label_accuracy_after"], 0.9)

    def test_no_changes_is_not_division_by_zero(self):
        clean = [0, 1, 2]
        metrics = correction_metrics(_manifest(clean, clean, [0, 0, 0]), np.array(clean))
        self.assertEqual(metrics["labels_changed"], 0)
        self.assertTrue(np.isnan(metrics["correction_precision"]))


class TestIdentificationMetrics(unittest.TestCase):
    def test_confusion_counts(self):
        estimated = [0.6, 0.5, 0.0, 0.0]
        truth = [0.6, 0.0, 0.5, 0.0]
        metrics = identification_metrics(estimated, truth)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["client_precision"], 0.5)
        self.assertAlmostEqual(metrics["client_recall"], 0.5)
        self.assertAlmostEqual(metrics["client_accuracy"], 0.5)

    def test_perfect_ranking(self):
        metrics = identification_metrics([0.9, 0.5, 0.0, 0.0], [0.9, 0.4, 0.0, 0.0])
        self.assertAlmostEqual(metrics["spearman_estimated_vs_true"], 1.0)
        self.assertAlmostEqual(metrics["mean_absolute_error"], 0.025)


class TestSummarise(unittest.TestCase):
    def test_true_noise_uses_realised_ratio(self):
        clean = [0, 0, 1, 1, 2, 2]
        noisy = [0, 9, 1, 1, 2, 2]        # client 0 has 1/2 wrong
        manifest = _manifest(clean, noisy, [0, 0, 1, 1, 2, 2])
        rates = true_client_noise(manifest)
        np.testing.assert_allclose(rates, [0.5, 0.0, 0.0])

    def test_summary_is_json_serialisable(self):
        import json

        clean = [0, 1, 2, 3]
        noisy = [0, 1, 2, 9]
        manifest = _manifest(clean, noisy, [0, 0, 1, 1])
        corrected = np.array([0, 1, 2, 3])
        summary = summarise(
            manifest, corrected,
            [{"estimated_noise": [0.5, 0.0]}],
        )
        json.dumps(summary)                      # must not raise
        self.assertEqual(summary["clients"], 2)
        self.assertEqual(summary["clients_with_noise"], 1)
        self.assertEqual(len(summary["per_client"]), 2)


class TestSpearmanTies(unittest.TestCase):
    def test_matches_scipy_when_values_tie(self):
        """Clean clients all report 0.0, so ties are the common case."""
        from scipy.stats import spearmanr

        from fusion.core.dawid_skene import spearman

        rng = np.random.default_rng(0)
        for _ in range(5):
            a = rng.integers(0, 4, 20).astype(float)
            b = rng.integers(0, 4, 20).astype(float)
            self.assertAlmostEqual(spearman(a, b), float(spearmanr(a, b).statistic), places=10)


if __name__ == "__main__":
    unittest.main()
