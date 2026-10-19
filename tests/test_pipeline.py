"""End-to-end pipeline check on synthetic data (CPU).

Marked slow-ish (~20 s): it exercises FedCorr, the label hand-off and all four
experiment arms, so a green run means the code is actually wired together.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestPipelineSmoke(unittest.TestCase):
    def setUp(self):
        # A cached manifest from a previous version cannot be verified against the
        # current checksum algorithm, so these tests must start from a clean state.
        for name in ("_unit_pipeline", "_unit_repro", "_unit_arms"):
            shutil.rmtree(os.path.join("outputs", name), ignore_errors=True)

    def test_four_arms_share_one_correction(self):
        from fusion.config import Config
        from fusion.pipeline import run_experiment

        raw = {
            "data": {
                "dataset": "cifar10",
                "num_clients": 3,
                "dirichlet_alpha": 10.0,
                "public_ratio": 0.2,
                "min_client_samples": 5,
                "noise_levels": [0.3, 0.6, 1.0],
            },
            "fedcorr": {
                "iteration1": 1,
                "local_ep": 1,
                "local_bs": 8,
                "beta": 0.0,
                "mixup": False,
                "lid_k": 5,
            },
            "fedds": {"rounds": 1, "local_ep": 1, "local_bs": 8, "last_k": 1, "frac": 1.0},
            "runtime": {
                "device": "cpu",
                "num_workers": 0,
                "seed": 1,
                "save_every": 0,
                "synthetic": True,
                "checkpoint_models": False,
                "output_root": "outputs",
                "exp_name": "_unit_pipeline",
            },
        }
        cfg = Config.from_dict(raw)
        payload = run_experiment(cfg)

        run_dir = cfg.run_dir()
        self.assertTrue(os.path.exists(os.path.join(run_dir, "summary.json")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "data", "manifest.npz")))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "fedcorr", "corrected_labels.npy")))

        with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(len(summary["results"]), 4)
        arms = {row["arm"] for row in summary["results"]}
        self.assertEqual(arms, {"fedavg", "fedds", "fedcorr+fedavg", "fedcorr+fedds"})
        self.assertEqual(summary["annotators_per_round"], 3)   # K=3 with frac=1.0

        corrected = np.load(os.path.join(run_dir, "fedcorr", "corrected_labels.npy"))
        manifest_labels = np.load(os.path.join(run_dir, "data", "manifest.npz"))["noisy_label"]
        self.assertEqual(corrected.shape, manifest_labels.shape)

    def test_same_seed_reproduces_the_same_accuracy(self):
        """A rerun with the same seed must reproduce the numbers exactly."""
        from fusion.config import Config
        from fusion.pipeline import run_experiment

        raw = {
            "data": {
                "dataset": "cifar10",
                "num_clients": 3,
                "dirichlet_alpha": 10.0,
                "public_ratio": 0.2,
                "min_client_samples": 5,
                "noise_levels": [0.6],
            },
            "fedcorr": {
                "iteration1": 1, "local_ep": 1, "local_bs": 8,
                "beta": 0.0, "mixup": False, "lid_k": 5,
            },
            "fedds": {"rounds": 1, "local_ep": 1, "local_bs": 8, "last_k": 1, "frac": 1.0},
            "runtime": {
                "device": "cpu", "num_workers": 0, "seed": 3, "save_every": 0,
                "synthetic": True, "checkpoint_models": False,
                "output_root": "outputs", "exp_name": "_unit_repro",
                "arms": ["fedcorr+fedds"],
            },
        }
        first = run_experiment(Config.from_dict(raw))
        second = run_experiment(Config.from_dict(raw))
        self.assertAlmostEqual(
            first["results"][0]["last1_accuracy"],
            second["results"][0]["last1_accuracy"],
            places=6,
            msg="same seed produced different accuracies; seeding is incomplete",
        )


if __name__ == "__main__":
    unittest.main()
