"""FedDS resume must reproduce the uninterrupted run exactly.

Because every round derives its own seed from the round index, continuing from
a checkpoint should give the same numbers as never having stopped.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFedDSResume(unittest.TestCase):
    root = os.path.join("outputs", "_resume")

    def setUp(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _trainer(self, seed: int, rounds: int, name: str, resume: bool = False):
        from fusion.config import Config
        from fusion.data.loaders import build_transform, make_cached_loader, synthetic_source
        from fusion.data.manifest import build_manifest
        from fusion.pipeline import seed_everything
        from fusion.stages.fedds_train import FedDSTrainer

        cfg = Config.from_dict({
            "data": {"dataset": "cifar10", "num_clients": 3, "dirichlet_alpha": 10.0,
                     "public_ratio": 0.2, "min_client_samples": 5, "noise_levels": [0.5]},
            "fedds": {"rounds": rounds, "local_ep": 1, "local_bs": 8, "last_k": 1, "frac": 1.0},
            "runtime": {"device": "cpu", "seed": seed, "save_every": 0,
                        "checkpoint_models": False, "output_root": self.root,
                        "exp_name": name, "arms": ["fedds"], "resume": resume},
        })
        spec = cfg.data.spec()
        images, labels = synthetic_source(60, spec.num_classes, spec.image_size, seed=0)
        manifest = build_manifest(
            clean_labels=labels, spec=spec, num_clients=3, dirichlet_alpha=10.0,
            public_ratio=0.2, noise_levels=[0.5], split_seed=42, noise_seed=7,
            min_client_samples=5,
        )
        test_images, test_labels = synthetic_source(16, spec.num_classes, spec.image_size, seed=1)
        test_loader, _ = make_cached_loader(
            test_images, test_labels, np.arange(16), build_transform(spec, "eval"), 16
        )
        public_loader, _ = make_cached_loader(
            images, np.zeros(len(images), dtype=np.int64), manifest.public_indices(),
            build_transform(spec, "eval"), 16,
        )
        seed_everything(seed)                    # identical model init for both runs
        trainer = FedDSTrainer(
            cfg=cfg, manifest=manifest, train_images=images, test_loader=test_loader,
            public_loader=public_loader, labels=manifest.noisy_label,
            device=torch.device("cpu"), output_dir=os.path.join(self.root, name),
            aggregate="fedds", tag="t",
        )
        return trainer

    def test_resume_matches_an_uninterrupted_run(self):
        fresh = self._trainer(11, rounds=4, name="fresh")
        fresh.run()

        stopped = self._trainer(11, rounds=2, name="resumed", resume=True)
        stopped.run()

        # simulate a *process restart*: a brand new trainer object, same config,
        # which is the case the RNG-state handling has to survive
        continued = self._trainer(11, rounds=4, name="resumed", resume=True)
        continued.run()

        with open(os.path.join(self.root, "fresh", "t_summary.json"), encoding="utf-8") as fh:
            expected = json.load(fh)["accuracy_curve"]
        with open(os.path.join(self.root, "resumed", "t_summary.json"), encoding="utf-8") as fh:
            actual = json.load(fh)["accuracy_curve"]

        self.assertEqual(len(expected), 4)
        self.assertEqual(len(actual), 4)
        for a, b in zip(expected, actual):
            self.assertAlmostEqual(a, b, places=6)

    def test_changed_hyperparameters_do_not_resume(self):
        """Changing local_bs must start a fresh run instead of trusting the checkpoint."""
        trainer = self._trainer(11, rounds=2, name="changed", resume=True)
        trainer.run()

        trainer.cfg.fedds.local_bs = 16          # a real change to the training setup
        trainer.cfg.fedds.rounds = 2
        trainer.run()

        with open(os.path.join(self.root, "changed", "t_summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertFalse(summary["resumed"])
        self.assertEqual(len(summary["accuracy_curve"]), 2)


if __name__ == "__main__":
    unittest.main()
