"""Semantic invariants for the FedDS stage.

The first test guards a bug that is invisible in accuracy numbers: if the global
model is overwritten while collecting client predictions, the next client starts
from the previous client's weights and the whole round becomes a chain instead
of federated averaging.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _fingerprint(model: torch.nn.Module) -> float:
    with torch.no_grad():
        return float(sum(float(parameter.sum()) for parameter in model.parameters()))


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFedDSSemantics(unittest.TestCase):
    def _build(self, seed: int = 1):
        from fusion.config import Config
        from fusion.data.loaders import build_transform, make_cached_loader, synthetic_source
        from fusion.data.manifest import build_manifest
        from fusion.models.registry import build_model
        from fusion.stages.fedds_train import FedDSTrainer

        raw = {
            "data": {
                "dataset": "cifar10",
                "num_clients": 3,
                "dirichlet_alpha": 10.0,
                "public_ratio": 0.2,
                "min_client_samples": 5,
                "noise_levels": [0.5],
            },
            "fedds": {"rounds": 1, "local_ep": 1, "local_bs": 8, "last_k": 1, "frac": 1.0},
            "runtime": {
                "device": "cpu",
                "seed": seed,
                "save_every": 0,
                "checkpoint_models": False,
                "output_root": "outputs",
                "exp_name": "_semantics",
                "arms": ["fedds"],
            },
        }
        cfg = Config.from_dict(raw)
        spec = cfg.data.spec()
        images, labels = synthetic_source(60, spec.num_classes, spec.image_size, seed=0)
        manifest = build_manifest(
            clean_labels=labels,
            spec=spec,
            num_clients=3,
            dirichlet_alpha=10.0,
            public_ratio=0.2,
            noise_levels=[0.5],
            split_seed=42,
            noise_seed=7,
            min_client_samples=5,
        )
        test_images, test_labels = synthetic_source(20, spec.num_classes, spec.image_size, seed=1)
        test_loader, _ = make_cached_loader(
            test_images, test_labels, np.arange(20), build_transform(spec, "eval"), 16
        )
        public_loader, _ = make_cached_loader(
            images, np.zeros(len(images), dtype=np.int64), manifest.public_indices(),
            build_transform(spec, "eval"), 16,
        )
        from fusion.pipeline import seed_everything

        seed_everything(seed)          # mirrors what the pipeline does before each arm
        trainer = FedDSTrainer(
            cfg=cfg,
            manifest=manifest,
            train_images=images,
            test_loader=test_loader,
            public_loader=public_loader,
            labels=manifest.corrected_label,
            device=torch.device("cpu"),
            output_dir=os.path.join("outputs", "_semantics"),
            aggregate="fedds",
            tag="semantics",
        )
        return trainer, build_model

    def test_clients_start_from_the_round_global_model(self):
        trainer, _ = self._build()
        starts = []
        original = trainer._local_train

        def spy(indices, seed, lr):
            starts.append(_fingerprint(trainer.local))
            return original(indices, seed, lr)

        trainer._local_train = spy
        before = _fingerprint(trainer.model)
        trainer.run()

        self.assertEqual(len(starts), 3)
        for value in starts:
            self.assertAlmostEqual(
                value, before, places=4,
                msg="a client did not start from the round-global model",
            )

    def test_global_model_is_not_mutated_while_collecting_predictions(self):
        trainer, _ = self._build()
        seen = []
        original = trainer._public_predictions

        def spy(model):
            seen.append(_fingerprint(trainer.model))
            return original(model)

        trainer._public_predictions = spy
        before = _fingerprint(trainer.model)
        trainer.run()

        self.assertEqual(len(seen), 3)
        for value in seen:
            self.assertAlmostEqual(
                value, before, places=4,
                msg="the global model changed during prediction collection",
            )

    def test_arms_share_initial_weights_for_the_same_seed(self):
        """A 2x2 comparison must not differ by initialisation noise."""
        first, _ = self._build(seed=1)
        second, _ = self._build(seed=1)
        different, _ = self._build(seed=2)

        self.assertEqual(first.initial_state_sha, second.initial_state_sha)
        self.assertNotEqual(first.initial_state_sha, different.initial_state_sha)


if __name__ == "__main__":
    unittest.main()
