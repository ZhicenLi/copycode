"""Synthetic end-to-end check for the FedCorr stage (CPU, no dataset needed)."""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), ".pylibs"))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from fusion.config import Config
from fusion.data.loaders import build_transform, make_loader, synthetic_source
from fusion.data.manifest import build_manifest
from fusion.stages.fedcorr_preprocess import FedCorrPreprocessor


def main() -> int:
    started = time.time()
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
            "iteration1": 2,
            "local_ep": 1,
            "local_bs": 8,
            "beta": 0.0,
            "mixup": False,
            "lid_k": 5,
        },
        "fedds": {"rounds": 2},
        "runtime": {"arms": ["fedavg"], "seed": 1, "save_every": 0},
    }
    cfg = Config.from_dict(raw)
    spec = cfg.data.spec()
    print(f"config ok: {cfg.data.num_clients} clients, "
          f"{cfg.annotators_per_round()} annotators/round")

    images, labels = synthetic_source(120, spec.num_classes, spec.image_size, seed=0)
    manifest = build_manifest(
        clean_labels=labels,
        spec=spec,
        num_clients=cfg.data.num_clients,
        dirichlet_alpha=cfg.data.dirichlet_alpha,
        public_ratio=cfg.data.public_ratio,
        noise_levels=cfg.data.noise_levels,
        split_seed=cfg.data.split_seed,
        noise_seed=cfg.data.noise_seed,
        min_client_samples=cfg.data.min_client_samples,
    )
    print(f"manifest ok: public={len(manifest.public_indices())}, "
          f"sizes={[len(manifest.client_indices(c)) for c in range(cfg.data.num_clients)]}")

    test_images, test_labels = synthetic_source(48, spec.num_classes, spec.image_size, seed=1)
    test_loader = make_loader(
        test_images, test_labels, np.arange(48),
        batch_size=32, train=False, transform=build_transform(spec, "eval"),
    )

    out = os.path.join(ROOT, "outputs", "_smoke_fedcorr")
    pre = FedCorrPreprocessor(
        cfg, manifest, images, test_loader, torch.device("cpu"), out
    )
    result = pre.run()

    changed = int(np.sum(result.corrected_labels != manifest.noisy_label))
    print(f"fedcorr done: corrected={changed} samples, "
          f"iterations={len(result.per_iteration)}, clients={len(result.per_client)}")
    for record in result.per_iteration:
        print("  ", {k: record[k] for k in
                     ("iteration", "noisy_clients", "corrected_samples", "test_accuracy")})
    print(f"artifacts -> {out}")
    print(f"elapsed {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
