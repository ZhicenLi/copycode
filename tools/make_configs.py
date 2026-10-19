"""Generate the parameter grid: 3 client counts x 2 alphas x 2 local epochs.

    python tools/make_configs.py

Writes `configs/generated/<dataset>_k<K>_a<alpha>_e<E>.json` for CIFAR-10 and
CIFAR-100 (24 files). Tiny-ImageNet is intentionally excluded until its loader
lands.
"""

from __future__ import annotations

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "configs", "generated")

BASE = {
    "data": {
        "data_root": "/root/autodl-tmp/data",
        "public_ratio": 0.1,
        "split_seed": 42,
        "noise_seed": 2026,
    },
    "model": {"fedds_pretrained": False},
    "fedcorr": {
        "local_ep": 5,
        "local_bs": 100,
        "lr": 0.03,
        "momentum": 0.5,
        "beta": 5.0,
        "mixup": True,
        "lid_k": 20,
        "relabel_ratio": 0.5,
        "confidence_thres": 0.5,
        "train_transform": "train",
        "loss_on": "logit",
    },
    "fedds": {
        "rounds": 200,
        "frac": 0.1,
        "local_bs": 64,
        "lr": 0.01,
        "momentum": 0.9,
        "public_bs": 64,
        "last_k": 10,
        "ds_max_iter": 500,
        "ds_tol": 1e-05,
        "ds_use_class_prior": False,
    },
    "runtime": {
        "device": "cuda:0",
        "num_workers": 2,
        "seed": 1,
        "save_every": 20,
        "resume": True,
        "reuse_correction": True,
        "output_root": "outputs",
    },
}

DATASETS = {
    "cifar10": {"iteration1": 5},
    "cifar100": {"iteration1": 10},
}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    written = 0
    for dataset, fedcorr_overrides in DATASETS.items():
        for num_clients in (10, 50, 100):
            for alpha in (0.5, 10.0):
                for local_ep in (1, 5):
                    cfg = copy.deepcopy(BASE)
                    cfg["data"].update(
                        dataset=dataset,
                        num_clients=num_clients,
                        dirichlet_alpha=alpha,
                    )
                    cfg["fedcorr"].update(fedcorr_overrides)
                    cfg["fedds"].update(local_ep=local_ep)
                    cfg["runtime"]["exp_name"] = (
                        f"{dataset}_k{num_clients}_a{alpha:g}_e{local_ep}"
                    )
                    name = f"{cfg['runtime']['exp_name']}.json"
                    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
                        json.dump(cfg, fh, indent=2, ensure_ascii=False)
                    written += 1
    print(f"wrote {written} configs to {OUT}")


if __name__ == "__main__":
    main()
