#!/usr/bin/env python
"""Entry point.

    python run.py configs/cifar10_k100_a05_e5.json
    python run.py configs/cifar10_k100_a05_e5.json --fedds.rounds=5 --runtime.exp_name=debug
"""

from __future__ import annotations

import sys

from fusion.config import Config, parse_cli
from fusion.pipeline import run_experiment


def main() -> int:
    path, overrides = parse_cli(sys.argv[1:])
    cfg = Config.from_file(path, overrides)
    print(f"experiment: {cfg.runtime.exp_name}")
    print(f"  dataset={cfg.data.dataset} clients={cfg.data.num_clients} "
          f"alpha={cfg.data.dirichlet_alpha} local_ep={cfg.fedds.local_ep}")
    print(f"  annotators/round={cfg.annotators_per_round()} arms={','.join(cfg.runtime.arms)}")
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
