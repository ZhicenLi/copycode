"""Deterministic seed derivation.

Every stochastic decision (split, noise, init, participation, local shuffling)
gets its own seed derived from the run seed, so changing one stream never
silently perturbs another.
"""

from __future__ import annotations

import hashlib
from typing import Union

import numpy as np

Part = Union[str, int]


def derive_seed(base: int, *parts: Part) -> int:
    """Hash the base and every part, with type and length prefixes.

    Plain delimiter joining collides: `(1, "2|3")` and `(1, 2, 3)` would produce
    the same string, and so would the int 2 and the string "2".
    """
    encoded = [f"base:{base}"]
    for part in parts:
        text = str(part)
        encoded.append(f"{type(part).__name__}:{len(text)}:{text}")
    digest = hashlib.sha256("|".join(encoded).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng(base: int, *parts: Part) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base, *parts))


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible random stream.

    torchvision's crop/flip sample from Python's `random`, which is *not*
    reseeded per worker by default -- without this all workers would replay the
    same augmentation sequence.
    """
    import random

    import torch

    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
