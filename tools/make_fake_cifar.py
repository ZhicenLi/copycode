"""Create a tiny CIFAR-10-format dataset.

This exercises the *real* torchvision loading path (directory layout, pickles,
`.data`/`.targets`, transforms) without a 170 MB download. Images are random,
so accuracies are meaningless -- the point is to prove the data plumbing on a
path that differs from the synthetic in-memory branch.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TARGET = os.path.join(ROOT, "cifar-10-batches-py")

TRAIN_PER_BATCH = 120
TEST_SIZE = 100
NUM_CLASSES = 10


def _batch(images: np.ndarray, labels: np.ndarray, name: str) -> dict:
    # torchvision loads with encoding="latin1" and then reads string keys
    return {
        "batch_label": name,
        "labels": labels.tolist(),
        "data": images.reshape(len(images), -1),
        "filenames": [f"{name}_{i}.png" for i in range(len(images))],
    }


def main() -> int:
    rng = np.random.default_rng(0)
    os.makedirs(TARGET, exist_ok=True)

    for index in range(1, 6):
        images = rng.integers(0, 256, (TRAIN_PER_BATCH, 32, 32, 3), dtype=np.uint8)
        labels = rng.integers(0, NUM_CLASSES, TRAIN_PER_BATCH)
        with open(os.path.join(TARGET, f"data_batch_{index}"), "wb") as handle:
            pickle.dump(_batch(images, labels, f"training batch {index}"), handle)

    images = rng.integers(0, 256, (TEST_SIZE, 32, 32, 3), dtype=np.uint8)
    labels = rng.integers(0, NUM_CLASSES, TEST_SIZE)
    with open(os.path.join(TARGET, "test_batch"), "wb") as handle:
        pickle.dump(_batch(images, labels, "testing batch 1"), handle)

    meta = {"label_names": [f"class_{i}" for i in range(NUM_CLASSES)]}
    with open(os.path.join(TARGET, "batches.meta"), "wb") as handle:
        pickle.dump(meta, handle)

    total = TRAIN_PER_BATCH * 5
    print(f"wrote {total} train / {TEST_SIZE} test images to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
