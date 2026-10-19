"""Local Intrinsic Dimensionality scores, ported from FedCorr's `lid_term`.

FedCorr computes LID over the model's **softmax prediction vectors** (not over
penultimate features), per client, with ``k=20`` nearest neighbours measured by
Euclidean distance. This module reproduces that exactly but drops the scipy
dependency.
"""

from __future__ import annotations

import numpy as np


def lid_scores(features: np.ndarray, k: int = 20, eps: float = 1e-6) -> np.ndarray:
    """Per-sample LID, following FedCorr's `-k / (sum(log(v / v[-1])) + eps)`."""
    x = np.asarray(features, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    k = min(k, n - 1)

    # Row blocks keep peak memory at O(block * n) instead of O(n^2). With K=10
    # CIFAR-10 a client holds 4,500 samples, and the dense matrix plus argsort
    # would otherwise need roughly half a gigabyte per client.
    scores = np.empty(n, dtype=np.float64)
    block = max(64, min(512, n))
    for start in range(0, n, block):
        stop = min(start + block, n)
        distances = _pairwise_sq(x[start:stop], x)      # (b, n)
        rows = np.arange(stop - start)
        distances[rows, np.arange(start, stop)] = np.inf  # exclude self
        idx = np.argsort(distances, axis=1)[:, :k]
        neighbour = np.maximum(np.take_along_axis(distances, idx, axis=1), eps)
        ratio = neighbour / (neighbour[:, -1:] + eps)
        scores[start:stop] = -k / (np.sum(np.log(ratio), axis=1) + eps)
    return scores


def _pairwise_sq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean distances between two blocks; FedCorr's cdist default metric."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inner = x @ y.T
    sq_x = np.einsum("ij,ij->i", x, x)
    sq_y = np.einsum("ij,ij->i", y, y)
    d2 = np.maximum(sq_x[:, None] + sq_y[None, :] - 2.0 * inner, 0.0)
    return np.sqrt(d2)


def per_client_lid(predictions: np.ndarray, k: int = 20) -> float:
    """Mean LID of one client's samples (what FedCorr accumulates)."""
    scores = lid_scores(predictions, k=k)
    return float(np.mean(scores))
