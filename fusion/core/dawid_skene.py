"""Dawid-Skene label aggregation.

Faithful to the authoritative implementation shipped with FedDS (hard-argmax
predictions, confusion matrices estimated by EM, reliability = mean diagonal),
with three deliberate differences that do not change the model:

* the E-step works in log space, so the product over many annotators cannot
  underflow to zero (matters once the annotator pool grows past ~20);
* the class prior is optional and off by default, because the released code
  never updates it even though the paper's Eq. (9) does;
* the number of EM iterations and the convergence flag are returned, because
  the published code hides them and they are the first thing to check when the
  reliability ranking looks wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DSResult:
    confusion: np.ndarray        # (num_annotators, C, C), rows sum to 1
    reliability: np.ndarray      # (num_annotators,), mean diagonal
    estimated: np.ndarray        # (num_items,) argmax of the posterior
    posterior: np.ndarray        # (num_items, C)
    iterations: int
    converged: bool
    zero_support_rows: int


def dawid_skene(
    predictions: np.ndarray,
    num_classes: int,
    max_iter: int = 500,
    tol: float = 1e-5,
    use_class_prior: bool = False,
) -> DSResult:
    """`predictions[i, a]` is annotator `a`'s hard label for item `i`."""
    pred = np.asarray(predictions, dtype=np.int64)
    if pred.ndim != 2:
        raise ValueError("predictions must be (num_items, num_annotators)")
    num_items, num_annotators = pred.shape
    if num_items == 0 or num_annotators == 0:
        raise ValueError("predictions must contain at least one item and one annotator")
    if pred.min() < 0 or pred.max() >= num_classes:
        raise ValueError(
            f"predictions must lie in [0, {num_classes}); got "
            f"[{pred.min()}, {pred.max()}]"
        )

    posterior = _majority_vote(pred, num_classes)
    prior = np.full(num_classes, 1.0 / num_classes)

    confusion = _m_step(posterior, pred, num_classes)
    converged = False
    iterations = 0

    for step in range(1, max_iter + 1):
        iterations = step
        previous = posterior
        log_joint = np.zeros((num_items, num_classes), dtype=np.float64)
        for annotator in range(num_annotators):
            # cm[c, pred_i] for every item i  ->  shape (num_items, C)
            log_joint += np.log(
                np.maximum(confusion[annotator][:, pred[:, annotator]].T, 1e-300)
            )
        if use_class_prior:
            log_joint += np.log(np.maximum(prior, 1e-300))
        posterior = _softmax(log_joint)

        confusion = _m_step(posterior, pred, num_classes)
        if use_class_prior:
            prior = posterior.mean(axis=0)

        if np.max(np.abs(posterior - previous)) < tol:
            converged = True
            break

    rows_empty = int(np.sum(confusion.sum(axis=2) == 0))
    reliability = np.array([float(np.mean(np.diag(matrix))) for matrix in confusion])
    return DSResult(
        confusion=confusion,
        reliability=reliability,
        estimated=np.argmax(posterior, axis=1),
        posterior=posterior,
        iterations=iterations,
        converged=converged,
        zero_support_rows=rows_empty,
    )


def _majority_vote(pred: np.ndarray, num_classes: int) -> np.ndarray:
    num_items = pred.shape[0]
    counts = np.zeros((num_items, num_classes), dtype=np.float64)
    rows = np.arange(num_items)
    for annotator in range(pred.shape[1]):
        counts[rows, pred[:, annotator]] += 1.0
    counts /= pred.shape[1]
    counts = np.maximum(counts, 1e-10)
    counts /= counts.sum(axis=1, keepdims=True)
    return counts


def _m_step(posterior: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    num_annotators = pred.shape[1]
    confusion = np.zeros((num_annotators, num_classes, num_classes), dtype=np.float64)
    eye = np.eye(num_classes, dtype=np.float64)
    for annotator in range(num_annotators):
        one_hot = eye[pred[:, annotator]]          # (num_items, C)
        matrix = posterior.T @ one_hot             # (C, C): [c, l] mass of c voting l
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        confusion[annotator] = matrix / row_sums
    return confusion


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    total = exp.sum(axis=1, keepdims=True)
    uniform = np.full_like(exp, 1.0 / exp.shape[1])
    return np.where(total > 0, exp / np.where(total > 0, total, 1.0), uniform)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, used to score a reliability ranking."""
    from scipy.stats import spearmanr  # optional

    return float(spearmanr(a, b).statistic)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation without scipy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    ra = _rank(a)
    rb = _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata for tied values.

    Plain argsort order would give tied entries arbitrary consecutive ranks and
    distort the Spearman correlation -- which matters here because clean clients
    all share the exact estimate 0.0.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    size = len(values)
    start = 0
    while start < size:
        stop = start
        while stop + 1 < size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks
