"""Two-component 1-D Gaussian mixture, matching FedCorr's GMM usage.

FedCorr calls ``sklearn.mixture.GaussianMixture(n_components=2)`` on a 1-D
statistic (cumulative LID, or per-sample loss) and treats the lower-mean
component as "clean". scikit-learn is used when available so the numbers match
the original run exactly; otherwise a small numpy EM with the same defaults
(reg_covar=1e-6, tol=1e-3, max_iter=100) takes over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GMMSplit:
    labels: np.ndarray        # (n,) component index per sample
    means: np.ndarray         # (2,)
    variances: np.ndarray     # (2,)
    weights: np.ndarray       # (2,)
    clean_component: int      # component with the smaller mean
    backend: str              # "sklearn" | "numpy"


def split_two_components(values: np.ndarray, random_state: int = 0) -> GMMSplit:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("need at least two values to fit a two-component mixture")
    if np.allclose(values, values[0]):
        # nothing to separate: declare every client clean instead of splitting noise
        return GMMSplit(
            labels=np.zeros(values.size, dtype=np.int64),
            means=np.array([values[0], values[0]]),
            variances=np.array([0.0, 0.0]),
            weights=np.array([1.0, 0.0]),
            clean_component=0,
            backend="constant",
        )

    try:
        from sklearn.mixture import GaussianMixture  # type: ignore

        model = GaussianMixture(n_components=2, random_state=random_state).fit(
            values.reshape(-1, 1)
        )
        labels = model.predict(values.reshape(-1, 1))
        means = model.means_.ravel()
        variances = model.covariances_.ravel()
        weights = model.weights_.ravel()
        backend = "sklearn"
    except ImportError:
        # keep the pipeline runnable without scikit-learn, but say so loudly:
        # the numpy fallback is not bit-identical to sklearn's initialisation
        import warnings

        warnings.warn(
            "scikit-learn is not installed; falling back to a numpy 2-component GMM. "
            "Grouping may differ slightly from the authoritative FedCorr run.",
            RuntimeWarning,
            stacklevel=2,
        )
        labels, means, variances, weights = _numpy_em(values)
        backend = "numpy"

    clean = int(np.argmin(means))
    return GMMSplit(
        labels=labels,
        means=np.asarray(means),
        variances=np.asarray(variances),
        weights=np.asarray(weights),
        clean_component=clean,
        backend=backend,
    )


def noisy_mask(split: GMMSplit) -> np.ndarray:
    return split.labels != split.clean_component


def _numpy_em(
    values: np.ndarray, max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6
):
    lo, hi = np.quantile(values, [0.25, 0.75])
    means = np.array([lo, hi], dtype=np.float64)
    var = max(float(np.var(values)), 1e-6)
    variances = np.array([var, var], dtype=np.float64)
    weights = np.array([0.5, 0.5], dtype=np.float64)

    for _ in range(max_iter):
        resp = _responsibilities(values, means, variances, weights)
        total = resp.sum(axis=0) + 1e-12
        new_weights = total / total.sum()
        new_means = (resp * values[:, None]).sum(axis=0) / total
        new_variances = (resp * (values[:, None] - new_means) ** 2).sum(axis=0) / total + reg_covar
        shift = float(np.max(np.abs(new_means - means)))
        means, variances, weights = new_means, new_variances, new_weights
        if shift < tol:
            break
    resp = _responsibilities(values, means, variances, weights)
    labels = np.argmax(resp, axis=1)
    return labels, means, variances, weights


def _responsibilities(values, means, variances, weights):
    log_prob = np.zeros((values.size, 2))
    for k in range(2):
        log_prob[:, k] = (
            np.log(max(weights[k], 1e-12))
            - 0.5 * np.log(2 * np.pi * variances[k])
            - 0.5 * (values - means[k]) ** 2 / variances[k]
        )
    log_prob -= log_prob.max(axis=1, keepdims=True)
    prob = np.exp(log_prob)
    return prob / prob.sum(axis=1, keepdims=True)
