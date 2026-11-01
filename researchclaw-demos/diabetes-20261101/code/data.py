"""Authentic bundled data and deterministic, shared split schedules."""
import hashlib
import json

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split

from experiment_config import (
    INNER_FOLDS,
    INNER_SEED,
    OUTER_FOLDS,
    OUTER_SEED,
    SPLIT_SEED,
    TEST_SIZE,
)


def hash_array(array):
    array = np.asarray(array)
    canonical = np.ascontiguousarray(
        array, dtype=array.dtype.newbyteorder("<")
    )
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + canonical.tobytes()).hexdigest()


def load_dataset():
    # Use the directly imported function; no unbound sklearn namespace.
    dataset = load_diabetes(scaled=True)
    X = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.float64).reshape(-1)

    if X.shape != (442, 10) or y.shape != (442,):
        raise ValueError("Unexpected Diabetes array shapes")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite dataset values")

    return X, y, {
        "source": "sklearn.datasets.load_diabetes(scaled=True)",
        "feature_names": list(dataset.feature_names),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "X_hash": hash_array(X),
        "y_hash": hash_array(y),
        "normalization_limitation": "Dataset-level feature statistics are used.",
    }


def inner_folds(n):
    if n < INNER_FOLDS:
        raise ValueError("Not enough samples for inner cross-validation")
    return list(
        KFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=INNER_SEED,
        ).split(np.arange(n))
    )


def create_split_schedule(n):
    train_ids, test_ids = train_test_split(
        np.arange(n, dtype=np.int64),
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        shuffle=True,
    )
    if (train_ids.size, test_ids.size) != (353, 89):
        raise ValueError("Unexpected train/test sizes")
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("Train/test overlap")
    if not np.array_equal(
        np.sort(np.concatenate((train_ids, test_ids))), np.arange(n)
    ):
        raise ValueError("Invalid train/test coverage")

    outer = list(
        KFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=OUTER_SEED,
        ).split(train_ids)
    )
    covered = np.concatenate([validation for _, validation in outer])
    if not np.array_equal(np.sort(covered), np.arange(train_ids.size)):
        raise ValueError("Invalid outer-fold coverage")

    final_inner = inner_folds(train_ids.size)

    def global_folds(ids, folds):
        return [
            {
                "train_row_ids": ids[train].tolist(),
                "validation_row_ids": ids[validation].tolist(),
            }
            for train, validation in folds
        ]

    payload = {
        "train_ids": train_ids.tolist(),
        "test_ids": test_ids.tolist(),
        "outer": [
            {
                "train_row_ids": train_ids[train].tolist(),
                "validation_row_ids": train_ids[validation].tolist(),
                "inner": global_folds(
                    train_ids[train], inner_folds(train.size)
                ),
            }
            for train, validation in outer
        ],
        "final_inner": global_folds(train_ids, final_inner),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["hash"] = hashlib.sha256(encoded).hexdigest()
    return train_ids, test_ids, outer, final_inner, payload