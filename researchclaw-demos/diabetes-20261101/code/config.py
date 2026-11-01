"""Fixed protocol, immutable condition registry, and validation."""
from types import MappingProxyType

PROTOCOL_VERSION = "diabetes-regression-v1"
PRIMARY_METRIC = "RMSE"

SPLIT_SEED = 42
OUTER_SEED = 42
INNER_SEED = 43
ESTIMATOR_SEED = 42
BOOTSTRAP_SEED = 44

TEST_SIZE = 0.2
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP_RESAMPLES = 2000

SEARCH_N_JOBS = 4
SEARCH_PRE_DISPATCH = 4
FOREST_N_JOBS = 1
NATIVE_THREADS = 1

CONDITIONS = (
    "dummy_mean",
    "pls_default",
    "svr_default",
    "forest_default",
    "pls_tuned",
    "svr_scaled_tuned",
    "forest_tuned",
    "svr_unscaled_tuned",
)

PLS_GRID = MappingProxyType({
    "regressor__n_components": tuple(range(1, 11)),
})
SVR_GRID = MappingProxyType({
    "regressor__C": (0.1, 1, 10, 100),
    "regressor__epsilon": (0.01, 0.1),
    "regressor__gamma": ("scale", 0.01, 0.1),
})

# Explicit completion of the truncated blueprint: 3 * 4 * 2 = 24.
FOREST_GRID = MappingProxyType({
    "max_depth": (None, 5, 10),
    "min_samples_leaf": (1, 2, 4, 8),
    "max_features": (1.0, "sqrt"),
})

TUNING_GRIDS = MappingProxyType({
    "pls_tuned": PLS_GRID,
    "svr_scaled_tuned": SVR_GRID,
    "svr_unscaled_tuned": SVR_GRID,
    "forest_tuned": FOREST_GRID,
})

COMPARISONS = (
    *((name, "dummy_mean") for name in CONDITIONS if name != "dummy_mean"),
    ("pls_tuned", "pls_default"),
    ("svr_scaled_tuned", "svr_default"),
    ("forest_tuned", "forest_default"),
    ("svr_scaled_tuned", "svr_unscaled_tuned"),
)

LIMITATIONS = (
    "load_diabetes(scaled=True) uses dataset-level feature statistics.",
    "Additional StandardScaler instances are fitted only inside training folds.",
    "Five outer-fold scores are dependent; their standard deviation is descriptive.",
    "Fixed-prediction bootstrap intervals omit training-set variability.",
    "One small dataset does not establish general model superiority or causality.",
    "Default-versus-tuned comparisons are configuration comparisons, not pure "
    "tuning-only causal effects: default PLS scales X and y internally, whereas "
    "the tuned PLS pipeline scales X only; default forests have 100 trees, "
    "whereas tuned forests have 300.",
    "The supplied forest grid was truncated; the implemented grid explicitly "
    "uses depths [None, 5, 10], leaf sizes [1, 2, 4, 8], and "
    "max_features [1.0, 'sqrt'].",
)


def grid_for(name):
    """Return a fresh sklearn-compatible grid, or None if untuned."""
    if name not in CONDITIONS:
        raise ValueError(f"Unknown condition: {name}")
    grid = TUNING_GRIDS.get(name)
    if grid is None:
        return None
    return {key: list(values) for key, values in grid.items()}


def validate_configuration():
    from sklearn.model_selection import ParameterGrid

    if len(CONDITIONS) != 8 or len(set(CONDITIONS)) != 8:
        raise ValueError("Expected eight unique conditions")
    if not set(TUNING_GRIDS).issubset(CONDITIONS):
        raise ValueError("Unknown condition in tuning registry")

    for name, expected in (
        ("pls_tuned", 10),
        ("svr_scaled_tuned", 24),
        ("svr_unscaled_tuned", 24),
        ("forest_tuned", 24),
    ):
        grid = grid_for(name)
        if grid is None or len(ParameterGrid(grid)) != expected:
            raise ValueError(f"Invalid grid size: {name}")

    for name in CONDITIONS:
        if name not in TUNING_GRIDS and grid_for(name) is not None:
            raise ValueError(f"Unexpected grid for untuned condition: {name}")

    if grid_for("svr_scaled_tuned") != grid_for("svr_unscaled_tuned"):
        raise ValueError("SVR ablation grids must match")

    for first, second in COMPARISONS:
        if first not in CONDITIONS or second not in CONDITIONS:
            raise ValueError("Unknown comparison condition")

    if PRIMARY_METRIC != "RMSE":
        raise ValueError("The protocol requires RMSE as the primary metric")
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between zero and one")
    if min(OUTER_FOLDS, INNER_FOLDS) < 2:
        raise ValueError("Cross-validation requires at least two folds")
    if min(
        BOOTSTRAP_RESAMPLES,
        SEARCH_N_JOBS,
        SEARCH_PRE_DISPATCH,
        FOREST_N_JOBS,
        NATIVE_THREADS,
    ) < 1:
        raise ValueError("Execution counts must be positive")


def protocol_metadata():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_resolution": (
            "The supplied primary_metric placeholder is resolved to RMSE."
        ),
        "selection_rule": "Maximize mean training-only inner-fold negative RMSE.",
        "seeds": {
            "split": SPLIT_SEED,
            "outer": OUTER_SEED,
            "inner": INNER_SEED,
            "estimator": ESTIMATOR_SEED,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "test_size": TEST_SIZE,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "execution": {
            "device": "CPU",
            "conditions": "sequential",
            "search_n_jobs": SEARCH_N_JOBS,
            "search_pre_dispatch": SEARCH_PRE_DISPATCH,
            "forest_n_jobs": FOREST_N_JOBS,
            "native_threads": NATIVE_THREADS,
        },
        "conditions": list(CONDITIONS),
        "grids": {name: grid_for(name) for name in TUNING_GRIDS},
        "limitations": list(LIMITATIONS),
        "artifact_schema": {
            "protocol.json": "Protocol, dataset hashes, versions, and split hash",
            "splits.json": "Global row IDs for all outer and inner folds",
            "training.json": "Outer prediction metrics and training-only selections",
            "frozen_models.joblib": "All final training-only fitted estimators",
            "predictions.npz": "Actual held-out and outer OOF predictions",
            "results.json": "Held-out metrics and fixed-prediction bootstrap intervals",
        },
    }