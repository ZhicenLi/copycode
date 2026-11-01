"""Run the complete leakage-controlled Diabetes regression benchmark.

Dependencies: numpy, scikit-learn, joblib, threadpoolctl.
Run: python main.py --output artifacts
The output directory must not already exist.
"""
# Import first so thread settings precede numerical-library imports.
from acceleration import execution_limits

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from experiment_config import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COMPARISONS,
    CONDITIONS,
    ESTIMATOR_SEED,
    FOREST_N_JOBS,
    SEARCH_N_JOBS,
    SEARCH_PRE_DISPATCH,
    grid_for,
    protocol_metadata,
    validate_configuration,
)
from data import create_split_schedule, inner_folds, load_dataset


def normalized_prediction(prediction, n):
    """Support both sklearn PLS prediction shapes without flattening errors."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape == (n, 1):
        prediction = prediction[:, 0]
    if prediction.shape != (n,):
        raise ValueError(
            f"Prediction shape {prediction.shape}; expected {(n,)}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    return prediction


def checked_target(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Target must be a nonempty one-dimensional array")
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite target values")
    return y


def rmse(y_true, y_pred):
    y_true = checked_target(y_true)
    y_pred = normalized_prediction(y_pred, y_true.size)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


RMSE_SCORER = make_scorer(rmse, greater_is_better=False)


def metrics(y_true, y_pred):
    y_true = checked_target(y_true)
    y_pred = normalized_prediction(y_pred, y_true.size)
    error = y_true - y_pred
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        # Undefined R2 is explicit rather than silently coerced.
        "R2": (
            float(1.0 - np.sum(error ** 2) / denominator)
            if denominator > 0 else None
        ),
    }


def build_estimator(name):
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "pls_default":
        return PLSRegression()
    if name == "svr_default":
        return SVR(kernel="rbf")
    if name == "forest_default":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=ESTIMATOR_SEED,
            n_jobs=FOREST_N_JOBS,
        )
    if name == "pls_tuned":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", PLSRegression(scale=False)),
        ])
    if name == "svr_scaled_tuned":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf")),
        ])
    if name == "svr_unscaled_tuned":
        # Genuine ablation: both fit and predict bypass StandardScaler.
        return Pipeline([
            ("scaler", "passthrough"),
            ("regressor", SVR(kernel="rbf")),
        ])
    if name == "forest_tuned":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=ESTIMATOR_SEED,
            n_jobs=FOREST_N_JOBS,
        )
    raise ValueError(f"Unknown condition: {name}")


def fit_condition(name, X, y, folds):
    X = np.asarray(X, dtype=np.float64)
    y = checked_target(y)
    if X.shape != (y.size, 10) or not np.isfinite(X).all():
        raise ValueError("Invalid estimator input array")

    estimator = build_estimator(name)
    grid = grid_for(name)
    if grid is None:
        estimator.fit(X, y)
        return estimator, {
            "selected_params": {},
            "inner_mean_negative_RMSE": None,
            "search_candidates": [],
        }

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=RMSE_SCORER,
        cv=folds,
        refit=True,
        n_jobs=SEARCH_N_JOBS,
        pre_dispatch=SEARCH_PRE_DISPATCH,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y)

    means = np.asarray(search.cv_results_["mean_test_score"])
    if not np.isfinite(means).all():
        raise RuntimeError(f"Nonfinite inner scores for {name}")

    candidate_records = []
    for index, params in enumerate(search.cv_results_["params"]):
        fold_scores = [
            float(search.cv_results_[f"split{k}_test_score"][index])
            for k in range(len(folds))
        ]
        if not np.isfinite(fold_scores).all():
            raise RuntimeError(f"Nonfinite inner fold score for {name}")
        candidate_records.append({
            "params": params,
            "mean_negative_RMSE": float(means[index]),
            "rank": int(search.cv_results_["rank_test_score"][index]),
            "fold_negative_RMSE": fold_scores,
        })

    return search.best_estimator_, {
        "selected_params": search.best_params_,
        "inner_mean_negative_RMSE": float(search.best_score_),
        "search_candidates": candidate_records,
    }


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    return value


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            json_value(payload),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def bootstrap_results(y, predictions):
    """Resample observed rows; never synthesize predictions or scores."""
    y = checked_target(y)
    predictions = {
        name: normalized_prediction(prediction, y.size)
        for name, prediction in predictions.items()
    }
    if set(predictions) != set(CONDITIONS):
        raise ValueError("Bootstrap requires every registered condition")

    # Randomness selects actual observed rows with replacement only.
    # All conditions use identical sampled indices for paired comparisons.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, y.size, size=(BOOTSTRAP_RESAMPLES, y.size)
    )
    sampled_y = y[indices]
    denominator = np.sum(
        (sampled_y - sampled_y.mean(axis=1, keepdims=True)) ** 2,
        axis=1,
    )
    distributions = {}
    intervals = {}

    def interval(values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"lower": None, "upper": None, "valid_resamples": 0}
        lower, upper = np.quantile(finite, [0.025, 0.975])
        return {
            "lower": float(lower),
            "upper": float(upper),
            "valid_resamples": int(finite.size),
        }

    for name, prediction in predictions.items():
        error = sampled_y - prediction[indices]
        squared_sum = np.sum(error ** 2, axis=1)
        ratio = np.full(BOOTSTRAP_RESAMPLES, np.nan, dtype=np.float64)
        np.divide(
            squared_sum, denominator, out=ratio, where=denominator > 0
        )
        distributions[name] = {
            "RMSE": np.sqrt(np.mean(error ** 2, axis=1)),
            "MAE": np.mean(np.abs(error), axis=1),
            "R2": 1.0 - ratio,
        }
        intervals[name] = {
            metric: interval(values)
            for metric, values in distributions[name].items()
        }

    comparisons = {}
    for first, second in COMPARISONS:
        differences = (
            distributions[first]["RMSE"] - distributions[second]["RMSE"]
        )
        comparisons[f"{first} minus {second}"] = {
            "RMSE_difference": (
                rmse(y, predictions[first]) - rmse(y, predictions[second])
            ),
            "paired_bootstrap_95_percent_interval": interval(differences),
            "interpretation": (
                "Negative RMSE difference favors the first condition."
            ),
        }
    return intervals, comparisons


def run_experiment(output_directory):
    validate_configuration()
    output_directory = Path(output_directory)
    # Refuse to mix this run with stale artifacts.
    output_directory.mkdir(parents=True, exist_ok=False)

    X, y, dataset_metadata = load_dataset()
    train_ids, test_ids, outer, final_inner, split_payload = (
        create_split_schedule(y.size)
    )
    X_train = X[train_ids]
    y_train = y[train_ids]
    if X_train.shape != (353, 10) or y_train.shape != (353,):
        raise ValueError("Invalid training array contract")

    metadata = protocol_metadata()
    metadata["dataset"] = dataset_metadata
    metadata["split_hash"] = split_payload["hash"]
    metadata["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    write_json(output_directory / "protocol.json", metadata)
    write_json(output_directory / "splits.json", split_payload)

    training_records = {}
    frozen_models = {}
    oof_predictions = {}

    with execution_limits():
        # No held-out predictions or metrics are accessed in this phase.
        for name in CONDITIONS:
            print(f"Training {name}", flush=True)
            oof = np.full(y_train.size, np.nan, dtype=np.float64)
            fold_records = []

            for fold_index, (train, validation) in enumerate(outer):
                model, selection = fit_condition(
                    name,
                    X_train[train],
                    y_train[train],
                    inner_folds(train.size),
                )
                prediction = normalized_prediction(
                    model.predict(X_train[validation]), validation.size
                )
                oof[validation] = prediction
                fold_records.append({
                    "outer_fold": fold_index,
                    "metrics": metrics(y_train[validation], prediction),
                    "selection": selection,
                })

            oof = normalized_prediction(oof, y_train.size)
            outer_rmse = np.asarray(
                [fold["metrics"]["RMSE"] for fold in fold_records],
                dtype=np.float64,
            )
            final_model, final_selection = fit_condition(
                name, X_train, y_train, final_inner
            )
            frozen_models[name] = final_model
            oof_predictions[name] = oof
            training_records[name] = {
                "outer_folds": fold_records,
                "outer_RMSE_mean": float(outer_rmse.mean()),
                "outer_RMSE_descriptive_std_ddof_1": float(
                    outer_rmse.std(ddof=1)
                ),
                "pooled_outer_OOF_metrics": metrics(y_train, oof),
                "final_training_selection": final_selection,
            }

        if tuple(frozen_models) != CONDITIONS:
            raise RuntimeError("Not every final model was frozen")

        # Persist all training-only choices before opening evaluation.
        write_json(output_directory / "training.json", training_records)
        joblib.dump(
            frozen_models,
            output_directory / "frozen_models.joblib",
            compress=3,
        )

        print(
            "All final models frozen; opening held-out evaluation.",
            flush=True,
        )
        X_test = X[test_ids]
        y_test = y[test_ids]
        if X_test.shape != (89, 10) or y_test.shape != (89,):
            raise ValueError("Invalid held-out array contract")

        predictions = {
            name: normalized_prediction(
                frozen_models[name].predict(X_test), y_test.size
            )
            for name in CONDITIONS
        }

    measured = {
        name: metrics(y_test, predictions[name]) for name in CONDITIONS
    }
    intervals, comparisons = bootstrap_results(y_test, predictions)
    results = {
        "heldout_metrics": measured,
        "fixed_prediction_bootstrap_95_percent_intervals": intervals,
        "paired_RMSE_comparisons": comparisons,
        "bootstrap_method": (
            f"Percentile bootstrap of {BOOTSTRAP_RESAMPLES} shared held-out "
            "row resamples; models and predictions remain fixed."
        ),
        "limitations": metadata["limitations"],
        "evaluation_policy": (
            "All final models and parameters were frozen before held-out "
            "evaluation; held-out results do not trigger retraining or "
            "model selection."
        ),
    }

    np.savez_compressed(
        output_directory / "predictions.npz",
        train_row_ids=train_ids,
        test_row_ids=test_ids,
        y_train=y_train,
        y_test=y_test,
        **{f"heldout__{name}": pred for name, pred in predictions.items()},
        **{f"outer_oof__{name}": pred for name, pred in oof_predictions.items()},
    )
    write_json(output_directory / "results.json", results)

    print("\nCondition                 RMSE        MAE         R2")
    for name in CONDITIONS:
        score = measured[name]
        r2_text = (
            f"{score['R2']:.4f}"
            if score["R2"] is not None else "undefined"
        )
        print(
            f"{name:25s} {score['RMSE']:10.4f} "
            f"{score['MAE']:10.4f} {r2_text:>10s}"
        )
    print(f"\nArtifacts: {output_directory.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="New artifact directory; must not already exist.",
    )
    args = parser.parse_args()
    run_experiment(args.output)


if __name__ == "__main__":
    main()