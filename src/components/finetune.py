"""Retrain the last training run's best model on the current features, then validate it.

Picks the best feature set x model from reports/training_metrics.csv (lowest MAE, the
metric training.py's hyperparameter search optimizes) and the settings that search
chose for it from reports/best_hyperparameters.json, then retrains that one model with
those settings on the current feature file - so it picks up features added since that
run, such as price_momentum - without repeating the search. Tree models can't be
updated in place when the feature list changes, so this is a full retrain of one model
with fixed settings, not incremental fine-tuning. The settings were tuned without the
new features, so a full training run may still find better ones.

Validation, both against the last run:
- test-set metrics on the same split training.py uses
- the same houses as the last 10-house check (reports/validation_houses.csv, or 10
  random test houses if that file doesn't exist)

Artifacts go to output/models/finetuned/ and reports/finetuned_*.csv, so the last run's
models and reports are left as they were. To make predict.py use the retrained model,
move its two files from output/models/finetuned/ up into output/models/.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.components.evaluating import (
    DISPLAY_COLUMNS, add_estimates, load_splits, print_validation, select_houses,
)
from src.components.training import (
    FEATURE_SETS, MODEL_TYPES, SELECTION_METRIC, TARGET, build_preprocessor, build_search_estimators, eval_metrics,
    save_artifacts,
)
from src.utils.common import resolve_path, save_dataframe
from src.utils.load_config import load_config

METRICS = ["rmse", "mae", "r2", "mape"]
FINETUNED_DIR = "finetuned/"


def best_previous_model(reports_path):
    """(metrics row, hyperparameters) of the last run's best feature set x model, among
    the model types in use (training.MODEL_TYPES)."""
    metrics = pd.read_csv(os.path.join(reports_path, "training_metrics.csv"))
    metrics = metrics[metrics["model"].isin(MODEL_TYPES)]
    best = metrics.loc[metrics[SELECTION_METRIC].idxmin()]
    with open(os.path.join(reports_path, "best_hyperparameters.json")) as f:
        params = json.load(f)[f"{best['feature_set']}__{best['model']}"]
    return best, params


def build_model(model_name, params, training_cfg):
    """The same estimator training.py searches over, with the given settings."""
    search = build_search_estimators(training_cfg["random_state"], training_cfg)[model_name]
    model = clone(search.estimator).set_params(**params)
    # training.py runs each fit single-threaded and parallelizes the search instead; a
    # single random forest fit can use every core. xgboost stays at n_jobs=1, which it
    # needs on this setup (see training.build_search_estimators). hist_gradient_boosting
    # uses every core on its own outside the search.
    if model_name == "random_forest":
        model.set_params(n_jobs=-1)
    return model


def validation_houses(test, reports_path, name):
    """The last 10-house check's houses with its estimate for this model as last_run
    columns, or 10 random test houses if there was no last check."""
    previous_path = os.path.join(reports_path, "validation_houses.csv")
    if not os.path.exists(previous_path):
        houses = select_houses(test)
        return houses, houses[DISPLAY_COLUMNS].reset_index(drop=True)

    previous = pd.read_csv(previous_path)
    houses = select_houses(test, parcel_ids=previous["parcel_id"].tolist())
    results = houses[DISPLAY_COLUMNS].reset_index(drop=True)
    if name not in previous:  # the last check was run on a different model type
        return houses, results
    last_run = previous[["parcel_id", name, f"{name}__pct_error"]].rename(
        columns={name: "last_run", f"{name}__pct_error": "last_run__pct_error"}
    )
    return houses, results.merge(last_run, on="parcel_id", how="left")


def run():
    cfg = load_config()
    training_cfg = cfg["training"]
    reports_path = resolve_path(cfg, "reports")

    previous, params = best_previous_model(reports_path)
    feature_set, model_name = previous["feature_set"], previous["model"]
    features = FEATURE_SETS[feature_set]
    print(f"retraining {feature_set} {model_name} with {params}")

    train, test = load_splits()
    preprocessor = build_preprocessor(features)
    model = build_model(model_name, params, training_cfg)
    model.fit(preprocessor.fit_transform(train[features]), np.log1p(train[TARGET]))

    preds = np.expm1(model.predict(preprocessor.transform(test[features])))
    metrics = pd.DataFrame({
        "last_run": previous[METRICS].astype(float),
        "finetuned": pd.Series(eval_metrics(test[TARGET], preds)),
    })
    metrics["change"] = metrics["finetuned"] - metrics["last_run"]
    print(metrics.to_string(float_format=lambda x: f"{x:,.4f}"))
    print()

    houses, results = validation_houses(test, reports_path, f"{feature_set}__{model_name}")
    add_estimates(results, houses, "finetuned", preprocessor, model, features)
    print_validation(results)

    save_artifacts(resolve_path(cfg, "models") + FINETUNED_DIR, {feature_set: preprocessor}, {(feature_set, model_name): model})
    save_dataframe(metrics.rename_axis("metric").reset_index(), reports_path, "finetuned_metrics.csv")
    save_dataframe(results, reports_path, "finetuned_validation_houses.csv")
    return metrics, results


if __name__ == "__main__":
    run()
