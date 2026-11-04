"""Train gradient boosting regressors for `total_value` (Mecklenburg).

Reads data/features/properties_features.csv and trains each model type in
MODEL_TYPES on two feature sets. Only histogram gradient boosting is enabled for now -
on a validation split carved from the training rows it beat the tuned random forest
on every metric (MAE $29,407 vs $33,066, RMSE $78,464 vs $88,876), and its saved
models are a few MB instead of over 1GB - so 2 models are trained. The random forest
and xgboost setups are kept in build_search_estimators and can be switched back on by
adding them to MODEL_TYPES (xgboost after `uv sync --group xgboost`).

- with_neighborhood_price: every feature, including neighborhood_median_price_per_sqft
- without_neighborhood_price: the same minus that column. The neighborhood price
  already contains most of the effect of location, amenities and area, so without it
  the `area` and accessibility features have to carry that signal themselves - this is
  the model to read when asking what those features are worth.

The split is random (80/20), not grouped by neighborhood: the use case is valuing a
house in a neighborhood that already has sales, and with a neighborhood-grouped split
every test house would be missing its neighborhood aggregates. Those aggregates and
price_momentum are recomputed here from training rows only, so a test house's own
sale never feeds its features.

`land_value`/`building_value` and sale-derived columns are dropped as target leakage
(total_value is exactly land_value + building_value) or near-entirely missing.
`neighborhood_code` is dropped in favor of its aggregates and `area`. See
config/config.yaml `training:` section for the split/hyperparameters used below.

Hyperparameters are tuned per feature set per model type via RandomizedSearchCV
(config/config.yaml `training.search_iter`/`cv_folds`), scored by MAE in dollars
(dollar_mae): the goal is a smaller typical miss, even at some cost in RMSE or R^2. Each fitted search's
`best_estimator_` (not the search wrapper itself) is what gets saved.

Each run is saved to its own folder, output/models/runs/{run_id}/ (see model_registry.py),
and becomes the production model predict.py loads only if check_and_promote finds it no
worse than the one it would replace.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components import model_registry
from src.components.feature_engineering import add_sale_features
from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

TARGET = "total_value"
NEIGHBORHOOD_PRICE = "neighborhood_median_price_per_sqft"
# Features built from sale prices - dropped from the feature file and recomputed from
# training rows only, so a test house's own sale never feeds its features.
SALE_DERIVED_FEATURES = [
    NEIGHBORHOOD_PRICE,
    "neighborhood_sale_count",
    "price_momentum",
    "neighborhood_adjusted_price_per_sqft",
    "neighborhood_sale_years",
    "neighborhood_value_estimate",
    "relative_heated_area",
    "relative_year_built",
]

CATEGORICAL_FEATURES = ["land_use_class", "area"]
NUMERIC_FEATURES = [
    "acreage",
    "year_built",
    "heated_area",
    "bedrooms",
    "bathrooms",
    "latitude",
    "longitude",
    "property_age",
    "neighborhood_median_price_per_sqft",
    "neighborhood_sale_count",
    "price_momentum",
    "education_nearest_miles",
    "places_count",
    "health_care_share",
    "activity_share",
    "emergency_fire_miles",
    "emergency_police_miles",
    "emergency_medical_miles",
    "elementary_school_score",
    "middle_school_score",
    "high_school_score",
    "violent_crime_rate",
    "property_crime_rate",
    "in_floodplain",
    "light_rail_miles",
    "neighborhood_adjusted_price_per_sqft",
    "neighborhood_value_estimate",
    "relative_heated_area",
    "lot_sqft_per_heated_sqft",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FEATURE_SETS = {
    "with_neighborhood_price": FEATURE_COLUMNS,
    "without_neighborhood_price": [c for c in FEATURE_COLUMNS if c != NEIGHBORHOOD_PRICE],
}
# Model types trained, validated and served. "random_forest" and "xgboost" are also
# available (see build_search_estimators) but left out for now; xgboost also needs
# `uv sync --group xgboost`.
MODEL_TYPES = ["hist_gradient_boosting"]
# The metric model selection optimizes, here and in finetune.py.
SELECTION_METRIC = "mae"


def load_training_data(features_path):
    """Rows with a positive target. The feature file's sale-derived features are
    dropped - they were computed on every row, so split_with_aggregates recomputes them.
    A file built before a feature was added doesn't have it, which is fine: it is computed
    here too."""
    df = read_dataframe(features_path, "properties_features.csv")
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    return df.drop(columns=SALE_DERIVED_FEATURES, errors="ignore")


def test_address_parcels():
    """Parcel ids of config `test_addresses`, the houses every example and check uses."""
    from src.components.search import lookup_property  # search imports this module

    return [row["parcel_id"] for address in load_config().get("test_addresses", []) for row in lookup_property(address)]


def split_rows(df, training_cfg):
    train, test = train_test_split(df, test_size=training_cfg["test_size"], random_state=training_cfg["random_state"])
    # the test addresses always go to the test set, so their estimates are real tests
    moved = train["parcel_id"].isin(test_address_parcels())
    return train[~moved], pd.concat([test, train[moved]])


def split_with_aggregates(df, training_cfg):
    train, test = split_rows(df, training_cfg)
    return add_sale_features(train, source=train), add_sale_features(test, source=train)


def build_preprocessor(features):
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, [c for c in features if c in NUMERIC_FEATURES]),
            ("cat", categorical_pipeline, [c for c in features if c in CATEGORICAL_FEATURES]),
        ]
    )


def dollar_mae(y_log_true, y_log_pred):
    """MAE in dollars for models trained on log1p(total_value)."""
    return mean_absolute_error(np.expm1(y_log_true), np.expm1(y_log_pred))


def eval_metrics(y_true, y_pred):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


# max_depth capped (never None/unbounded): unbounded depth on the earlier ~900k-row
# multi-county data once produced a 13.6GB random_forest.joblib (grossly overfit and
# impractical to load later). Depth 20 on Mecklenburg alone still gives ~1.1GB files.
RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}
# max_iter is an upper bound: early stopping on an internal 10% validation split picks
# the number of trees per fit, which also keeps the search affordable.
HIST_GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_leaf_nodes": [63, 127, 255, 511],
    "min_samples_leaf": [5, 10, 20, 50],
    "l2_regularization": [0.0, 0.1, 1.0],
    "max_features": [0.5, 0.8, 1.0],
}
XGBOOST_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def build_search_estimators(random_state, training_cfg, model_types=MODEL_TYPES):
    """A hyperparameter search for each of `model_types`. xgboost is imported only when it
    is asked for: it is an optional dependency (`uv sync --group xgboost`), left out of the
    default install and the Docker image while it isn't in MODEL_TYPES."""
    # n_jobs=1 on each base estimator, n_jobs=-1 on the search itself: parallelizes across
    # search candidates/CV folds (separate worker processes) instead of nesting parallelism
    # inside each fit. Also required for xgboost specifically - on this xgboost 3.4.1 /
    # Python 3.14 / macOS arm64 combo, n_jobs=-1 *inside* a single XGBRegressor fit segfaults
    # deterministically regardless of tree_method, and the default tree_method="hist"
    # segfaults even single-threaded (verified via bisection); tree_method="approx" + n_jobs=1
    # is the combination that survives.
    search_kwargs = dict(
        n_iter=training_cfg["search_iter"],
        cv=training_cfg["cv_folds"],
        scoring=make_scorer(dollar_mae, greater_is_better=False),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    def xgboost():
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=random_state, n_jobs=1, tree_method="approx"), XGBOOST_PARAM_DISTRIBUTIONS

    estimators = {
        # no n_jobs: it uses OpenMP threads, which joblib limits per search worker
        "hist_gradient_boosting": lambda: (
            HistGradientBoostingRegressor(
                random_state=random_state,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
            ),
            HIST_GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS,
        ),
        "random_forest": lambda: (
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        ),
        "xgboost": xgboost,
    }
    return {name: RandomizedSearchCV(*estimators[name](), **search_kwargs) for name in model_types}


def train_models(train, test, training_cfg):
    results = {}
    models = {}
    preprocessors = {}
    best_params = {}
    y_train_log = np.log1p(train[TARGET])

    for set_name, features in FEATURE_SETS.items():
        preprocessor = build_preprocessor(features)
        X_train_p = preprocessor.fit_transform(train[features])
        X_test_p = preprocessor.transform(test[features])

        searches = build_search_estimators(training_cfg["random_state"], training_cfg)
        for model_name in MODEL_TYPES:
            search = searches[model_name]
            search.fit(X_train_p, y_train_log)
            best_model = search.best_estimator_
            preds = np.expm1(best_model.predict(X_test_p))
            results[(set_name, model_name)] = eval_metrics(test[TARGET], preds)
            models[(set_name, model_name)] = best_model
            best_params[f"{set_name}__{model_name}"] = search.best_params_

        preprocessors[set_name] = preprocessor

    return results, models, preprocessors, best_params


def save_artifacts(models_path, preprocessors, models):
    os.makedirs(models_path, exist_ok=True)
    for set_name, preprocessor in preprocessors.items():
        joblib.dump(preprocessor, os.path.join(models_path, f"{set_name}_preprocessor.joblib"))
    for (set_name, model_name), model in models.items():
        joblib.dump(model, os.path.join(models_path, f"{set_name}_{model_name}.joblib"))


# A new run replaces the production model only if none of its models' MAE is more than
# this much above the production model's, on the houses neither run trained on (a retrain
# on refreshed data moves it a little either way).
MAX_MAE_INCREASE = 0.01
# With fewer houses held out by both runs than this, the comparison is left to a person.
MIN_SHARED_HOUSES = 1_000


def metrics_table(results):
    metrics_df = pd.DataFrame(results).T
    metrics_df.index = metrics_df.index.set_names(["feature_set", "model"])
    return metrics_df.sort_values(["feature_set", SELECTION_METRIC])


def save_run(models_path, train, test, preprocessors, models, metrics_df, best_params, training_cfg, **info):
    """Save a run to output/models/runs/{run_id}/ (model_registry.py) and return its id.
    `info` goes to model_registry.run_info: `features_file`, and `source` and `base_run`
    for a run that starts from a copy of another (finetune.py)."""
    run_id, path = model_registry.new_run(models_path, info.get("base_run"))
    save_artifacts(path, preprocessors, models)
    save_dataframe(metrics_df.reset_index(), path, "training_metrics.csv")
    with open(path + "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)
    parcels = model_registry.held_out(train, test)
    if info.get("base_run") is not None:  # the copied models were trained on the base run's split
        base = model_registry.held_out_parcels(models_path, info["base_run"])
        parcels = parcels & base if base is not None else None
    record = model_registry.run_info(run_id, **info)
    if record["data"] is not None:
        record["data"]["rows"] = len(train) + len(test)
    record.update(
        train_rows=len(train),
        test_rows=len(test),
        held_out_parcels=model_registry.save_held_out(path, parcels),
        training=training_cfg,
        feature_sets={
            fs: list(joblib.load(path + f"{fs}_preprocessor.joblib").feature_names_in_) for fs in FEATURE_SETS
        },
        model_types=MODEL_TYPES,
        metrics={f"{fs}/{m}": row.to_dict() for (fs, m), row in metrics_df.iterrows()},
    )
    model_registry.write_info(models_path, run_id, record)
    return run_id


def _predict_with(models_path, run_id, feature_set, model_name, rows):
    """Predictions in dollars from one run's model, or None when the run doesn't have it.
    The columns come from the run's own preprocessor, so a run made before a feature was
    added or removed is fed what it was trained on."""
    path = model_registry.run_path(models_path, run_id)
    if not os.path.exists(path + f"{feature_set}_{model_name}.joblib"):
        return None
    preprocessor = joblib.load(path + f"{feature_set}_preprocessor.joblib")
    model = joblib.load(path + f"{feature_set}_{model_name}.joblib")
    return np.expm1(model.predict(preprocessor.transform(rows[list(preprocessor.feature_names_in_)])))


def compare_with_production(models_path, run_id, test):
    """(promote, reason) for run `run_id`: each of its models against the production
    run's, both scored on the test houses that both runs held out. Neither model trained
    on those, so a different split (as after a data refresh) can't favor either one."""
    production = model_registry.production_run(models_path)
    if production is None:
        return True, "the first run: no production model to compare with"
    held_out = model_registry.held_out_parcels(models_path, production)
    if held_out is None:
        return False, f"production ({production}) has no record of its held-out houses: compare by hand"
    shared = test[test["parcel_id"].isin(held_out & model_registry.held_out_parcels(models_path, run_id))]
    if len(shared) < MIN_SHARED_HOUSES:
        return False, (
            f"only {len(shared):,} test houses are held out by both this run and production ({production}), "
            f"fewer than {MIN_SHARED_HOUSES:,}: compare by hand"
        )
    changes, better = [], True
    for feature_set in FEATURE_SETS:
        for model_name in MODEL_TYPES:
            new = mean_absolute_error(
                shared[TARGET], _predict_with(models_path, run_id, feature_set, model_name, shared)
            )
            try:
                old = _predict_with(models_path, production, feature_set, model_name, shared)
            except (KeyError, ValueError) as e:
                return (
                    False,
                    f"production's {feature_set} {model_name} can't score this run's features ({e}): compare by hand",
                )
            if old is None:
                changes.append(f"{feature_set} {model_name}: new, nothing to compare with")
                continue
            old = mean_absolute_error(shared[TARGET], old)
            better &= new <= old * (1 + MAX_MAE_INCREASE)
            changes.append(f"{feature_set} {model_name} MAE ${new:,.0f} vs ${old:,.0f} ({new / old - 1:+.2%})")
    verdict = "no" if better else f"an MAE more than {MAX_MAE_INCREASE:.0%}"
    return better, (
        f"{verdict} worse than production ({production}) on the {len(shared):,} test houses both held out: "
        + "; ".join(changes)
    )


def check_and_promote(models_path, reports_path, run_id, test):
    """Promote run `run_id` if compare_with_production allows it, and record the decision."""
    promoted, reason = compare_with_production(models_path, run_id, test)
    if promoted:
        model_registry.promote(models_path, reports_path, run_id, reason)
    model_registry.record_check(models_path, run_id, promoted, reason)
    if promoted:
        print(f"run {run_id} is now production: {reason}")
    else:
        print(f"run {run_id} NOT promoted, production is still {model_registry.production_run(models_path)}: {reason}")
        print(f"to promote it anyway: uv run python -m src.components.model_registry promote {run_id}")
    return promoted, reason


def run():
    cfg = load_config()
    training_cfg = cfg["training"]
    features_path = resolve_path(cfg, "features_data")
    models_path = resolve_path(cfg, "models")
    reports_path = resolve_path(cfg, "reports")

    train, test = split_with_aggregates(load_training_data(features_path), training_cfg)
    results, models, preprocessors, best_params = train_models(train, test, training_cfg)

    metrics_df = metrics_table(results)
    print(metrics_df)
    print()
    print(json.dumps(best_params, indent=2))

    run_id = save_run(
        models_path,
        train,
        test,
        preprocessors,
        models,
        metrics_df,
        best_params,
        training_cfg,
        features_file=features_path + "properties_features.csv",
    )
    check_and_promote(models_path, reports_path, run_id, test)
    return metrics_df


if __name__ == "__main__":
    run()
