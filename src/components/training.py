"""Train gradient boosting regressors for `total_value` (Mecklenburg).

Reads data/features/properties_features.csv and trains each model type in
MODEL_TYPES on two feature sets. Only histogram gradient boosting is enabled for now -
on a validation split carved from the training rows it beat the tuned random forest
on every metric (MAE $29,407 vs $33,066, RMSE $78,464 vs $88,876), and its saved
models are a few MB instead of over 1GB - so 2 models are trained. The random forest
and xgboost setups are kept in build_search_estimators and can be switched back on by
adding them to MODEL_TYPES.

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
from xgboost import XGBRegressor

from src.components.feature_engineering import add_neighborhood_aggregates, add_price_momentum
from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

TARGET = "total_value"
NEIGHBORHOOD_PRICE = "neighborhood_median_price_per_sqft"
# Features built from sale prices - dropped from the feature file and recomputed from
# training rows only, so a test house's own sale never feeds its features.
SALE_DERIVED_FEATURES = [NEIGHBORHOOD_PRICE, "neighborhood_sale_count", "price_momentum"]

CATEGORICAL_FEATURES = ["land_use_class", "area"]
NUMERIC_FEATURES = [
    "acreage", "year_built", "heated_area", "bedrooms", "bathrooms",
    "latitude", "longitude", "property_age",
    "neighborhood_median_price_per_sqft", "neighborhood_sale_count", "price_momentum",
    "education_nearest_miles", "places_count", "health_care_share", "activity_share",
    "emergency_fire_miles", "emergency_police_miles", "emergency_medical_miles",
    "elementary_school_score", "middle_school_score", "high_school_score",
    "violent_crime_rate", "property_crime_rate", "in_floodplain", "light_rail_miles",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FEATURE_SETS = {
    "with_neighborhood_price": FEATURE_COLUMNS,
    "without_neighborhood_price": [c for c in FEATURE_COLUMNS if c != NEIGHBORHOOD_PRICE],
}
# Model types trained, validated and served. "random_forest" and "xgboost" are also
# available (see build_search_estimators) but left out for now.
MODEL_TYPES = ["hist_gradient_boosting"]
# The metric model selection optimizes, here and in finetune.py.
SELECTION_METRIC = "mae"


def load_training_data(features_path):
    """Rows with a positive target. The feature file's sale-derived features are
    dropped - they were computed on every row, so split_with_aggregates recomputes them."""
    df = read_dataframe(features_path, "properties_features.csv")
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    return df.drop(columns=SALE_DERIVED_FEATURES)


def test_address_parcels():
    """Parcel ids of config `test_addresses`, the houses every example and check uses."""
    from src.components.search import lookup_property  # search imports this module
    return [row["parcel_id"] for address in load_config().get("test_addresses", [])
            for row in lookup_property(address)]


def split_with_aggregates(df, training_cfg):
    train, test = train_test_split(
        df, test_size=training_cfg["test_size"], random_state=training_cfg["random_state"]
    )
    # the test addresses always go to the test set, so their estimates are real tests
    moved = train["parcel_id"].isin(test_address_parcels())
    train, test = train[~moved], pd.concat([test, train[moved]])

    def add(rows):
        return add_price_momentum(add_neighborhood_aggregates(rows, source=train), source=train)

    return add(train), add(test)


def build_preprocessor(features):
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, [c for c in features if c in NUMERIC_FEATURES]),
        ("cat", categorical_pipeline, [c for c in features if c in CATEGORICAL_FEATURES]),
    ])


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


def build_search_estimators(random_state, training_cfg):
    # n_jobs=1 on each base estimator, n_jobs=-1 on the search itself: parallelizes across
    # search candidates/CV folds (separate worker processes) instead of nesting parallelism
    # inside each fit. Also required for xgboost specifically - on this xgboost 3.4.1 /
    # Python 3.14 / macOS arm64 combo, n_jobs=-1 *inside* a single XGBRegressor fit segfaults
    # deterministically regardless of tree_method, and the default tree_method="hist"
    # segfaults even single-threaded (verified via bisection); tree_method="approx" + n_jobs=1
    # is the combination that survives.
    search_kwargs = dict(
        n_iter=training_cfg["search_iter"], cv=training_cfg["cv_folds"],
        scoring=make_scorer(dollar_mae, greater_is_better=False), random_state=random_state, n_jobs=-1, refit=True,
    )
    return {
        # no n_jobs: it uses OpenMP threads, which joblib limits per search worker
        "hist_gradient_boosting": RandomizedSearchCV(
            HistGradientBoostingRegressor(
                random_state=random_state, max_iter=3000,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
            ),
            HIST_GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS, **search_kwargs,
        ),
        "random_forest": RandomizedSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            RANDOM_FOREST_PARAM_DISTRIBUTIONS, **search_kwargs,
        ),
        "xgboost": RandomizedSearchCV(
            XGBRegressor(random_state=random_state, n_jobs=1, tree_method="approx"),
            XGBOOST_PARAM_DISTRIBUTIONS, **search_kwargs,
        ),
    }


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


def run():
    cfg = load_config()
    training_cfg = cfg["training"]
    features_path = resolve_path(cfg, "features_data")
    models_path = resolve_path(cfg, "models")
    reports_path = resolve_path(cfg, "reports")

    train, test = split_with_aggregates(load_training_data(features_path), training_cfg)
    results, models, preprocessors, best_params = train_models(train, test, training_cfg)

    metrics_df = pd.DataFrame(results).T
    metrics_df.index = metrics_df.index.set_names(["feature_set", "model"])
    metrics_df = metrics_df.sort_values(["feature_set", SELECTION_METRIC])
    print(metrics_df)
    print()
    print(json.dumps(best_params, indent=2))

    save_artifacts(models_path, preprocessors, models)
    save_dataframe(metrics_df.reset_index(), reports_path, "training_metrics.csv")
    os.makedirs(reports_path, exist_ok=True)
    with open(os.path.join(reports_path, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    return metrics_df


if __name__ == "__main__":
    run()
