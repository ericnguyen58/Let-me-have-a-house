"""Model evaluation beyond training.py's test-set metrics.

validate_houses is a spot check: every trained model's estimate next to the actual
total_value for a handful of individual houses, so errors can be read house by house
instead of only as averages. Houses come from training.py's test split (rebuilt here
with the same config and random_state), so no model saw them during training, and
their sale-derived features are computed from training rows only, as in training.
The split only matches training's if properties_features.csv hasn't changed since
the models were trained.
"""

import numpy as np

from src.components import predict
from src.components.training import FEATURE_SETS, MODEL_TYPES, TARGET, load_training_data, split_with_aggregates
from src.utils.common import resolve_path, save_dataframe
from src.utils.load_config import load_config

VALIDATION_SIZE = 10
DISPLAY_COLUMNS = [
    "parcel_id", "area", "land_use_class", "heated_area", "bedrooms", "bathrooms",
    "year_built", "sale_price", "sale_date", TARGET,
]


def load_splits():
    """(train, test) exactly as training.py splits them."""
    cfg = load_config()
    return split_with_aggregates(load_training_data(resolve_path(cfg, "features_data")), cfg["training"])


def select_houses(test, n=VALIDATION_SIZE, parcel_ids=None, random_state=0):
    """n random test-set houses, or the given parcel_ids, which must all be in the test
    split - a house the models were trained on would give a misleadingly good result."""
    if parcel_ids is None:
        return test.sample(n, random_state=random_state)
    houses = test[test["parcel_id"].isin(parcel_ids)]
    missing = set(parcel_ids) - set(houses["parcel_id"])
    if missing:
        raise ValueError(f"not in the test split (trained on, or not in the data): {sorted(map(str, missing))}")
    return houses


def add_estimates(results, houses, name, preprocessor, model, features):
    """Add column `name` (the model's estimate for each of `houses`, which must be in the
    same order as `results`) and `name`__pct_error against the actual total_value."""
    results[name] = np.expm1(model.predict(preprocessor.transform(houses[features])))
    results[f"{name}__pct_error"] = (results[name] - results[TARGET]) / results[TARGET]


def validate_houses(n=VALIDATION_SIZE, parcel_ids=None, random_state=0):
    """Estimates from every feature set x model for the selected test-set houses, with
    each estimate's percent error against the actual total_value."""
    _, test = load_splits()
    houses = select_houses(test, n, parcel_ids, random_state)
    results = houses[DISPLAY_COLUMNS].reset_index(drop=True)
    for feature_set, features in FEATURE_SETS.items():
        for model_name in MODEL_TYPES:
            preprocessor, model = predict.load_model(feature_set, model_name)
            add_estimates(results, houses, f"{feature_set}__{model_name}", preprocessor, model, features)
    return results


def summarize(results):
    """Mean and worst absolute percent error per model, best first."""
    errors = results.filter(like="__pct_error").abs()
    errors.columns = errors.columns.str.removesuffix("__pct_error")
    return errors.agg(["mean", "max"]).T.sort_values("mean")


def print_validation(results):
    error_cols = [c for c in results if c.endswith("__pct_error")]
    shown = results[["parcel_id", "area", TARGET] + error_cols].copy()
    shown["parcel_id"] = shown["parcel_id"].astype("Int64")
    shown[TARGET] = shown[TARGET].map("${:,.0f}".format)
    shown[error_cols] = shown[error_cols].map("{:+.1%}".format)
    print(shown.rename(columns=lambda c: c.removesuffix("__pct_error")).to_string(index=False))
    print()
    print(summarize(results).to_string(float_format=lambda x: f"{x:.1%}"))


def run():
    results = validate_houses()
    print_validation(results)
    save_dataframe(results, resolve_path(load_config(), "reports"), "validation_houses.csv")
    return results


if __name__ == "__main__":
    run()
