"""Predict `total_value` for a single Mecklenburg house using a trained model.

Loads the preprocessor + model artifacts training.py saved to output/models/
(`{feature_set}_preprocessor.joblib`, `{feature_set}_{model_name}.joblib`) and
applies them to one property's feature values. The default feature set is
with_neighborhood_price, the one meant for estimates; without_neighborhood_price is
for studying what `area` and the accessibility features are worth.

Missing feature keys are left as NaN and handled by the saved preprocessor the same
way training does (median/most-frequent imputation) - callers don't need to supply
every field. Imputed location fields (`area`, latitude/longitude, the accessibility
features) fall back to county-wide typical values, so estimates are much weaker
without them.
"""

import functools
import joblib
import numpy as np
import pandas as pd

from src.components.training import FEATURE_SETS, MODEL_TYPES
from src.utils.common import resolve_path
from src.utils.load_config import load_config

DEFAULT_FEATURE_SET = "with_neighborhood_price"
DEFAULT_MODEL_NAME = "random_forest"


@functools.lru_cache(maxsize=None)
def load_artifact(models_path, filename):
    """Cached joblib load - shared with search.py so both modules reuse the same
    in-memory preprocessor/model instead of loading their own copies."""
    return joblib.load(models_path + filename)


def load_model(feature_set=DEFAULT_FEATURE_SET, model_name=DEFAULT_MODEL_NAME):
    """(preprocessor, model) for a feature set, from output/models/."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}, expected one of {list(FEATURE_SETS)}")
    if model_name not in MODEL_TYPES:
        raise ValueError(f"model {model_name!r} is not in use, expected one of {MODEL_TYPES}")
    models_path = resolve_path(load_config(), "models")
    preprocessor = load_artifact(models_path, f"{feature_set}_preprocessor.joblib")
    model = load_artifact(models_path, f"{feature_set}_{model_name}.joblib")
    return preprocessor, model


def predict_total_value(features, feature_set=DEFAULT_FEATURE_SET, model_name=DEFAULT_MODEL_NAME):
    """features: dict of training.FEATURE_SETS[feature_set] -> value (missing keys become NaN).

    Returns the predicted total_value in dollars (original scale, log1p inverted).
    """
    preprocessor, model = load_model(feature_set, model_name)
    row = pd.DataFrame([{col: features.get(col, np.nan) for col in FEATURE_SETS[feature_set]}])
    log_pred = model.predict(preprocessor.transform(row))[0]
    return float(np.expm1(log_pred))


if __name__ == "__main__":
    example = {
        "land_use_class": "single_family",
        "area": "CHARLOTTE_1",
        "acreage": 0.46323194,
        "year_built": 1950,
        "heated_area": 1017,
        "bedrooms": 2,
        "bathrooms": 1,
        "latitude": 35.33386622,
        "longitude": -80.87131071,
        "property_age": 76,
        "neighborhood_median_price_per_sqft": 161.31783787093235,
        "neighborhood_sale_count": 46,
        "price_momentum": 0.2,
        "education_nearest_miles": 0.5,
        "places_count": 6000,
        "health_care_share": 0.06,
        "activity_share": 0.03,
        "emergency_fire_miles": 1.2,
        "emergency_police_miles": 2.0,
        "emergency_medical_miles": 2.5,
    }
    for feature_set in FEATURE_SETS:
        for model_name in MODEL_TYPES:
            pred = predict_total_value(example, feature_set=feature_set, model_name=model_name)
            print(f"{feature_set} {model_name}: ${pred:,.0f}")
