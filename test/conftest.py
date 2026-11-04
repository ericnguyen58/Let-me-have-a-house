"""Shared fixtures. Tests run without the real data: `features` builds a small synthetic
feature table with every column the pipeline uses. Tests marked `data` use the local
data/ and output/models/ files and are skipped when those are missing, as in CI."""

import os

import numpy as np
import pandas as pd
import pytest

from src.components import search
from src.components.feature_engineering import add_sale_features
from src.components.training import NUMERIC_FEATURES
from src.utils.common import resolve_path
from src.utils.load_config import load_config

AREA_PREMIUM = {"CHARLOTTE_1": 1.0, "CHARLOTTE_2": 1.4, "MATTHEWS": 1.2}


def local_data_files():
    cfg = load_config()
    models = resolve_path(cfg, "models")
    return [
        resolve_path(cfg, "features_data") + "properties_features.csv",
        models + "production.json",
        models + "out_of_fold_estimates.csv",
        resolve_path(cfg, "raw_data") + "zillow/county_zhvi.csv",
        resolve_path(cfg, "raw_data") + "merklenburg/Cama_Table_transformed.csv",
    ]


def pytest_collection_modifyitems(config, items):
    missing = [p for p in local_data_files() if not os.path.exists(p)]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"needs local data, missing {os.path.basename(missing[0])}")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


def make_features(n=600, seed=0):
    """A feature table shaped like data/features/properties_features.csv, where value
    grows with size, age and area, and every sale is arms-length in 2020-2022."""
    rng = np.random.default_rng(seed)
    area = rng.choice(list(AREA_PREMIUM), n)
    heated_area = rng.uniform(900, 4000, n).round()
    year_built = rng.integers(1950, 2022, n)
    total_value = (
        150 * heated_area * pd.Series(area).map(AREA_PREMIUM).to_numpy() * (1 + (year_built - 1950) / 200)
    ) * rng.lognormal(0, 0.05, n)
    sale_date = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    df = pd.DataFrame(
        {
            "county": "Mecklenburg",
            "parcel_id": np.arange(1, n + 1, dtype=float),
            "situs_street_num": rng.integers(4000, 9999, n),
            "situs_street_name": rng.choice(["KELLY WOODS LN", "MAIN ST", "SETON HOUSE LN"], n),
            "situs_city": "CHARLOTTE",
            "neighborhood_code": [f"{a}-{i}" for a, i in zip(area, rng.integers(0, 4, n), strict=True)],
            "total_value": total_value.round(-2),
            "land_value": (total_value * 0.3).round(-2),
            "building_value": (total_value * 0.7).round(-2),
            "sale_price": (total_value * rng.lognormal(0.1, 0.05, n)).round(-2),
            "sale_date": sale_date.strftime("%Y-%m-%d"),
            "arms_length_sale": True,
            "elementary_school": "Alexander",
            "middle_school": "Alexander",
            "high_school": "North",
            "land_use_class": rng.choice(["single_family", "townhouse", "condo"], n),
            "area": area,
            "acreage": rng.uniform(0.05, 1, n),
            "year_built": year_built,
            "heated_area": heated_area,
            "bedrooms": rng.integers(2, 6, n),
            "bathrooms": rng.integers(2, 9, n) / 2,
            "latitude": 35.2 + rng.normal(0, 0.05, n),
            "longitude": -80.8 + rng.normal(0, 0.05, n),
            "property_age": 2026 - year_built,
            "in_floodplain": rng.integers(0, 2, n),
        }
    )
    for column in NUMERIC_FEATURES:
        if column not in df and column not in ["neighborhood_median_price_per_sqft", "neighborhood_sale_count"]:
            df[column] = rng.uniform(0, 10, n)
    df["price_per_sqft"] = df["sale_price"] / df["heated_area"]
    # the real file carries the sale-derived features computed on every row
    return add_sale_features(df.drop(columns="price_momentum"))


@pytest.fixture
def features():
    return make_features()


@pytest.fixture(scope="module")
def module_features():
    """make_features() shared by a module's tests, which must not modify it."""
    return make_features()


@pytest.fixture(scope="session")
def synthetic_training():
    """training.py's split and training on make_features(), done once per test session (it
    takes some seconds): (training config, train, test, train_models' results)."""
    from src.components import training

    cfg = {"test_size": 0.25, "random_state": 0, "search_iter": 2, "cv_folds": 2}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(training, "test_address_parcels", lambda: [])  # those need the real file
        train, test = training.split_with_aggregates(make_features().drop(columns=training.SALE_DERIVED_FEATURES), cfg)
    return cfg, train, test, training.train_models(train, test, cfg)


@pytest.fixture
def search_on(monkeypatch):
    """Point search.py at a given feature table instead of the real file, and clear its
    caches before and after so no synthetic rows leak into tests that use the real data."""
    caches = [search._index, search._area_medians, search._priced_houses]

    def use(df):
        for cache in caches:
            cache.cache_clear()
        monkeypatch.setattr(search, "_load_features_df", lambda path: df)

    yield use
    for cache in caches:
        cache.cache_clear()
