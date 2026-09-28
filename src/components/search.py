"""Retrieval layer: similar-house search and retrospective deal-finding.

Both reuse a feature set's saved preprocessor (output/models/{feature_set}_preprocessor.joblib,
written by training.py) so retrieval happens in the same feature space the models were
trained on, rather than a separate embedding - no vector DB, just NearestNeighbors over
the existing preprocessed feature matrix. Scope is Mecklenburg only, matching training.py.

find_deals is retrospective by design: this dataset has no live listing feed, only
historical arms-length sale_price and the trained model. The model estimates the 2023
assessed value, which usually sits below market price, so deal_score also carries that
normal assessed-vs-market gap, not only genuine bargains. A "deal" here means
a past sale that went for notably less than the model would have predicted for that
property's features - not a live-market recommendation.

Known limitation, not fixable from this data alone: after the price/sqft floor and
recency window below, some flagged "deals" are still most likely flips, not mispricing -
a distressed property sells cheap reflecting its as-is condition, gets renovated, and the
county's *current* total_value/features reflect the renovated state, not the condition at
sale time. Verified this isn't market appreciation (same pattern holds even for sales
within the last few months) or model error (predictions agree with each property's own
assessed total_value within ~10% - checked on the earlier per-county models). Fixing this properly would need a renovation-date or
property-condition field, which the source data doesn't have - callers should treat
find_deals output as a starting point to investigate, not a verified "deal" list.
"""

import functools
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.components.predict import DEFAULT_FEATURE_SET, DEFAULT_MODEL_NAME, load_model
from src.components.training import FEATURE_COLUMNS, FEATURE_SETS
from src.utils.common import read_dataframe, resolve_path
from src.utils.load_config import load_config

SIMILAR_HOUSE_DISPLAY_COLUMNS = [
    "parcel_id", "area", "land_use_class", "heated_area", "bedrooms", "bathrooms",
    "acreage", "total_value", "neighborhood_code",
]
DEAL_DISPLAY_COLUMNS = [
    "parcel_id", "area", "sale_price", "predicted_value", "deal_score", "sale_date",
    "land_use_class", "heated_area", "bedrooms", "bathrooms", "neighborhood_code",
]
# lookup_property returns every model feature, so its result can be passed straight
# into predict_total_value / find_similar_houses instead of relying on imputed values.
PROPERTY_DISPLAY_COLUMNS = [
    "parcel_id", "situs_street_num", "situs_street_name", "situs_city", "neighborhood_code",
    "total_value", "land_value", "building_value", "sale_price", "sale_date",
] + FEATURE_COLUMNS

# A sale below this flat price/sqft is treated as implausible, not a great deal - e.g. a
# $32k sale of a 3,004 sqft house ($10.7/sqft) is almost certainly a family transfer or
# distressed/related-party sale slipping through the arms_length_sale price/date
# heuristic, not a real market transaction. Deliberately a flat floor rather than a
# fraction of the row's own neighborhood_median_price_per_sqft: that median is itself
# computed from arms_length_sale rows and can be dragged down by these same bad sales in
# small/low-sample neighborhoods (e.g. one 8-sale neighborhood's "median" was $42/sqft).
# $50/sqft cuts off roughly the bottom 1% of genuine Mecklenburg arms-length sales
# (county's own distribution: 1st percentile ~$53/sqft, median ~$186/sqft).
MIN_PRICE_PER_SQFT = 50

# Even past the price/sqft floor, comparing an old sale_price to *today's* predicted/
# assessed value conflates "underpriced at the time" with "the market went up since
# then" - verified this isn't a false lead: the model's predictions agree closely
# (within ~10%) with each property's own independently-assessed total_value, so the
# "deals" are real gaps between sale_price and current value, not model error. But
# Mecklenburg has appreciated substantially over the
# arms_length_sale 10-year lookback, especially in gentrifying neighborhoods, so an old,
# fairly-priced-at-the-time sale can look like a huge "deal" against today's value.
# Restricting to recent sales keeps that gap small enough to read as mispricing rather
# than market growth.
MAX_SALE_AGE_YEARS = 2


@functools.lru_cache(maxsize=1)
def _load_features_df(features_path):
    return read_dataframe(features_path, "properties_features.csv")


@functools.lru_cache(maxsize=None)
def _index(feature_set):
    """The feature table + a NearestNeighbors index over its preprocessed features for
    one feature set, cached so repeated queries don't reload/refit. Houses with no
    total_value are left out - a comp without a value is no use."""
    df = _load_features_df(resolve_path(load_config(), "features_data"))
    df = df[df["total_value"].notna()].reset_index(drop=True)
    preprocessor, _ = load_model(feature_set)
    nn = NearestNeighbors().fit(preprocessor.transform(df[FEATURE_SETS[feature_set]]))
    return df, nn


def find_similar_houses(features, k=5, feature_set=DEFAULT_FEATURE_SET):
    """k nearest comps to `features` (dict of training.FEATURE_SETS[feature_set] -> value),
    by distance in the same preprocessed feature space the models were trained on."""
    df, nn = _index(feature_set)
    preprocessor, _ = load_model(feature_set)
    query_row = pd.DataFrame([{col: features.get(col, np.nan) for col in FEATURE_SETS[feature_set]}])

    distances, indices = nn.kneighbors(preprocessor.transform(query_row), n_neighbors=k)
    results = df.iloc[indices[0]][SIMILAR_HOUSE_DISPLAY_COLUMNS].copy()
    results["distance"] = distances[0]
    return results.to_dict(orient="records")


def find_deals(model_name=DEFAULT_MODEL_NAME, feature_set=DEFAULT_FEATURE_SET, top_n=10,
               min_sale_price=10_000, area=None):
    """Past arms-length sales where sale_price came in well below what the trained
    model would have predicted for that property - the biggest retrospective deals.
    deal_score = (predicted_value - sale_price) / predicted_value; higher = bigger deal.
    `area` (e.g. "MATTHEWS", "CHARLOTTE_3") limits results to one area.
    """
    df = _load_features_df(resolve_path(load_config(), "features_data"))
    preprocessor, model = load_model(feature_set, model_name)

    sale_date = pd.to_datetime(df["sale_date"], format="mixed")
    recent_cutoff = pd.Timestamp(datetime.now()) - pd.DateOffset(years=MAX_SALE_AGE_YEARS)

    sold = df[
        df["arms_length_sale"].astype(bool)
        & df["sale_price"].notna()
        & (df["sale_price"] >= min_sale_price)
        & df["heated_area"].gt(0)
        & (df["sale_price"] / df["heated_area"] >= MIN_PRICE_PER_SQFT)
        & (sale_date >= recent_cutoff)
        & (area is None or df["area"] == area)
    ].copy()

    sold["predicted_value"] = np.expm1(model.predict(preprocessor.transform(sold[FEATURE_SETS[feature_set]])))
    sold["deal_score"] = (sold["predicted_value"] - sold["sale_price"]) / sold["predicted_value"]

    return sold.nlargest(top_n, "deal_score")[DEAL_DISPLAY_COLUMNS].to_dict(orient="records")


def lookup_property(address=None, parcel_id=None, limit=5):
    """Properties matching a parcel_id or a street address like "221 Altondale Ave, Charlotte".

    Street names are stored abbreviated ("ALTONDALE AV"), so an address matches on its
    house number plus the first word of the street name - "Ave"/"Avenue" don't matter.
    If a Mecklenburg city name appears in the address, results are limited to it.
    Can return several rows (same street name in different towns, or several units at one
    number), so the caller picks the right one.
    """
    df = _load_features_df(resolve_path(load_config(), "features_data"))
    if parcel_id is not None:
        match = df[df["parcel_id"] == float(parcel_id)]
    elif address:
        parsed = re.match(r"\s*(\d+)\s+([A-Za-z0-9]+)", address)
        if not parsed:
            raise ValueError(f"expected an address starting with a house number, got {address!r}")
        number, street_word = int(parsed.group(1)), parsed.group(2).upper()
        match = df[
            (df["situs_street_num"] == number)
            & (df["situs_street_name"].str.split().str[0] == street_word)
        ]
        cities = [c for c in match["situs_city"].dropna().unique() if c in address.upper()]
        if cities:
            match = match[match["situs_city"].isin(cities)]
    else:
        raise ValueError("give an address or a parcel_id")

    results = match.head(limit)[PROPERTY_DISPLAY_COLUMNS]
    return results.astype(object).where(results.notna(), None).to_dict(orient="records")


def area_stats(area=None):
    """Typical values per area (or for one area): house count, median assessed value, size
    and age, median $/sqft of recent arms-length sales, and median price_momentum and
    accessibility. Useful for comparing areas, and as realistic stand-in feature values
    when the user describes a house without an exact address."""
    df = _load_features_df(resolve_path(load_config(), "features_data"))
    if area is not None:
        df = df[df["area"] == area]
        if df.empty:
            raise ValueError(f"unknown area {area!r}")

    stats = df.groupby("area").agg(
        houses=("total_value", "size"),
        median_total_value=("total_value", "median"),
        median_heated_area=("heated_area", "median"),
        median_year_built=("year_built", "median"),
        median_price_momentum=("price_momentum", "median"),
        median_education_nearest_miles=("education_nearest_miles", "median"),
        median_emergency_medical_miles=("emergency_medical_miles", "median"),
        median_places_count=("places_count", "median"),
    )
    sales = df[df["arms_length_sale"].astype(bool)]
    stats["median_sale_price_per_sqft"] = sales.groupby("area")["price_per_sqft"].median()
    return stats.round(3).reset_index().to_dict(orient="records")


if __name__ == "__main__":
    example = {
        "land_use_class": "single_family", "acreage": 0.3, "year_built": 2005,
        "heated_area": 2000, "bedrooms": 3, "bathrooms": 2, "latitude": 35.2,
        "longitude": -80.8, "property_age": 20, "neighborhood_median_price_per_sqft": 180,
        "neighborhood_sale_count": 50, "price_momentum": 0.15, "area": "CHARLOTTE_9", "education_nearest_miles": 0.5,
        "places_count": 8000, "health_care_share": 0.07, "activity_share": 0.03,
        "emergency_fire_miles": 1.2, "emergency_police_miles": 2.0, "emergency_medical_miles": 2.5,
    }
    print("--- similar houses ---")
    for row in find_similar_houses(example, k=5):
        print(row)

    print("\n--- top deals ---")
    for row in find_deals(top_n=5):
        print(row)
