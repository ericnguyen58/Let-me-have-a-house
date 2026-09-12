"""Add modeling features on top of the cleaned properties table.

Reads data/processed/properties_clean.csv and adds property_age, price_per_sqft,
per-neighborhood price aggregates, and an accessibility score (proximity to
groceries/parks/schools/etc, see ACCESSIBILITY_CATEGORIES). The neighborhood
aggregates are computed on the full clean dataset rather than a training-only
split, which is a mild leakage caveat worth revisiting once a real train/test
split exists (see training.py, not yet built).
"""

from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

CURRENT_YEAR = datetime.now().year
EARTH_RADIUS = 3959 #miles

# Curated "surrounding places" categories for the accessibility score (out of the
# ~1765 raw `cat` values in nc_places_map.csv). weight = how much the category
# matters to housing accessibility; tau_miles = distance-decay range, i.e. the
# score falls to ~37% at tau_miles and ~5% at 3x tau_miles. `cats` lets one
# score bucket pool several raw categories (e.g. all school levels); it
# defaults to the bucket's own name when omitted.
ACCESSIBILITY_CATEGORIES = {
    "grocery_store": {"weight": 1.0, "tau_miles": 1.0},
    "park": {"weight": 0.8, "tau_miles": 0.5},
    "school": {
        "weight": 0.8,
        "tau_miles": 1.0,
        "cats": ["school", "elementary_school", "middle_school", "high_school", "private_school"],
    },
    "hospital": {"weight": 0.7, "tau_miles": 3.0},
    "pharmacy": {"weight": 0.6, "tau_miles": 1.5},
    "restaurant": {"weight": 0.5, "tau_miles": 1.0},
    "public_transit_facility_or_service": {"weight": 0.5, "tau_miles": 1.0},
}

def add_property_age(df):
    df["property_age"] = CURRENT_YEAR - df["year_built"]
    return df


def add_price_per_sqft(df):
    df["price_per_sqft"] = np.nan
    valid = df["arms_length_sale"] & df["heated_area"].gt(0)
    df.loc[valid, "price_per_sqft"] = df.loc[valid, "sale_price"] / df.loc[valid, "heated_area"]
    return df

def nearest_distance(house_lat, house_lon, place_lat, place_lon):
    """Distance (miles) and index of the nearest place for each house."""
    house_rad = np.radians(np.column_stack([house_lat, house_lon]))
    place_rad = np.radians(np.column_stack([place_lat, place_lon]))
    tree = BallTree(place_rad, metric="haversine")
    dist_rad, idx = tree.query(house_rad, k=1)
    return dist_rad[:, 0] * EARTH_RADIUS, idx[:, 0]


def category_accessibility_score(house_df, nc_places_df, cats, tau_miles):
    """exp(-distance/tau) closeness (0-1) to the nearest place in `cats`, indexed like house_df.

    NaN for houses missing coordinates (dropped before the nearest-neighbor query,
    reindexed back afterwards) rather than imputed.
    """
    places = nc_places_df.loc[nc_places_df["cat"].isin(cats), ["lat", "lon"]].dropna()
    houses = house_df[["latitude", "longitude"]].dropna()
    if places.empty or houses.empty:
        return pd.Series(np.nan, index=house_df.index)

    distance_miles, _ = nearest_distance(
        houses["latitude"], houses["longitude"], places["lat"], places["lon"]
    )
    score = pd.Series(np.exp(-distance_miles / tau_miles), index=houses.index)
    return score.reindex(house_df.index)


def accessibility_score(house_df, nc_places_df, categories=None):
    """Weighted-average closeness (0-1) to a curated set of nearby place categories.

    NaN for houses missing coordinates; ignored (not imputed) otherwise.
    """
    categories = categories or ACCESSIBILITY_CATEGORIES
    scores = pd.DataFrame({
        name: category_accessibility_score(house_df, nc_places_df, cfg.get("cats", [name]), cfg["tau_miles"])
        for name, cfg in categories.items()
    })
    weights = pd.Series({name: cfg["weight"] for name, cfg in categories.items()})
    # skipna=False: a house missing coordinates is NaN in every category column,
    # so the row should sum to NaN rather than silently treating them as 0.
    return scores.mul(weights).sum(axis=1, skipna=False) / weights.sum()

def add_neighborhood_aggregates(df):
    group_cols = ["county", "neighborhood_code"]
    agg = (
        df.loc[df["price_per_sqft"].notna()]
        .groupby(group_cols)["price_per_sqft"]
        .agg(neighborhood_median_price_per_sqft="median", neighborhood_sale_count="count")
        .reset_index()
    )
    return df.merge(agg, on=group_cols, how="left")


def build_features(df, nc_places_df):
    df = add_property_age(df)
    df = add_price_per_sqft(df)
    df = add_neighborhood_aggregates(df)
    df["access"] = accessibility_score(df, nc_places_df)
    return df


def run():
    cfg = load_config()
    processed_path = resolve_path(cfg, "processed_data")
    features_path = resolve_path(cfg, "features_data")

    clean = read_dataframe(processed_path, "properties_clean.csv", parse_dates=["sale_date"], low_memory=False)
    clean["arms_length_sale"] = clean["arms_length_sale"].astype(bool)
    places = read_dataframe(processed_path, "nc_places_map.csv")

    features = build_features(clean, places)
    save_dataframe(features, features_path, "properties_features.csv")
    return features


if __name__ == "__main__":
    run()
    # places = pd.read_csv('/Users/a70411/Let-me-have-a-house/data/processed/nc_places_map.csv')
    # houses = pd.read_csv('/Users/a70411/Let-me-have-a-house/data/features/properties_features.csv')
    #
    # print(accessibility_score(houses, places).describe())

