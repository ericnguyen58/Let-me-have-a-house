"""Add modeling features on top of the cleaned properties table.

Reads data/processed/properties_clean.csv and adds property_age, price_per_sqft, and
per-neighborhood price aggregates. These aggregates are computed on the full clean
dataset rather than a training-only split, which is a mild leakage caveat worth
revisiting once a real train/test split exists (see training.py, not yet built).
"""

from datetime import datetime
import numpy as np

from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

CURRENT_YEAR = datetime.now().year
EARTH_RADIUS = 3959 #miles

def add_property_age(df):
    df["property_age"] = CURRENT_YEAR - df["year_built"]
    return df


def add_price_per_sqft(df):
    df["price_per_sqft"] = np.nan
    valid = df["arms_length_sale"] & df["heated_area"].gt(0)
    df.loc[valid, "price_per_sqft"] = df.loc[valid, "sale_price"] / df.loc[valid, "heated_area"]
    return df

def accessibility_score(house_df, nc_places_df):
    house_df = house_df[['latitude','longitude']].dropna()
    nc_places_df = nc_places_df[['lat','lon']].dropna()

    accessibility_score = ()

def add_neighborhood_aggregates(df):
    group_cols = ["county", "neighborhood_code"]
    agg = (
        df.loc[df["price_per_sqft"].notna()]
        .groupby(group_cols)["price_per_sqft"]
        .agg(neighborhood_median_price_per_sqft="median", neighborhood_sale_count="count")
        .reset_index()
    )
    return df.merge(agg, on=group_cols, how="left")


def build_features(df):
    df = add_property_age(df)
    df = add_price_per_sqft(df)
    df = add_neighborhood_aggregates(df)
    return df


def run():
    cfg = load_config()
    processed_path = resolve_path(cfg, "processed_data")
    features_path = resolve_path(cfg, "features_data")

    clean = read_dataframe(processed_path, "properties_clean.csv", parse_dates=["sale_date"], low_memory=False)
    clean["arms_length_sale"] = clean["arms_length_sale"].astype(bool)

    features = build_features(clean)
    save_dataframe(features, features_path, "properties_features.csv")
    return features


if __name__ == "__main__":
    run()
