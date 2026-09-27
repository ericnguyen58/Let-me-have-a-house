"""Merge the four raw county sources into one schema, then clean the result.

The loaders/mappers below are promoted from `notebooks/eda.ipynb`, where they were
validated against the raw data (county identification, shared-schema mapping, Wake's
building-card rollup). This module adds the cleaning step the notebook didn't do and
turns it into a reusable, rerunnable pipeline stage.

Note: `data/processed/properties_combined.csv` (the notebook's raw-merge output) is
left untouched by this module - only `properties_clean.csv` is written here.

Scope is Mecklenburg only. The Buncombe/Guilford/Wake loaders and mappers are
commented out rather than deleted, so they can be switched back on. After cleaning,
each parcel gets an `area` (see assign_area) so houses can be compared across towns
and parts of Charlotte.
"""

import json
import re
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.utils.common import resolve_path, save_dataframe
from src.utils.load_config import load_config

COMMON_COLUMNS = [
    "county", "parcel_id", "owner_name", "situs_street_num", "situs_street_name", "situs_city", "situs_zip",
    "sale_price", "sale_date", "deed_book", "deed_page", "land_value", "building_value", "total_value",
    "acreage", "land_use_class", "neighborhood_code", "year_built", "heated_area", "bedrooms", "bathrooms",
    "latitude", "longitude",
]

# Canonical land-use buckets. County taxonomies don't line up exactly, so this is a
# best-effort mapping of the most common raw values into a small set every county
# rolls up into; anything present but unrecognized falls through to "other" and
# anything missing stays missing.
LAND_USE_MAP = {
    "R": "single_family",
    "100": "single_family",
    "SINGLE FAMILY RESIDENTIAL": "single_family",
    "SINGLE FAMILY RESIDENTIAL - ACREAGE": "single_family",
    "RESIDENTIAL": "single_family",
    "TOWN HOUSE  SFR": "townhouse",
    "TOWNHOUSE": "townhouse",
    "CONDOMINIUM": "condo",
    "311": "condo",
    "V": "vacant_land",
    "N": "vacant_land",
    "VACANT": "vacant_land",
    "VACANT LAND": "vacant_land",
}

MIN_SALE_YEAR = 1900
ARMS_LENGTH_MIN_PRICE = 10_000
ARMS_LENGTH_MAX_PRICE = 20_000_000
ARMS_LENGTH_LOOKBACK_YEARS = 10

# Keywords identifying an owner_name as a business/institutional entity rather than an
# individual person (LLCs, trusts, banks, HOAs, government bodies, etc.) - used to drop
# non-individual-owned properties, which are noisier comps for a homebuyer-focused dataset.
# Missing/blank owner_name is left in (can't confirm either way) rather than dropped.
NON_INDIVIDUAL_OWNER_KEYWORDS = [
    "LLC", "L L C", "INC", "INCORPORATED", "CORP", " CO ", " CO,", " CO.", "COMPANY",
    "LP", "LLP", "LTD", "TRUST", "ESTATE OF", "BANK", "CHURCH", "ASSOC", "ASSN",
    "PARTNERSHIP", "PROPERTIES", "INVESTMENTS", "HOLDINGS", "MGMT", "MANAGEMENT",
    "FOUNDATION", "MINISTR", "HOMEOWNERS", " HOA", "AUTHORITY", "DEVELOPMENT",
    "BUILDERS", "CONSTRUCTION", "REALTY", "ENTERPRISES", "VENTURES", "CAPITAL",
    "UNIVERSITY", "HOSPITAL", "SCHOOL", "GOVERNMENT", "CITY OF", "COUNTY OF", "STATE OF",
    "UNITED STATES", "HABITAT FOR HUMANITY",
]
NON_INDIVIDUAL_OWNER_PATTERN = "|".join(re.escape(k) for k in NON_INDIVIDUAL_OWNER_KEYWORDS)

# Residential-only land_use_class values - drops "other"/"vacant_land", which catches large
# institutional parcels (hospitals, colleges, airports, factories) that have no owner_name
# giveaway at all (e.g. "FUJIFILM DIOSYNTH BIOTECHNOLOGIES", "RALEIGH DURHAM INTERNATIONAL"),
# since none of them are ever tagged single_family/condo/townhouse.
RESIDENTIAL_LAND_USE_CLASSES = ["single_family", "condo", "townhouse"]

# Sane heated_area bounds (sqft) for a residential unit - catches county-assessor mislabeling
# that survives both filters above: a handful of parcels up to 350k+ sqft are still tagged
# single_family despite obviously being commercial/apartment buildings. Rows with heated_area
# missing (can't verify either way) are left in, same policy as blank owner_name.
MIN_HEATED_AREA = 300
MAX_HEATED_AREA = 10_000

# UNINC is unincorporated land scattered between towns, and STALLINGS is 52 parcels of a
# town that sits mostly in Union County - neither works as an area of its own, so each
# of their parcels takes the majority city of its nearest labeled parcels.
MERGE_INTO_NEAREST_CITY = ["UNINC", "STALLINGS"]
NEAREST_CITY_NEIGHBORS = 15

# Charlotte is ~75% of the county, so it's split into areas made of whole neighborhoods
# (neighborhood_code), each at most MAX_AREA_SIZE houses. Splitting ~223k houses in half
# repeatedly lands at ~28k per area; the largest single neighborhood is ~1.5k houses, so
# a split never overshoots by much. Other towns keep their natural size.
SPLIT_CITY = "CHARLOTTE"
MAX_AREA_SIZE = 30_000


# Buncombe's ArcGIS export carries these as JSON strings rather than numbers, so after
# concatenation with the other counties these columns end up as mixed-type `object`
# columns unless coerced.
NUMERIC_COLUMNS = [
    "situs_street_num", "situs_zip", "sale_price", "land_value", "building_value",
    "total_value", "acreage", "year_built", "heated_area", "bedrooms", "bathrooms",
    "latitude", "longitude",
]


# def load_arcgis_json(path):
#     """Flatten an ArcGIS FeatureServer query response into a DataFrame.
#     Keeps attributes as columns; geometry is kept as a raw dict in `_geometry`.
#     """
#     with open(path) as f:
#         payload = json.load(f)
#     df = pd.DataFrame([f["attributes"] for f in payload["features"]])
#     df["_geometry"] = [f.get("geometry") for f in payload["features"]]
#     return df


def load_mecklenburg(raw_path):
    return pd.read_csv(
        raw_path + "merklenburg/Cama_Table_transformed.csv",
        sep=";", encoding="utf-8-sig", decimal=",", low_memory=False,
    )


# def load_buncombe(raw_path):
#     return load_arcgis_json(raw_path + "buncombe/buncombe.json")


# def load_buncombe_building(raw_path):
#     """Separate ArcGIS layer, building characteristics only (no owner/sale/land
#     fields) - joined onto the parcel layer in `bun_to_common` via PIN."""
#     return load_arcgis_json(raw_path + "buncombe_building/buncombe_building.json")


# def load_guilford(raw_path):
#     return load_arcgis_json(raw_path + "guilford/guilford.json")


# def load_wake(raw_path):
#     return pd.read_excel(raw_path + "wake/RealEstData09022026.xlsx", sheet_name="REALINFO")


# def polygon_centroid(rings):
#     """Approximate centroid: mean of the exterior ring's vertices (lon, lat)."""
#     if not rings:
#         return None, None
#     pts = rings[0]
#     lons = [p[0] for p in pts]
#     lats = [p[1] for p in pts]
#     return sum(lats) / len(lats), sum(lons) / len(lons)


def meck_to_common(df):
    out = pd.DataFrame({
        "county": "Mecklenburg",
        "parcel_id": df["parcelid"],
        "owner_name": (df["ownrlstnme"].fillna("") + " " + df["ownrfrstnme"].fillna("")).str.strip(),
        "situs_street_num": df["streetnumber"],
        "situs_street_name": df["streetname"],
        "situs_city": df["loccity"],
        "situs_zip": pd.NA,  # only the owner's mailing zip is present, not situs
        "sale_price": df["saleprice"],
        "sale_date": pd.to_datetime(df["saledate"], format="%d/%m/%y %H:%M", errors="coerce"),
        "deed_book": df["deed_book"],
        "deed_page": df["deed_page"],
        "land_value": df["totlandval"],
        "building_value": df["totalbldgval"],
        "total_value": df["totalvalue"],
        "acreage": df["gisacres"],
        "land_use_class": df["landuse_description"],
        "neighborhood_code": df["neighborhood"],
        "year_built": df["yearbuilt"],
        "heated_area": df["heatedarea"],
        "bedrooms": df["bedrooms"],
        "bathrooms": df["fullbath"] + 0.5 * df["halfbath"],
        "latitude": df["xcoord"],   # NB: this county's export has x/y swapped vs. the usual lon/lat convention
        "longitude": df["ycoord"],  # (xcoord holds ~35 = latitude, ycoord holds ~-80 = longitude for Mecklenburg)
    })
    return out[COMMON_COLUMNS]


# def buncombe_building_rollup(df):
#     """The building layer is one row per building card (BldgNo); collapse to one row
#     per PIN, summing across cards (a parcel can carry more than one structure)."""
#     df = df.copy()
#     df["bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]
#     return df.groupby("PIN", as_index=False).agg(
#         heated_area=("SqFeet", "sum"),
#         bedrooms=("Bedroom", "sum"),
#         bathrooms=("bathrooms", "sum"),
#         year_built=("YearBuilt", "min"),
#     )


# def bun_to_common(df, building_df):
#     lat, lon = zip(*(polygon_centroid(g["rings"]) if g else (None, None) for g in df["_geometry"]))
#     df = df.copy()
#     df["latitude"], df["longitude"] = lat, lon

#     # PIN is the shared key, but may not share dtype across the two layers - join on
#     # a stringified copy rather than the raw column.
#     building = buncombe_building_rollup(building_df)
#     df["_pin_key"] = df["PIN"].astype(str)
#     building["_pin_key"] = building["PIN"].astype(str)
#     df = df.merge(building.drop(columns="PIN"), on="_pin_key", how="left")

#     out = pd.DataFrame({
#         "county": "Buncombe",
#         "parcel_id": df["PIN"],
#         "owner_name": df["Owner"],
#         "situs_street_num": df["HouseNumber"],
#         "situs_street_name": (df["StreetName"].fillna("") + " " + df["StreetType"].fillna("")).str.strip(),
#         "situs_city": pd.NA,  # CityName is the owner's mailing city, not situs
#         "situs_zip": pd.NA,   # Zipcode is the owner's mailing zip, not situs
#         "sale_price": df["SalePrice"],
#         "sale_date": pd.to_datetime(df["DeedDate"], format="%Y%m%d", errors="coerce"),  # deed date, used as sale-date proxy
#         "deed_book": df["DeedBook"],
#         "deed_page": df["DeedPage"],
#         "land_value": df["LandValue"],
#         "building_value": df["BuildingValue"],
#         "total_value": df["TotalMarketValue"],
#         "acreage": df["Acreage"],
#         "land_use_class": df["LandUse"].replace("", pd.NA).fillna(df["Class"]),
#         "neighborhood_code": df["NeighborhoodCode"],
#         "year_built": df["year_built"],    # from the building layer
#         "heated_area": df["heated_area"],  # from the building layer
#         "bedrooms": df["bedrooms"],        # from the building layer
#         "bathrooms": df["bathrooms"],      # from the building layer
#         "latitude": df["latitude"],
#         "longitude": df["longitude"],
#     })
#     return out[COMMON_COLUMNS]


# def gui_to_common(df):
#     out = pd.DataFrame({
#         "county": "Guilford",
#         "parcel_id": df["PIN_PLUS_EXTENSION"],
#         "owner_name": df["OWNER1_FULLNAME"].fillna(df["OWNER"]),
#         "situs_street_num": df["SITUS_STREET_NUMBER"],
#         "situs_street_name": (df["SITUS_STREET"].fillna("") + " " + df["SITUS_STREET_TYPE"].fillna("")).str.strip(),
#         "situs_city": df["SITUS_CITY"],
#         "situs_zip": df["SITUS_ZIP"],
#         "sale_price": df["PACKAGE_SALE_PRICE"],
#         "sale_date": pd.to_datetime(df["PACKAGE_SALE_DATE"], unit="ms", errors="coerce"),
#         "deed_book": df["DEED_BOOK_PAGE_NUMBER"].str.split("-").str[0],
#         "deed_page": df["DEED_BOOK_PAGE_NUMBER"].str.split("-").str[1],
#         "land_value": df["TOTAL_LAND_VALUE_ASSESSED"],
#         "building_value": df["TOTAL_BUILDING_VALUE_ASSESSED"],
#         "total_value": df["TOTAL_PROPERTY_VALUE"],
#         "acreage": df["CALCULATED_ACREAGE"],
#         "land_use_class": df["LAND_CLASS"],
#         "neighborhood_code": df["NEIGHBORHOOD"],
#         "year_built": df["YEAR_BUILT"],
#         "heated_area": df["HEATED_AREA"],
#         "bedrooms": df["BEDROOMS"],
#         "bathrooms": df["BATHROOMS"],
#         "latitude": df["CentroidYCoordinat"],
#         "longitude": df["CentroidXCoordinate"],
#     })
#     return out[COMMON_COLUMNS]


# def wake_rollup_to_parcel(df):
#     """Wake's REALINFO is one row per building card; collapse to one row per REAL_ESTATE_ID (parcel)."""
#     first = df.sort_values("CARD_NUMBER").groupby("REAL_ESTATE_ID", as_index=False).first()
#     agg = df.groupby("REAL_ESTATE_ID").agg(
#         heated_area_total=("HEATED_AREA", "sum"),
#         year_built_earliest=("Year_Built", "min"),
#     ).reset_index()
#     return first.merge(agg, on="REAL_ESTATE_ID")

def nc_places_map(path,cfg):
    poi = gpd.read_parquet(path)
    df = pd.DataFrame({
        "lon": poi.geometry.x,
        "lat": poi.geometry.y,
        "name": poi["names"].str["primary"],
        "cat": poi["taxonomy"].str["primary"],
    })
    return df.to_csv(resolve_path(cfg,"processed_data")+"nc_places_map.csv", index=False)


# def wake_to_common(df):
#     parcels = wake_rollup_to_parcel(df)
#     out = pd.DataFrame({
#         "county": "Wake",
#         "parcel_id": parcels["REAL_ESTATE_ID"],
#         "owner_name": parcels["OWNER1"],
#         "situs_street_num": parcels["Street_Number"],
#         "situs_street_name": (parcels["Street_Name"].fillna("") + " " + parcels["Street_Type"].fillna("")).str.strip(),
#         "situs_city": parcels["PHYSICAL_CITY"],
#         "situs_zip": parcels["PHYSICAL_ZIP_CODE"],
#         "sale_price": parcels["Total_sale_Price"],
#         "sale_date": pd.to_datetime(parcels["Total_Sale_Date"], errors="coerce"),
#         "deed_book": parcels["DEED_BOOK"],
#         "deed_page": parcels["DEED_PAGE"],
#         "land_value": parcels["Assessed_Land_Value"],
#         "building_value": parcels["Assessed_Building_Value"],
#         "total_value": parcels["Assessed_Land_Value"] + parcels["Assessed_Building_Value"],
#         "acreage": parcels["Deeded_Acreage"],
#         "land_use_class": parcels["Land_classification"],
#         "neighborhood_code": parcels["VCS"],
#         "year_built": parcels["year_built_earliest"],
#         "heated_area": parcels["heated_area_total"],
#         "bedrooms": pd.NA,   # not captured (only room/bath codes, not clean counts)
#         "bathrooms": pd.NA,  # BATH is a rating code here, not a fixture count
#         "latitude": pd.NA,   # no coordinates in this extract
#         "longitude": pd.NA,
#     })
#     return out[COMMON_COLUMNS]


def merge_all_counties(raw_path):
    return pd.concat(
        [
            meck_to_common(load_mecklenburg(raw_path)),
            # bun_to_common(load_buncombe(raw_path), load_buncombe_building(raw_path)),
            # gui_to_common(load_guilford(raw_path)),
            # wake_to_common(load_wake(raw_path)),
        ],
        ignore_index=True,
    )


def normalize_land_use(series):
    """Map raw land_use_class values onto LAND_USE_MAP; unrecognized non-null values
    become "other", missing values stay missing."""
    mapped = series.map(LAND_USE_MAP)
    fallback = series.where(series.isna(), "other")
    return mapped.fillna(fallback)


def sale_price_outlier_mask(sale_price):
    """Flags sale_price values outside 1.5*IQR on a log1p scale (real-estate prices are
    right-skewed, so IQR on the raw scale would flag almost every high-end sale)."""
    valid = sale_price.notna() & (sale_price > 0)
    log_price = np.log1p(sale_price[valid])
    q1, q3 = log_price.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = pd.Series(False, index=sale_price.index)
    is_outlier.loc[valid] = (log_price < lower) | (log_price > upper)
    return is_outlier


def is_individual_owner(owner_name):
    upper = owner_name.fillna("").str.upper()
    is_entity = upper.str.contains(NON_INDIVIDUAL_OWNER_PATTERN, regex=True)
    is_blank = upper.str.strip() == ""
    return is_blank | ~is_entity


def clean_properties(df):
    df = df.drop_duplicates().copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    current_year = datetime.now().year
    bad_date = (df["sale_date"].dt.year < MIN_SALE_YEAR) | (df["sale_date"].dt.year > current_year)
    df.loc[bad_date, "sale_date"] = pd.NaT

    df.loc[df["sale_price"] == 0, "sale_price"] = pd.NA
    df.loc[sale_price_outlier_mask(df["sale_price"]), "sale_price"] = pd.NA

    # year_built == 0 is each source's placeholder for "unknown", not a real
    # construction year (unlike sale_date, genuine pre-1900 NC homes exist, so only
    # the placeholder value is nulled here, not an early cutoff).
    df.loc[(df["year_built"] == 0) | (df["year_built"] > current_year), "year_built"] = pd.NA

    recent_cutoff = pd.Timestamp(year=current_year - ARMS_LENGTH_LOOKBACK_YEARS, month=1, day=1)
    df["arms_length_sale"] = (
        df["sale_price"].between(ARMS_LENGTH_MIN_PRICE, ARMS_LENGTH_MAX_PRICE)
        & (df["sale_date"] >= recent_cutoff)
    )

    df["land_use_class"] = normalize_land_use(df["land_use_class"])
    df = df[df["land_use_class"].isin(RESIDENTIAL_LAND_USE_CLASSES)]

    bad_heated_area = df["heated_area"].notna() & ~df["heated_area"].between(MIN_HEATED_AREA, MAX_HEATED_AREA)
    df = df[~bad_heated_area]

    # Individual-owner filter needs owner_name, so runs before it (and situs_zip, unused
    # downstream) are dropped.
    df = df[is_individual_owner(df["owner_name"])].copy()
    df = df.drop(columns=["owner_name",
                          # "situs_zip"
                          ])

    return df


def nearest_label(train_df, label_col, query_df, k=NEAREST_CITY_NEIGHBORS):
    """Majority `label_col` among each query row's k nearest train rows (haversine on lat/lon)."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="haversine", algorithm="ball_tree")
    knn.fit(np.radians(train_df[["latitude", "longitude"]]), train_df[label_col])
    return knn.predict(np.radians(query_df[["latitude", "longitude"]]))


def median_split(nbhds, max_size):
    """Recursively halve neighborhoods (index: neighborhood_code; columns: latitude,
    longitude, n) by house count along their wider axis, until each part has at most
    max_size houses. Returns a list of parts."""
    if nbhds["n"].sum() <= max_size or len(nbhds) == 1:
        return [nbhds]
    # a degree of longitude is only ~cos(latitude) as wide as a degree of latitude
    lat_span = nbhds["latitude"].max() - nbhds["latitude"].min()
    lon_span = (nbhds["longitude"].max() - nbhds["longitude"].min()) * np.cos(np.radians(nbhds["latitude"].mean()))
    nbhds = nbhds.sort_values("latitude" if lat_span >= lon_span else "longitude")
    first_half = nbhds["n"].cumsum().shift(fill_value=0) < nbhds["n"].sum() / 2
    return median_split(nbhds[first_half], max_size) + median_split(nbhds[~first_half], max_size)


def assign_area(df):
    """Add `area`: the parcel's town, with MERGE_INTO_NEAREST_CITY parcels moved to the
    nearest town and Charlotte split into CHARLOTTE_1..N (numbered north to south)."""
    df = df.copy()
    df["area"] = df["situs_city"]
    merge = df["area"].isin(MERGE_INTO_NEAREST_CITY)
    df.loc[merge, "area"] = nearest_label(df[~merge], "area", df[merge])

    in_city = df["area"] == SPLIT_CITY
    has_nbhd = in_city & df["neighborhood_code"].notna()
    nbhds = df[has_nbhd].groupby("neighborhood_code").agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean"), n=("parcel_id", "size"),
    )
    parts = sorted(median_split(nbhds, MAX_AREA_SIZE), key=lambda part: -part["latitude"].mean())
    nbhd_area = {code: f"{SPLIT_CITY}_{i}" for i, part in enumerate(parts, 1) for code in part.index}
    df.loc[has_nbhd, "area"] = df.loc[has_nbhd, "neighborhood_code"].map(nbhd_area)

    # the few Charlotte parcels with no neighborhood_code take their nearest neighbors' area
    no_nbhd = in_city & ~has_nbhd
    if no_nbhd.any():
        df.loc[no_nbhd, "area"] = nearest_label(df[has_nbhd], "area", df[no_nbhd])
    return df


def run():
    cfg = load_config()
    raw_path = resolve_path(cfg, "raw_data")
    processed_path = resolve_path(cfg, "processed_data")
    nc_places_map(raw_path+ "NC parquet/nc_places.parquet",cfg)
    combined = merge_all_counties(raw_path)
    clean = assign_area(clean_properties(combined))
    save_dataframe(clean, processed_path, "properties_clean.csv")
    return clean


if __name__ == "__main__":
    run()
