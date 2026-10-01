"""Add modeling features on top of the cleaned properties table.

Reads data/processed/properties_clean.csv and adds property_age, price_per_sqft,
per-neighborhood price aggregates, and accessibility features in four groups:

- education: miles to the nearest school
- health care: share of places within COUNT_RADIUS_MILES that are health care
- emergency: miles to the nearest fire station, police station, and emergency
  department (see label_emergency_departments)
- activity: share of places within COUNT_RADIUS_MILES that are activities
  (parks, theaters, gyms, ...)

plus places_count, the number of places of any kind within COUNT_RADIUS_MILES.

It also adds location context from Mecklenburg GIS and NC DPI (see add_location_context):
the performance score of the house's assigned elementary, middle and high school,
violent and property crime rates of its neighborhood profile area, whether it is in
the FEMA floodplain, and miles to the nearest LYNX Blue Line station.
Health care and activity are shares of places_count rather than raw counts: raw
counts both mostly measured how built-up the area is (correlation 0.91), so
places_count carries that on its own and the shares say what kind of places are
nearby (correlation -0.27).

Distances are raw miles rather than decay scores: the tree models split on
thresholds, so any monotonic transform of distance gives them the same splits.

price_momentum is the yearly growth rate of price_per_sqft in the house's
neighborhood over MOMENTUM_START to MOMENTUM_END (the three years before the
January 2023 assessment the target comes from), falling back to the house's area
where the neighborhood has too few sales - a measure of where buyers expected
prices to go, as far as recorded sales show it.

The neighborhood aggregates and price_momentum here are computed on every row;
training.py recomputes both from its training rows only.
"""

import re
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

CURRENT_YEAR = datetime.now().year
EARTH_RADIUS = 3959 #miles

# price_momentum window: the three years before the January 1, 2023 revaluation that
# total_value comes from, so the feature carries nothing from after the target date.
MOMENTUM_START = pd.Timestamp("2020-01-01")
MOMENTUM_END = pd.Timestamp("2023-01-01")
# Below this many sales in the window a neighborhood's trend is too noisy, and the
# house's area trend is used instead (446 of ~2,900 neighborhoods clear it, ~46% of houses).
MIN_MOMENTUM_SALES = 30

# Health care and activity are counted within this radius; 5 miles still separates
# dense and sparse parts of the county (e.g. ~620 activity places around Uptown vs ~70
# around Mint Hill), where a 20-mile circle covers most of the county for every house.
COUNT_RADIUS_MILES = 5
# Nearest-distance features are capped here, so a house with nothing of a kind nearby
# gets the cap rather than an arbitrarily large distance.
MAX_DISTANCE_MILES = 20

# Overture's emergency_department category is mostly individual ER doctors, and real
# emergency departments are filed under many categories (obstetrics_and_gynecology,
# travel_service, outpatient_care_facility, none at all), so they're found by name
# instead: anything named like an emergency department/room, plus acute-care hospitals
# with an ED whose Overture name doesn't say so.
ED_NAME_PATTERN = r"(?i:emergency department|emergency room)|\b(?:ER|ED)$"
ED_NAME_EXCLUDED_CATS = ["veterinarian", "emergency_pet_hospital", "dental_clinic"]
ED_HOSPITAL_NAMES = [
    "Atrium Health University City", "Atrium Health Pineville", "Carolinas Medical Center Mercy",
    "Novant Health Presbyterian Medical Center", "Novant Health Matthews Medical Center",
    "Novant Health Ballantyne Medical Center",
    # Overture has no point named for these two hospitals, so a building on each campus stands in
    "Novant Huntersville L&D Unit",  # Novant Health Huntersville Medical Center
    "Novant Health Heart & Vascular Institute - Mint Hill",  # Novant Health Mint Hill Medical Center
    "Atrium Health Union", "Atrium Health Union West", "Atrium Health Cabarrus",
    "Atrium Health Kannapolis", "Lake Norman Regional Medical Center",
]

EDUCATION_CATS = ["school", "elementary_school", "middle_school", "high_school", "private_school"]
HEALTH_CARE_CATS = [
    "health_care", "doctors_office", "family_practice", "internal_medicine", "pediatric_clinic",
    "public_health_clinic", "urgent_care_clinic", "outpatient_care_facility", "dental_clinic",
    "general_dentistry", "pharmacy",
    "hospital",  # emergency departments are relabeled first, so these are hospitals without one
]
EMERGENCY_CATS = {
    "fire": ["fire_station"],
    "police": ["police_station"],
    # Ambulances aren't included: Medic moves them between posts during the day, and
    # Overture's ambulance_or_ems_service places are mostly clinicians and businesses.
    "medical": ["emergency_department_verified"],
}
ACTIVITY_CATS = [
    "park", "playground", "dog_park", "hiking_trail", "nature_reserve", "botanical_garden",
    "golf_course", "gym", "sport_or_fitness_facility", "skate_park", "trampoline_park",
    "ice_skating_rink", "bowling_alley", "movie_theater", "theatre_venue", "music_venue",
    "stadium_arena", "amusement_park", "arts_and_entertainment", "museum", "art_museum",
    "history_museum", "science_museum", "childrens_museum", "zoo", "aquarium", "library",
    "community_center", "shopping_mall",
]

# School grades, crime rates: the year the January 2023 assessment would have seen.
# NC DPI's `year` 2022 is the 2021-22 school year.
SCHOOL_GRADE_YEAR = 2022
CMS_AGENCY_PREFIX = "600"
# NC DPI category_code a school can have and still serve each CMS zone level:
# E elementary, M middle, H high, I K-8 / PreK-8, T 6-12 or similar combined grades.
SCHOOL_LEVELS = {
    "elementary": ("CMSElementarySchoolDistricts", "elem_name", ["E", "I"]),
    "middle": ("CMSMiddleSchoolDistricts", "midd_name", ["M", "I", "T"]),
    "high": ("CMSHighSchoolDistricts", "high_name", ["H", "T"]),
}
# Words dropped before matching zone names ("J.W. Grier") to report card names
# ("Joseph W Grier Academy"), along with single-letter initials.
SCHOOL_NAME_STOPWORDS = {
    "elementary", "elem", "middle", "high", "school", "academy", "the", "of", "and", "prek", "k",
}
CRIME_INDICATORS = {"Violent_Crimes": "violent_crime_rate", "Property_Crimes": "property_crime_rate"}


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


def label_emergency_departments(places):
    """Relabel real emergency departments (by name, see ED_NAME_PATTERN and
    ED_HOSPITAL_NAMES) as emergency_department_verified. Leftover emergency_department
    places, which are mostly individual doctors, stay out of every group."""
    places = places.dropna(subset=["lat", "lon"]).copy()
    named_ed = places["name"].str.contains(ED_NAME_PATTERN, na=False) & ~places["cat"].isin(ED_NAME_EXCLUDED_CATS)
    places.loc[named_ed | places["name"].isin(ED_HOSPITAL_NAMES), "cat"] = "emergency_department_verified"
    return places


def nearest_place_miles(house_df, places, cats):
    """Miles to the nearest place in `cats`, capped at MAX_DISTANCE_MILES. NaN for houses
    missing coordinates."""
    places = places[places["cat"].isin(cats)]
    houses = house_df[["latitude", "longitude"]].dropna()
    distance, _ = nearest_distance(houses["latitude"], houses["longitude"], places["lat"], places["lon"])
    return pd.Series(np.minimum(distance, MAX_DISTANCE_MILES), index=houses.index).reindex(house_df.index)


def places_within_miles(house_df, places, radius_miles, cats=None):
    """Number of places in `cats` (default: any category) within radius_miles of each
    house. NaN for houses missing coordinates."""
    places = places[places["cat"].isin(cats)] if cats else places.dropna(subset=["cat"])
    houses = house_df[["latitude", "longitude"]].dropna()
    tree = BallTree(np.radians(places[["lat", "lon"]]), metric="haversine")
    counts = tree.query_radius(np.radians(houses), r=radius_miles / EARTH_RADIUS, count_only=True)
    return pd.Series(counts, index=houses.index).reindex(house_df.index)


def add_accessibility(df, nc_places_df):
    places = label_emergency_departments(nc_places_df)
    df["education_nearest_miles"] = nearest_place_miles(df, places, EDUCATION_CATS)
    df["places_count"] = places_within_miles(df, places, COUNT_RADIUS_MILES)
    df["health_care_share"] = places_within_miles(df, places, COUNT_RADIUS_MILES, HEALTH_CARE_CATS) / df["places_count"]
    for name, cats in EMERGENCY_CATS.items():
        df[f"emergency_{name}_miles"] = nearest_place_miles(df, places, cats)
    df["activity_share"] = places_within_miles(df, places, COUNT_RADIUS_MILES, ACTIVITY_CATS) / df["places_count"]
    return df

def _name_tokens(name):
    words = re.sub(r"[^a-z ]", " ", name.lower()).split()
    return {w for w in words if len(w) > 1 and w not in SCHOOL_NAME_STOPWORDS}


def school_scores(zone_names, report_cards, categories):
    """School performance score for each CMS zone school name. A zone name matches a
    report card school of an allowed category whose name contains all of its words;
    if several do, the one with the fewest extra words ("Alexander" -> "J. M. Alexander
    Middle" over "Alexander Graham Middle"). Schools that opened after SCHOOL_GRADE_YEAR
    have no grade and get NaN."""
    candidates = report_cards[report_cards["category_code"].isin(categories)]
    cand_tokens = [(_name_tokens(n), score) for n, score in zip(candidates["name"], candidates["spg_score"])]
    scores = {}
    for zone_name in set(zone_names.dropna()):
        tokens = _name_tokens(zone_name)
        matches = [(len(t - tokens), score) for t, score in cand_tokens if tokens and tokens <= t]
        scores[zone_name] = min(matches)[1] if matches else np.nan
    return zone_names.map(scores)


def load_report_cards(raw_path):
    """CMS schools' names, category codes and overall performance score (spg_score,
    0-100) for SCHOOL_GRADE_YEAR."""
    rc_path = raw_path + "school_report_cards/"
    loc = pd.read_excel(rc_path + "rcd_location.xlsx")
    spg = pd.read_excel(rc_path + "rcd_acc_spg2.xlsx")
    loc = loc[(loc["year"] == SCHOOL_GRADE_YEAR) & loc["agency_code"].astype(str).str.startswith(CMS_AGENCY_PREFIX)]
    spg = spg[(spg["year"] == SCHOOL_GRADE_YEAR) & (spg["subgroup"] == "ALL")]
    return loc[["agency_code", "name", "category_code"]].merge(spg[["agency_code", "spg_score"]], on="agency_code")


def add_location_context(df, raw_path):
    """Assigned-school scores, neighborhood crime rates, floodplain and light rail
    distance, all by the house's coordinates. Houses without coordinates get NaN."""
    gis_path = raw_path + "meck_gis/"
    houses = df[["latitude", "longitude"]].dropna()
    points = gpd.GeoDataFrame(index=houses.index, geometry=gpd.points_from_xy(houses["longitude"], houses["latitude"]), crs=4326)

    def join(layer, columns):
        """Columns of the `layer` polygon each house falls in (first one if several overlap)."""
        polygons = gpd.read_file(gis_path + f"{layer}.geojson")[columns + ["geometry"]]
        joined = gpd.sjoin(points, polygons, how="left", predicate="within")
        return joined[~joined.index.duplicated()][columns].reindex(df.index)

    report_cards = load_report_cards(raw_path)
    for level, (layer, name_col, categories) in SCHOOL_LEVELS.items():
        # the name is for display (lookup_property), not a model feature
        df[f"{level}_school"] = join(layer, [name_col])[name_col]
        df[f"{level}_school_score"] = school_scores(df[f"{level}_school"], report_cards, categories)

    # Crime rates per 1,000 residents. CMPD covers Charlotte only, so NPAs in the towns
    # with their own police departments have no value.
    qol = pd.read_csv(gis_path + "quality_of_life.csv")
    rates = qol.pivot_table(index="npa", columns="raw_data_name", values="normalized").rename(columns=CRIME_INDICATORS)
    npa = join("NeighborhoodProfileAreas", ["npa"])
    df = df.join(npa.merge(rates, left_on="npa", right_index=True, how="left")[list(CRIME_INDICATORS.values())].set_axis(df.index))

    floodplain = join("FEMAFloodplain", ["objectid"])["objectid"]
    df["in_floodplain"] = floodplain.notna().astype(float).where(df["latitude"].notna())

    stations = gpd.read_file(gis_path + "CATSLynxBlueLineStations.geojson")
    distance, _ = nearest_distance(houses["latitude"], houses["longitude"], stations.geometry.y, stations.geometry.x)
    df["light_rail_miles"] = pd.Series(np.minimum(distance, MAX_DISTANCE_MILES), index=houses.index).reindex(df.index)
    return df


def add_neighborhood_aggregates(df, source=None):
    """Median price_per_sqft and sale count per neighborhood, computed from `source`
    rows (default: df itself) - training.py passes its training rows so test rows'
    sales never feed their own features."""
    source = df if source is None else source
    group_cols = ["county", "neighborhood_code"]
    agg = (
        source.loc[source["price_per_sqft"].notna()]
        .groupby(group_cols)["price_per_sqft"]
        .agg(neighborhood_median_price_per_sqft="median", neighborhood_sale_count="count")
        .reset_index()
    )
    return df.merge(agg, on=group_cols, how="left")


def _growth_by(sales, keys):
    """Yearly growth rate of price_per_sqft per group - exp of the least-squares slope of
    log(price_per_sqft) against time in years, minus 1 - and the number of sales."""
    sales = sales.assign(xy=sales["t"] * sales["log_ppsf"], xx=sales["t"] ** 2)
    groups = sales.groupby(keys)
    m = groups[["t", "log_ppsf", "xy", "xx"]].mean()
    slope = (m["xy"] - m["t"] * m["log_ppsf"]) / (m["xx"] - m["t"] ** 2)
    return pd.DataFrame({"growth": np.expm1(slope), "sales": groups.size()}).reset_index()


def add_price_momentum(df, source=None):
    """price_momentum: neighborhood growth rate where it has at least MIN_MOMENTUM_SALES
    sales in the window, else the area's. Computed from `source` rows (default: df
    itself), like add_neighborhood_aggregates.

    Each house only carries its most recent sale, so a neighborhood's sales in the
    window are different houses each year - the trend partly reflects which houses
    happened to sell, not only price change."""
    source = df if source is None else source
    sale_date = pd.to_datetime(source["sale_date"], format="mixed")
    in_window = source["price_per_sqft"].notna() & (sale_date >= MOMENTUM_START) & (sale_date < MOMENTUM_END)
    sales = pd.DataFrame({
        "county": source["county"], "neighborhood_code": source["neighborhood_code"], "area": source["area"],
        "t": (sale_date - MOMENTUM_START).dt.days / 365.25,
        "log_ppsf": np.log(source["price_per_sqft"]),
    })[in_window]

    nbhd = _growth_by(sales, ["county", "neighborhood_code"])
    area = _growth_by(sales, ["area"])
    out = df.merge(nbhd, on=["county", "neighborhood_code"], how="left").merge(
        area[["area", "growth"]].rename(columns={"growth": "area_growth"}), on="area", how="left",
    )
    out["price_momentum"] = out["growth"].where(out["sales"] >= MIN_MOMENTUM_SALES, out["area_growth"])
    return out.drop(columns=["growth", "sales", "area_growth"])


def build_features(df, nc_places_df, raw_path):
    df = add_property_age(df)
    df = add_price_per_sqft(df)
    df = add_neighborhood_aggregates(df)
    df = add_price_momentum(df)
    df = add_accessibility(df, nc_places_df)
    df = add_location_context(df, raw_path)
    return df


def run():
    cfg = load_config()
    processed_path = resolve_path(cfg, "processed_data")
    features_path = resolve_path(cfg, "features_data")

    clean = read_dataframe(processed_path, "properties_clean.csv", parse_dates=["sale_date"])
    clean["arms_length_sale"] = clean["arms_length_sale"].astype(bool)
    places = read_dataframe(processed_path, "nc_places_map.csv")

    features = build_features(clean, places, resolve_path(cfg, "raw_data"))
    save_dataframe(features, features_path, "properties_features.csv")
    return features


if __name__ == "__main__":
    run()
