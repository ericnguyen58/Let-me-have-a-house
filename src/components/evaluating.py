"""Model evaluation beyond training.py's test-set metrics.

validate_houses is a spot check: every trained model's estimate next to the actual
total_value for a handful of individual houses, so errors can be read house by house
instead of only as averages. Houses come from training.py's test split (rebuilt here
with the same config and random_state), so no model saw them during training, and
their sale-derived features are computed from training rows only, as in training.
The split only matches training's if properties_features.csv hasn't changed since
the models were trained.

drop_column_test retrains the production model without one feature group at a time
and reports how much test error that costs: a group's value on top of everything
else, with a bootstrap interval and seed noise so small changes can be told from noise.

same_house_by_area holds a house fixed and gives it the location of real houses in
each area (every non-house feature, taken together from one real parcel), so the
model's estimates show what the same house is worth in each area.
`area_premium_decomposition` splits each area's premium into feature groups.
"""

import itertools
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.components import predict
from src.components.training import (
    FEATURE_COLUMNS, FEATURE_SETS, MODEL_TYPES, NEIGHBORHOOD_PRICE, TARGET,
    build_preprocessor, load_training_data, split_with_aggregates,
)
from src.utils.common import resolve_path, save_dataframe
from src.utils.load_config import load_config

# Every feature in exactly one group. Shared with notebooks/shap_explain.ipynb.
FEATURE_GROUPS = {
    "house": ["land_use_class", "acreage", "year_built", "property_age", "heated_area", "bedrooms", "bathrooms"],
    "location": ["area", "latitude", "longitude", "neighborhood_sale_count", "price_momentum"],
    "neighborhood_price": [NEIGHBORHOOD_PRICE],
    "accessibility": ["education_nearest_miles", "places_count", "health_care_share", "activity_share",
                      "emergency_fire_miles", "emergency_police_miles", "emergency_medical_miles"],
    "school": ["elementary_school_score", "middle_school_score", "high_school_score"],
    "crime": ["violent_crime_rate", "property_crime_rate"],
    "flood": ["in_floodplain"],
    "light_rail": ["light_rail_miles"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURE_COLUMNS)
HOUSE_FEATURES = FEATURE_GROUPS["house"]

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


# --- drop-column test ---------------------------------------------------------------

# Single features dropped on their own, besides the groups.
DROP_SINGLE_FEATURES = ["area", "places_count", "health_care_share"]
NOISE_SEEDS = [1, 2, 3]  # extra seeds the full model is refit with, to measure seed noise
BOOTSTRAP_SAMPLES = 1000


def production_model(feature_set, random_state):
    """The production gradient boosting setup: best settings from training's search, same
    early stopping as training.build_search_estimators."""
    path = resolve_path(load_config(), "reports") + "best_hyperparameters.json"
    with open(path) as f:
        params = json.load(f)[f"{feature_set}__hist_gradient_boosting"]
    return HistGradientBoostingRegressor(
        **params, max_iter=3000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        random_state=random_state,
    )


def fit_and_predict(train, test, features, feature_set, random_state):
    """Test-set predictions in dollars of a production model fit on `features`."""
    pre = build_preprocessor(features)
    model = production_model(feature_set, random_state).fit(
        pre.fit_transform(train[features]), np.log1p(train[TARGET]))
    return np.expm1(model.predict(pre.transform(test[features])))


def paired_bootstrap(actual, base_pred, pred, n=BOOTSTRAP_SAMPLES, seed=0):
    """95% intervals of the change in MAE and in mean percent error (pred minus base_pred),
    resampling test houses."""
    base_abs, abs_ = np.abs(base_pred - actual), np.abs(pred - actual)
    d_mae, d_pct = abs_ - base_abs, (abs_ - base_abs) / actual
    rng = np.random.default_rng(seed)
    samples = np.array([(d_mae[i].mean(), d_pct[i].mean())
                        for i in (rng.integers(0, len(actual), len(actual)) for _ in range(n))])
    return np.percentile(samples[:, 0], [2.5, 97.5]), np.percentile(samples[:, 1], [2.5, 97.5])


def drop_column_test(feature_set, train=None, test=None):
    """Test error of the production model for `feature_set` with each group in
    FEATURE_GROUPS (and each of DROP_SINGLE_FEATURES) left out, against the full model.

    `real` is True when the bootstrap interval of the change in mean percent error
    excludes 0 and the change is larger than the largest gap between the full model's
    seeds - otherwise the change can't be told from noise."""
    cfg = load_config()
    if train is None:
        train, test = load_splits()
    seed = cfg["training"]["random_state"]
    features = FEATURE_SETS[feature_set]
    actual = test[TARGET].to_numpy()

    def errors(pred):
        return np.mean(np.abs(pred - actual)), np.mean(np.abs(pred - actual) / actual)

    print(f"{feature_set}: full model, seeds {[seed, *NOISE_SEEDS]}", flush=True)
    base_pred = fit_and_predict(train, test, features, feature_set, seed)
    base_mae, base_pct = errors(base_pred)
    seed_pcts = [base_pct] + [errors(fit_and_predict(train, test, features, feature_set, s))[1] for s in NOISE_SEEDS]
    seed_noise = max(seed_pcts) - min(seed_pcts)

    dropped = {g: cols for g, cols in FEATURE_GROUPS.items() if g != "house"}
    dropped.update({f: [f] for f in DROP_SINGLE_FEATURES})
    rows = []
    for name, cols in dropped.items():
        kept = [f for f in features if f not in cols]
        if len(kept) == len(features):
            continue  # nothing of this group is in the feature set
        print(f"{feature_set}: without {name}", flush=True)
        pred = fit_and_predict(train, test, kept, feature_set, seed)
        mae, pct = errors(pred)
        mae_ci, pct_ci = paired_bootstrap(actual, base_pred, pred)
        rows.append({
            "feature_set": feature_set, "dropped": name, "kind": "group" if name in FEATURE_GROUPS else "feature",
            "mae": mae, "pct_error": pct, "mae_change": mae - base_mae, "pct_error_change": pct - base_pct,
            "mae_change_low": mae_ci[0], "mae_change_high": mae_ci[1],
            "pct_error_change_low": pct_ci[0], "pct_error_change_high": pct_ci[1],
            "seed_noise": seed_noise,
            "real": bool((pct_ci[0] > 0 or pct_ci[1] < 0) and abs(pct - base_pct) > seed_noise),
        })
    full = {"feature_set": feature_set, "dropped": "(none)", "kind": "full", "mae": base_mae,
            "pct_error": base_pct, "seed_noise": seed_noise}
    return pd.DataFrame([full, *rows])


def run_drop_column_test():
    train, test = load_splits()
    results = pd.concat([drop_column_test(fs, train, test) for fs in FEATURE_SETS], ignore_index=True)
    print(results[["feature_set", "dropped", "mae", "pct_error", "pct_error_change", "real"]].to_string(index=False))
    save_dataframe(results, resolve_path(load_config(), "reports"), "drop_column.csv")
    return results


# --- the same house in different areas --------------------------------------------------

DONORS_PER_AREA = 300  # real parcels per area whose location a reference house is given
COUNTY_DONORS = 3000  # parcels county-wide, the baseline each area's premium is against
SIZE_BAND = 0.25  # coverage: a neighborhood has a house within this share of the reference size
RAW_SIZE_BAND = 0.20  # raw-data check: houses within this share of the reference size
DECOMPOSITION_PAIRS = 200  # (area donor, county donor) pairs per area in the decomposition
DECOMPOSITION_GROUPS = ["location", "accessibility", "school", "crime", "flood", "light_rail"]
DECOMPOSITION_FEATURE_SET = "without_neighborhood_price"


def all_houses():
    """Every house, training and test, with neighborhood features from training sales."""
    train, test = load_splits()
    return pd.concat([train, test], ignore_index=True)


def typical_house(houses, land_use_class):
    """The real house of `land_use_class` closest to the class's median house: median
    heated area, bedrooms, bathrooms, year built and acreage, in standardized units."""
    rows = houses[houses["land_use_class"] == land_use_class]
    cols = ["heated_area", "bedrooms", "bathrooms", "year_built", "acreage"]
    z = (rows[cols] - rows[cols].median()) / rows[cols].std()
    return rows.loc[z.pow(2).sum(axis=1).idxmin()]


def reference_houses(houses):
    """{name: house row}: the config test addresses and the typical house of each class."""
    from src.components.search import lookup_property  # search imports training
    refs = {}
    for address in load_config()["test_addresses"]:
        parcel = lookup_property(address)[0]["parcel_id"]
        refs[address.split(",")[0]] = houses[houses["parcel_id"] == parcel].iloc[0]
    for land_use in ["single_family", "townhouse", "condo"]:
        refs[f"typical {land_use}"] = typical_house(houses, land_use)
    return refs


def donors(houses, land_use_class, seed=0):
    """(per-area donors, county-wide donors) of `land_use_class`."""
    same = houses[houses["land_use_class"] == land_use_class]
    per_area = pd.concat([g.sample(min(len(g), DONORS_PER_AREA), random_state=seed)
                          for _, g in same.groupby("area")])
    county = same.sample(min(len(same), COUNTY_DONORS), random_state=seed)
    return per_area, county


def with_house(locations, house):
    """`locations` (donor rows) with every house feature set to `house`'s."""
    out = locations.copy()
    for f in HOUSE_FEATURES:
        out[f] = house[f]
    return out


def log_estimates(rows, feature_set):
    pre, model = predict.load_model(feature_set, "hist_gradient_boosting")
    return model.predict(pre.transform(rows[FEATURE_SETS[feature_set]]))


def size_coverage(houses, donor_rows, house):
    """Per donor: whether its neighborhood has a house of the same class within SIZE_BAND
    of `house`'s heated area - if not, the estimate is outside what the data has seen."""
    size = house["heated_area"]
    same = houses[houses["land_use_class"] == house["land_use_class"]]
    near = same["heated_area"].between(size * (1 - SIZE_BAND), size * (1 + SIZE_BAND))
    covered = set(same.loc[near, "neighborhood_code"])
    return donor_rows["neighborhood_code"].isin(covered).to_numpy()


def raw_area_prices(houses, house):
    """Per area, median total_value / heated_area of houses of the same class within
    RAW_SIZE_BAND of `house`'s size, times its size: an estimate with no model."""
    size = house["heated_area"]
    same = houses[(houses["land_use_class"] == house["land_use_class"])
                  & houses["heated_area"].between(size * (1 - RAW_SIZE_BAND), size * (1 + RAW_SIZE_BAND))]
    per_sqft = same[TARGET] / same["heated_area"]
    return pd.DataFrame({"raw_estimate": per_sqft.groupby(same["area"]).median() * size,
                         "raw_houses": per_sqft.groupby(same["area"]).size()})


def same_house_by_area(houses=None, refs=None):
    """For each reference house x area x feature set: the estimate of the house with each
    area donor's location (median and 10th-90th percentile), its premium over the same
    house with county-wide locations (exp of the mean log difference, minus 1), the share
    of donors whose neighborhood has houses of its size, and the raw-data check."""
    houses = all_houses() if houses is None else houses
    refs = reference_houses(houses) if refs is None else refs
    rows = []
    for name, house in refs.items():
        per_area, county = donors(houses, house["land_use_class"])
        area_rows, county_rows = with_house(per_area, house), with_house(county, house)
        covered = size_coverage(houses, per_area, house)
        raw = raw_area_prices(houses, house)
        raw_county = (houses.loc[houses["land_use_class"] == house["land_use_class"]]
                      .pipe(lambda h: h[h["heated_area"].between(house["heated_area"] * (1 - RAW_SIZE_BAND),
                                                                  house["heated_area"] * (1 + RAW_SIZE_BAND))])
                      .pipe(lambda h: (h[TARGET] / h["heated_area"]).median() * house["heated_area"]))
        for feature_set in FEATURE_SETS:
            log_area, log_county = log_estimates(area_rows, feature_set), log_estimates(county_rows, feature_set)
            frame = pd.DataFrame({"area": per_area["area"].to_numpy(), "log": log_area, "covered": covered})
            for area, g in frame.groupby("area"):
                dollars = np.expm1(g["log"])
                rows.append({
                    "house": name, "land_use_class": house["land_use_class"], "heated_area": house["heated_area"],
                    "assessed": house[TARGET], "feature_set": feature_set, "area": area, "donors": len(g),
                    "estimate": dollars.median(), "estimate_p10": dollars.quantile(0.1),
                    "estimate_p90": dollars.quantile(0.9),
                    "premium": math.expm1(g["log"].mean() - log_county.mean()),
                    "size_coverage": g["covered"].mean(),
                    "raw_estimate": raw["raw_estimate"].get(area, np.nan),
                    "raw_houses": raw["raw_houses"].get(area, 0),
                    "raw_premium": raw["raw_estimate"].get(area, np.nan) / raw_county - 1,
                })
    return pd.DataFrame(rows)


def shapley_weights(n):
    """Weight of a coalition of size s that a group joins, for n groups."""
    return [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]


def area_premium_decomposition(house, houses, feature_set=DECOMPOSITION_FEATURE_SET, seed=0):
    """Each area's premium for `house` split into DECOMPOSITION_GROUPS (Shapley values).

    Pairs an area donor a with a county-wide donor b; a coalition S of groups takes S from
    a and every other location feature from b. A group's share is its average marginal
    effect on the log estimate over all orderings, averaged over pairs. The shares add up
    to the mean log difference between the house in the area and county-wide, so
    exp(total) - 1 is the area's premium. Mixing groups from two parcels makes
    combinations no real house has; read the shares as the model's attribution."""
    groups = [g for g in DECOMPOSITION_GROUPS if any(f in FEATURE_SETS[feature_set] for f in FEATURE_GROUPS[g])]
    n, weights = len(groups), shapley_weights(len(groups))
    per_area, county = donors(houses, house["land_use_class"], seed)
    rng = np.random.default_rng(seed)
    coalitions = [frozenset(c) for k in range(n + 1) for c in itertools.combinations(groups, k)]
    rows = []
    for area, a in per_area.groupby("area"):
        a = a.sample(DECOMPOSITION_PAIRS, replace=len(a) < DECOMPOSITION_PAIRS, random_state=seed)
        b = county.iloc[rng.integers(0, len(county), DECOMPOSITION_PAIRS)]
        a, b = with_house(a, house).reset_index(drop=True), with_house(b, house).reset_index(drop=True)
        value = {}
        for c in coalitions:  # mean log estimate when the groups in c come from the area donor
            mixed = b.copy()
            for g in c:
                mixed[FEATURE_GROUPS[g]] = a[FEATURE_GROUPS[g]]
            value[c] = log_estimates(mixed, feature_set).mean()
        share = {g: sum(weights[len(c)] * (value[c | {g}] - value[c]) for c in coalitions if g not in c)
                 for g in groups}
        total = value[frozenset(groups)] - value[frozenset()]
        rows.append({"area": area, "feature_set": feature_set, "log_premium": total,
                     "premium": math.expm1(total), **{f"{g}_log": v for g, v in share.items()}})
    return pd.DataFrame(rows)


def run_same_house_by_area():
    houses = all_houses()
    refs = reference_houses(houses)
    reports = resolve_path(load_config(), "reports")
    by_area = same_house_by_area(houses, refs)
    save_dataframe(by_area, reports, "same_house_by_area.csv")
    decomposition = pd.concat(
        [area_premium_decomposition(refs[name], houses).assign(house=name)
         for name in refs if name.startswith("typical")], ignore_index=True)
    save_dataframe(decomposition, reports, "area_premium_decomposition.csv")
    return by_area, decomposition


def run():
    results = validate_houses()
    print_validation(results)
    save_dataframe(results, resolve_path(load_config(), "reports"), "validation_houses.csv")
    return results


COMMANDS = {"validate": run, "drop_column": run_drop_column_test, "same_house_by_area": run_same_house_by_area}

if __name__ == "__main__":
    COMMANDS[sys.argv[1] if len(sys.argv) > 1 else "validate"]()
