"""Estimate today's market price from the 2023 assessed value with a sales ratio, and
backtest it.

Nothing else in the project imports this module, so it can be removed without touching
the rest of the pipeline (its example run uses search.lookup_property, and
chain_backtest refits the model through training.py and finetune.py).

Sales ratio method: for recent arms-length sales, ratio = sale_price / total_value (the
2023 assessed value). A house's market estimate is its assessed value times the median
ratio of recent sales in its own neighborhood - never another neighborhood's or the
area's, since neighborhoods inside one area can move very differently (a golf-course
neighborhood vs. the starter homes next to it). The window is the last 12 months,
widened to 24 and then 36 months while the neighborhood has fewer than
MIN_NEIGHBORHOOD_SALES sales; a neighborhood without that many sales in 36 months gets
no estimate. Neither does a house built in or after ASSESSMENT_YEAR: its 2023 assessed
value isn't a 2023 value of the finished house the way it is for older houses, and in
the backtest its median error was 14% against 6.5%. This is the ratio study assessors use to check their own values, turned
around to move a 2023 value to today's prices.

backtest() checks the method on sales it didn't use: ratios come from sales before a
cutoff date, and are scored on sales from the cutoff onward, against no adjustment and
against Zillow's county index (predict.market_adjustment's source), all scored on the
same sales.

The range is calibrated on how far real sales landed from this method's estimates:
range_factors() re-runs the method as of RANGE_CALIBRATION_MONTHS ago and takes the
10th and 90th percentiles of sale price / estimate over the sales since, so about 80%
of sales should fall inside it. (The spread of the neighborhood's own ratios is too
narrow for this: in the backtest only a third of sales fell inside its middle half.)

chain_backtest() checks the same adjustment applied to the model's estimate of the
assessed value instead of the county's, for houses without a usable assessment: model ->
sales ratio, with Zillow's county index where the neighborhood has no ratio. The model
is refit for it (model_estimates) so it never saw the scored houses or any sale from the
cutoff on. Two ratios are compared: sale / assessed value (the one above) and sale /
model estimate, which also absorbs the model's own bias.

Houses that sell are not a random sample (renovated houses and flips are
overrepresented), so ratios lean a little high; medians and the price/sqft floor limit
that.
"""

import functools
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.common import read_dataframe, resolve_path, save_dataframe
from src.utils.load_config import load_config

RATIO_WINDOWS_MONTHS = [12, 24, 36]
# Backtested: 3 covers ~96% of sales with a 7.0% median error; requiring 10 covers 77%
# with a slightly worse one, since the extra sales come from older, lower-priced years.
MIN_NEIGHBORHOOD_SALES = 3
# Same floor as search.MIN_PRICE_PER_SQFT: below it a sale is most likely a family
# transfer or distressed sale, not a market price.
MIN_PRICE_PER_SQFT = 50
ZILLOW_BASE_MONTH = "2022-12-31"  # last index month before the January 2023 assessment
ASSESSMENT_YEAR = 2023  # total_value is the January 1, 2023 revaluation
RANGE_CALIBRATION_MONTHS = 6
RANGE_QUANTILES = (0.10, 0.90)  # an 80% range
# How far back before a cutoff ratios and range calibration reach: the range is calibrated
# on the RANGE_CALIBRATION_MONTHS before it, with ratios from up to 36 months before that.
RATIO_HISTORY_MONTHS = RANGE_CALIBRATION_MONTHS + RATIO_WINDOWS_MONTHS[-1]
# Folds for the held-out houses' sale-derived features, so none uses its own sale.
FEATURE_FOLDS = 5

# market_value's choices, all from the backtests (multi_cutoff_backtest and the fallback
# comparison in the README):
# - a house built in or after ASSESSMENT_YEAR keeps its county value unless that is below
#   this share of the model's estimate - then it is most likely a partial assessment of an
#   unfinished house (2026 sales: 6.4% median error, against 12.3% always using the county
#   value and 8.6% always using the model)
PARTIAL_ASSESSMENT_SHARE = 0.85
# - a neighborhood without MIN_NEIGHBORHOOD_SALES in 36 months uses its sales from the
#   last FALLBACK_MONTHS, each moved to today by the Zillow county index, from as few as
#   FALLBACK_MIN_SALES; then the area's median ratio over AREA_RATIO_MONTHS (at least
#   AREA_MIN_SALES sales); Zillow's county factor only when neither exists (no-ratio sales:
#   6.4% median error, against 10.0% with Zillow alone)
FALLBACK_MONTHS = 60
FALLBACK_MIN_SALES = 1
AREA_RATIO_MONTHS = 12
AREA_MIN_SALES = 30
RATIO_METHODS = ["neighborhood", "neighborhood_adjusted", "area", "zillow_county"]
# Out-of-fold model estimates (build_out_of_fold_estimates), in the models directory
OOF_FOLDS = 5
OOF_FILE = "out_of_fold_estimates.csv"


def load_sales(features_path):
    """Arms-length sales with a usable price and assessed value, dated no later than
    today (the county file has a few sales dated in the future)."""
    df = read_dataframe(features_path, "properties_features.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"], format="mixed")
    sales = df[
        df["arms_length_sale"].astype(bool)
        & df["sale_price"].gt(0)
        & df["total_value"].gt(0)
        & df["heated_area"].gt(0)
        & (df["sale_price"] / df["heated_area"] >= MIN_PRICE_PER_SQFT)
        & (df["sale_date"] <= pd.Timestamp(datetime.now()))
    ].copy()
    sales["ratio"] = sales["sale_price"] / sales["total_value"]
    return sales


def neighborhood_ratios(sales, as_of):
    """Per neighborhood: median sale/assessed ratio, the number of sales behind it and the
    window in months, using the shortest window in RATIO_WINDOWS_MONTHS before `as_of`
    with at least MIN_NEIGHBORHOOD_SALES sales. Neighborhoods that never reach it are
    left out."""
    as_of = pd.Timestamp(as_of)
    found = []
    for months in RATIO_WINDOWS_MONTHS:
        window = sales[(sales["sale_date"] < as_of) & (sales["sale_date"] >= as_of - pd.DateOffset(months=months))]
        stats = window.groupby("neighborhood_code")["ratio"].agg(["median", "count"]).assign(months=months)
        found.append(stats[stats["count"] >= MIN_NEIGHBORHOOD_SALES])
    ratios = pd.concat(found)
    return ratios[~ratios.index.duplicated()]  # keep each neighborhood's shortest window


def range_factors(sales, as_of, base="total_value"):
    """(low, high) multipliers for an estimate's range: RANGE_QUANTILES of sale price /
    estimate for sales in the RANGE_CALIBRATION_MONTHS before `as_of`, estimated with
    ratios as of the start of that period. `base` is the value the ratio multiplies, and
    sales["ratio"] must be sale_price / that value."""
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.DateOffset(months=RANGE_CALIBRATION_MONTHS)
    ratios = neighborhood_ratios(sales, start)
    later = sales[(sales["sale_date"] >= start) & (sales["sale_date"] < as_of)
                  & sales["neighborhood_code"].isin(ratios.index)
                  & (sales["year_built"] < ASSESSMENT_YEAR)]  # houses that get an estimate
    misses = later["sale_price"] / (later[base] * later["neighborhood_code"].map(ratios["median"]))
    return tuple(misses.quantile(list(RANGE_QUANTILES)))


def market_estimate(assessed_value, neighborhood_code, year_built=None, as_of=None):
    """Today's market price estimate for a house with a 2023 assessed value (the county's
    total_value, or predict.predict_total_value's estimate of it), with an 80% range.
    market_estimate is None when the house was built in or after ASSESSMENT_YEAR, or the
    neighborhood has too few recent sales."""
    if year_built is not None and year_built >= ASSESSMENT_YEAR:
        return {"market_estimate": None, "reason": f"built in {int(year_built)}, after the January "
                f"{ASSESSMENT_YEAR} assessment the method scales from"}
    as_of = as_of or datetime.now()
    sales = load_sales(resolve_path(load_config(), "features_data"))
    ratios = neighborhood_ratios(sales, as_of)
    if neighborhood_code not in ratios.index:
        return {"market_estimate": None, "reason": f"fewer than {MIN_NEIGHBORHOOD_SALES} sales in "
                f"{neighborhood_code} in the last {RATIO_WINDOWS_MONTHS[-1]} months"}
    row = ratios.loc[neighborhood_code]
    estimate = assessed_value * row["median"]
    low, high = range_factors(sales, as_of)
    return {
        "market_estimate": round(float(estimate), 2),
        "range_low": round(float(estimate * low), 2), "range_high": round(float(estimate * high), 2),
        "ratio": round(float(row["median"]), 4),
        "neighborhood": neighborhood_code, "sales_used": int(row["count"]), "months": int(row["months"]),
    }


def zillow_factor(as_of):
    """Zillow Mecklenburg index in the last month before `as_of`, over ZILLOW_BASE_MONTH."""
    zhvi = pd.read_csv(resolve_path(load_config(), "raw_data") + "zillow/county_zhvi.csv")
    row = zhvi[(zhvi["RegionName"] == "Mecklenburg County") & (zhvi["State"] == "NC")].iloc[0]
    months = [c for c in zhvi.columns if c[:4].isdigit() and pd.Timestamp(c) < pd.Timestamp(as_of)]
    return row[months[-1]] / row[ZILLOW_BASE_MONTH]


@functools.lru_cache(maxsize=1)
def _zillow_index():
    """Zillow's Mecklenburg County index by month end."""
    zhvi = pd.read_csv(resolve_path(load_config(), "raw_data") + "zillow/county_zhvi.csv")
    row = zhvi[(zhvi["RegionName"] == "Mecklenburg County") & (zhvi["State"] == "NC")].iloc[0]
    return pd.Series({pd.Timestamp(c): row[c] for c in zhvi.columns if c[:4].isdigit()}).sort_index()


def _index_before(dates):
    """The Zillow index in the last month ending before each of `dates`."""
    index = _zillow_index()
    pos = index.index.searchsorted(dates, side="left") - 1
    return pd.Series(index.values[np.clip(pos, 0, None)], index=dates.index)


def ratio_tables(sales, as_of):
    """Every ratio a house can get as of `as_of`, from sales before it, keyed by
    RATIO_METHODS: per neighborhood (neighborhood_ratios); per neighborhood over the last
    FALLBACK_MONTHS with each sale's ratio moved to `as_of` by the Zillow county index; per
    area; and Zillow's county factor. sales["ratio"] must be sale_price over the value the
    ratio will multiply."""
    as_of = pd.Timestamp(as_of)
    past = sales[sales["sale_date"] < as_of]
    recent = past[past["sale_date"] >= as_of - pd.DateOffset(months=FALLBACK_MONTHS)]
    moved = recent["ratio"] * _index_before(pd.Series([as_of])).iloc[0] / _index_before(recent["sale_date"])
    adjusted = moved.groupby(recent["neighborhood_code"]).agg(["median", "count"])
    area_sales = past[past["sale_date"] >= as_of - pd.DateOffset(months=AREA_RATIO_MONTHS)]
    area = area_sales.groupby("area")["ratio"].agg(["median", "count"])
    return {
        "neighborhood": neighborhood_ratios(sales, as_of),
        "neighborhood_adjusted": adjusted[adjusted["count"] >= FALLBACK_MIN_SALES].assign(months=FALLBACK_MONTHS),
        "area": area[area["count"] >= AREA_MIN_SALES].assign(months=AREA_RATIO_MONTHS),
        "zillow_county": zillow_factor(as_of),
    }


def apply_ratios(tables, rows):
    """(ratio, method, sales behind it, months) for each of `rows` (neighborhood_code and
    area columns): the first of RATIO_METHODS that has a ratio for the row."""
    ratio = pd.Series(np.nan, index=rows.index)
    method = pd.Series(RATIO_METHODS[-1], index=rows.index)
    count = pd.Series(np.nan, index=rows.index)
    months = pd.Series(np.nan, index=rows.index)
    for name in RATIO_METHODS[:-1]:
        key = rows["area" if name == "area" else "neighborhood_code"]
        found = ratio.isna() & key.isin(tables[name].index)
        ratio[found] = key[found].map(tables[name]["median"])
        count[found] = key[found].map(tables[name]["count"])
        months[found] = key[found].map(tables[name]["months"])
        method[found] = name
    ratio = ratio.fillna(tables["zillow_county"])
    return ratio, method, count, months


def range_factors_by_method(sales, as_of, base):
    """(low, high) multipliers per ratio method, like range_factors: RANGE_QUANTILES of
    sale price / estimate for houses built before ASSESSMENT_YEAR that sold in the
    RANGE_CALIBRATION_MONTHS before `as_of`, estimated with ratio_tables as of the start
    of that period. The fallback methods have few such sales, so they share one range."""
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.DateOffset(months=RANGE_CALIBRATION_MONTHS)
    later = sales[(sales["sale_date"] >= start) & (sales["sale_date"] < as_of)
                  & (sales["year_built"] < ASSESSMENT_YEAR)]
    ratio, method, _, _ = apply_ratios(ratio_tables(sales, start), later)
    misses = later["sale_price"] / (later[base] * ratio)
    is_main = method == RATIO_METHODS[0]
    main = tuple(misses[is_main].quantile(list(RANGE_QUANTILES)))
    fallback = tuple(misses[~is_main].quantile(list(RANGE_QUANTILES)))
    return {name: main if name == RATIO_METHODS[0] else fallback for name in RATIO_METHODS}


@functools.lru_cache(maxsize=1)
def _cached_sales():
    return load_sales(resolve_path(load_config(), "features_data"))


@functools.lru_cache(maxsize=1)
def _out_of_fold():
    """build_out_of_fold_estimates' file as (estimates by row_id, estimates by parcel_id)."""
    path = os.path.join(resolve_path(load_config(), "models"), OOF_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run `uv run python -m src.components.market_estimate calibrate`")
    oof = pd.read_csv(path)
    return oof.set_index("row_id")["model_estimate"], oof.groupby("parcel_id")["model_estimate"].first()


@functools.lru_cache(maxsize=4)
def _pricing(base, as_of):
    """(ratio_tables, range_factors_by_method) for a starting value - "county" (ratios of
    sale / county value) or "model" (sale / out-of-fold model estimate) - as of a date."""
    sales = _cached_sales()
    column = "total_value"
    if base == "model":
        column = "model_estimate"
        sales = sales.assign(model_estimate=_out_of_fold()[0].reindex(sales.index)).dropna(subset=[column])
        sales = sales.assign(ratio=sales["sale_price"] / sales[column])
    return ratio_tables(sales, as_of), range_factors_by_method(sales, as_of, column)


def _model_estimate(house):
    """(estimate, source): an existing house's out-of-fold estimate, else the production
    model's (a what-if, or a house not in the data)."""
    if house.get("total_value") and house.get("parcel_id") is not None:
        by_parcel = _out_of_fold()[1]
        if house["parcel_id"] in by_parcel.index:
            return float(by_parcel[house["parcel_id"]]), "out-of-fold model"
    from src.components.predict import predict_total_value
    return predict_total_value(house), "production model"


def market_value(house, as_of=None):
    """Today's market price estimate for one house, with an 80% range.

    `house` is a dict like search.lookup_property returns. Leave out `total_value` (or set
    it to None) for a what-if or a house without a county assessment: the model's estimate
    of the 2023 assessed value is then the starting value. The rules, all backtested:
    - starting value: the county's 2023 assessed value; for a house built in or after
      ASSESSMENT_YEAR the model's estimate instead when the county value is below
      PARTIAL_ASSESSMENT_SHARE of it (a likely partial assessment)
    - adjustment: the median sale / starting-value ratio of recent sales in the
      neighborhood (ratios against the model's estimate when that is the starting value);
      without enough of them the time-adjusted neighborhood ratio, the area's ratio, or
      Zillow's county factor, in that order (RATIO_METHODS), with `confidence` lowered
    Needs build_out_of_fold_estimates' file (`... market_estimate calibrate`)."""
    as_of = pd.Timestamp(as_of or datetime.now()).normalize()
    county = house.get("total_value") or None
    year_built = house.get("year_built")
    model_value, model_source = _model_estimate(house)
    if county is None:
        base, reason = "model", "no county assessed value given"
    elif year_built is not None and year_built >= ASSESSMENT_YEAR and county < PARTIAL_ASSESSMENT_SHARE * model_value:
        base, reason = "model", (f"built in {int(year_built)}; its county value is {county / model_value:.0%} of the "
                                 f"model's estimate, most likely a partial assessment of an unfinished house")
    else:
        base, reason = "county", "county assessed value"
    start_value = county if base == "county" else model_value

    tables, ranges = _pricing(base, as_of.date().isoformat())
    row = pd.DataFrame([{"neighborhood_code": house.get("neighborhood_code"), "area": house.get("area")}])
    ratio, method, count, months = (series.iloc[0] for series in apply_ratios(tables, row))
    low, high = ranges[method]
    estimate = start_value * ratio
    return {
        "market_estimate": round(float(estimate), 2),
        "range_low": round(float(estimate * low), 2), "range_high": round(float(estimate * high), 2),
        "starting_value": round(float(start_value), 2),
        "starting_value_source": "county assessed value" if base == "county" else model_source,
        "reason": reason,
        "county_value": county, "model_estimate": round(model_value, 2),
        "ratio": round(float(ratio), 4), "ratio_method": method,
        "sales_used": None if pd.isna(count) else int(count), "months": None if pd.isna(months) else int(months),
        "confidence": {"neighborhood": "normal", "zillow_county": "low"}.get(method, "lower"),
        "as_of": as_of.date().isoformat(),
    }


def check_test_houses():
    """market_value for config `market_test_houses` (two houses per path), with a flag
    where a house no longer takes the path in its `expect`. A what-if is the house with
    its `changes` and no county value; `change_vs_unchanged` compares it with the same
    house, unchanged, through the same model, so it is what the change alone is worth."""
    from src.components.search import lookup_property

    rows = []
    for group, houses in load_config()["market_test_houses"].items():
        for spec in houses:
            matches = lookup_property(spec["address"])
            if len(matches) != 1:
                raise ValueError(f"{spec['address']!r} matched {len(matches)} houses, expected 1")
            house = matches[0]
            row = {"group": group, "address": spec["address"]}
            if "changes" in spec:
                unchanged = market_value(dict(house, total_value=None))
                result = market_value(dict(house, total_value=None, **spec["changes"]))
                row["changes"] = ", ".join(f"{k} {house[k]:g} -> {v:g}" for k, v in spec["changes"].items())
                row["change_vs_unchanged"] = round(result["market_estimate"] - unchanged["market_estimate"], 2)
            else:
                result = market_value(house)
            path = {k: result[k] for k in spec["expect"]}
            row.update(result, path_as_expected=path == spec["expect"])
            rows.append(row)
    return pd.DataFrame(rows)


def _errors(actual, predicted):
    pct = (predicted - actual).abs() / actual
    return {"mae": (predicted - actual).abs().mean(), "median_pct_error": pct.median(),
            "mean_pct_error": pct.mean(), "within_10pct": (pct <= 0.10).mean()}


def backtest(cutoff="2026-04-01", end=None):
    """Ratios and the Zillow factor use only data before `cutoff`; each method then
    predicts the sale price of every sale from `cutoff` to `end` (default: today)
    from that house's 2023 assessed value. Sales in neighborhoods with no ratio are
    left out of every method, so all three are scored on the same sales."""
    sales = load_sales(resolve_path(load_config(), "features_data"))
    end = pd.Timestamp(end or datetime.now())
    test = sales[(sales["sale_date"] >= pd.Timestamp(cutoff)) & (sales["sale_date"] <= end)].copy()
    ratios = neighborhood_ratios(sales, cutoff)

    covered = test["neighborhood_code"].isin(ratios.index)
    new = test["year_built"] >= ASSESSMENT_YEAR
    new_misses = (test.loc[covered & new, "sale_price"] / (test.loc[covered & new, "total_value"]
                  * test.loc[covered & new, "neighborhood_code"].map(ratios["median"])) - 1).abs()
    print(f"{len(test):,} sales from {pd.Timestamp(cutoff).date()} to {end.date()}; "
          f"{covered.mean():.1%} have a neighborhood ratio; {(covered & new).sum()} of those are houses built "
          f"in {ASSESSMENT_YEAR} or later (median error {new_misses.median():.1%}, left out)")
    test = test[covered & ~new]
    print("windows used (months: sales):", test["neighborhood_code"].map(ratios["months"]).value_counts().to_dict())
    test["sales_ratio"] = test["total_value"] * test["neighborhood_code"].map(ratios["median"])
    test["zillow_index"] = test["total_value"] * zillow_factor(cutoff)
    test["no_adjustment"] = test["total_value"]
    low, high = range_factors(sales, cutoff)  # calibrated on sales before the cutoff only
    inside = test["sale_price"].between(test["sales_ratio"] * low, test["sales_ratio"] * high)
    print(f"80% range {low - 1:+.1%} to {high - 1:+.1%} of the estimate; "
          f"{inside.mean():.1%} of test sales fell inside it")

    results = pd.DataFrame({
        method: _errors(test["sale_price"], test[method])
        for method in ["no_adjustment", "zillow_index", "sales_ratio"]
    }).T
    results["test_sales"] = len(test)
    return results


def _parcel_folds(parcel_ids, n_folds, seed=0):
    """A fold number per row, the same for every row of a parcel - the feature file lists
    some parcels twice, and a twin row in another fold would leak the house into training."""
    codes, uniques = pd.factorize(parcel_ids)
    return pd.Series(np.random.default_rng(seed).integers(n_folds, size=len(uniques))[codes], index=parcel_ids.index)


def _estimate_held_out(df, held_out, source, feature_folds):
    """Model estimates for the `held_out` rows of `df` (load_training_data plus a row_id
    column), from a model refit on the other rows with the last training run's best
    settings. Sale-derived features come from `source`; for held-out rows each of
    `feature_folds` folds also leaves out its own sales. Returns a Series by row_id."""
    from src.components.feature_engineering import add_neighborhood_aggregates, add_price_momentum
    from src.components.finetune import best_previous_model, build_model
    from src.components.training import FEATURE_SETS, TARGET, build_preprocessor

    def with_aggregates(rows, src):
        return add_price_momentum(add_neighborhood_aggregates(rows, source=src), source=src)

    cfg = load_config()
    train = with_aggregates(df[~held_out], source)
    fold = _parcel_folds(df.loc[held_out, "parcel_id"], feature_folds)
    scored = pd.concat([
        with_aggregates(df.loc[fold.index[fold == k]], source.drop(index=fold.index[fold == k]))
        for k in range(feature_folds)
    ])
    previous, params = best_previous_model(resolve_path(cfg, "reports"))
    features = FEATURE_SETS[previous["feature_set"]]
    print(f"refitting {previous['feature_set']} {previous['model']} on {len(train):,} houses; "
          f"{len(scored):,} held out", flush=True)
    preprocessor = build_preprocessor(features)
    model = build_model(previous["model"], params, cfg["training"])
    model.fit(preprocessor.fit_transform(train[features]), np.log1p(train[TARGET]))
    preds = np.expm1(model.predict(preprocessor.transform(scored[features])))
    return pd.Series(preds, index=scored["row_id"].values)


def _training_rows():
    from src.components.training import load_training_data
    df = load_training_data(resolve_path(load_config(), "features_data"))
    df["row_id"] = df.index  # the aggregates' merges reset the index
    return df


def model_estimates(sales, cutoff):
    """The model's estimate of the 2023 assessed value for every house that sold from
    RATIO_HISTORY_MONTHS before `cutoff` on, indexed like `sales`.

    The model is refit on every other house, so it never saw these houses' assessed
    values. Sale-derived features use only sales before `cutoff`, and for these houses
    also leave out their own sale, as lookup_property does."""
    cutoff = pd.Timestamp(cutoff)
    df = _training_rows()
    before_cutoff = pd.to_datetime(df["sale_date"], format="mixed") < cutoff
    source = df.assign(price_per_sqft=df["price_per_sqft"].where(before_cutoff))
    start = cutoff - pd.DateOffset(months=RATIO_HISTORY_MONTHS)
    held_out = df["row_id"].isin(sales.index[sales["sale_date"] >= start])
    return _estimate_held_out(df, held_out, source, FEATURE_FOLDS).reindex(sales.index)


def build_out_of_fold_estimates():
    """Every house's model estimate from a model that never saw it (OOF_FOLDS fits, each
    leaving out one fold of parcels), saved to the models directory as OOF_FILE.

    market_value needs these instead of the production model's estimates: that model was
    trained on every house's assessed value, partial assessments included, so its estimate
    for a house it saw sits close to that house's own county value. Rerun after retraining
    or refreshing the data (about 12 minutes)."""
    df = _training_rows()
    fold = _parcel_folds(df["parcel_id"], OOF_FOLDS)
    estimates = pd.concat([_estimate_held_out(df, fold == k, df, 1) for k in range(OOF_FOLDS)])
    out = df[["parcel_id"]].assign(model_estimate=estimates.reindex(df.index))
    save_dataframe(out.reset_index(names="row_id"), resolve_path(load_config(), "models"), OOF_FILE)
    return out


CHAIN_METHODS = ["assessed_x_ratio", "assessed_x_zillow", "model_x_ratio", "model_x_calibrated_ratio",
                 "model_x_zillow", "chain"]
SEGMENTS = ["built before 2023, neighborhood ratio", "built 2023 or later, neighborhood ratio",
            "no neighborhood ratio"]
# Quarterly cutoffs for multi_cutoff_backtest: each window runs to the next cutoff, so
# every sale is scored once.
BACKTEST_CUTOFFS = ["2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01",
                    "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"]


def chain_predictions(cutoff="2026-04-01", end=None, sales=None):
    """One row per sale from `cutoff` to `end` (default: today) with every method's
    estimate, its segment, and whether the sale fell inside each 80% range - with
    ratios, ranges, the Zillow factor and the model all built from data before `cutoff`.

    Methods: the county's assessed value or the model's estimate of it, times the
    neighborhood ratio of sale / assessed value, the ratio of sale / model estimate
    ("calibrated"), or Zillow's county factor; and the full chain - model x calibrated
    ratio where the neighborhood has one, else model x Zillow."""
    cutoff = pd.Timestamp(cutoff)
    end = pd.Timestamp(end or datetime.now())
    sales = load_sales(resolve_path(load_config(), "features_data")) if sales is None else sales.copy()
    sales["model_estimate"] = model_estimates(sales, cutoff)
    calibrated = sales.assign(ratio=sales["sale_price"] / sales["model_estimate"])

    ratio = neighborhood_ratios(sales, cutoff)["median"]
    ratio_cal = neighborhood_ratios(calibrated, cutoff)["median"]
    zillow = zillow_factor(cutoff)

    test = sales[(sales["sale_date"] >= cutoff) & (sales["sale_date"] <= end)].copy()
    nbhd = test["neighborhood_code"]
    test["assessed_x_ratio"] = test["total_value"] * nbhd.map(ratio)
    test["assessed_x_zillow"] = test["total_value"] * zillow
    test["model_x_ratio"] = test["model_estimate"] * nbhd.map(ratio)
    test["model_x_calibrated_ratio"] = test["model_estimate"] * nbhd.map(ratio_cal)
    test["model_x_zillow"] = test["model_estimate"] * zillow
    test["chain"] = test["model_x_calibrated_ratio"].fillna(test["model_x_zillow"])

    covered, new = nbhd.isin(ratio_cal.index), test["year_built"] >= ASSESSMENT_YEAR
    test["segment"] = np.select([covered & ~new, covered & new], SEGMENTS[:2], SEGMENTS[2])
    for method, ratios, base in [("assessed_x_ratio", sales, "total_value"),
                                 ("model_x_calibrated_ratio", calibrated, "model_estimate")]:
        low, high = range_factors(ratios, cutoff, base=base)
        test[f"{method}_in_range"] = test["sale_price"].between(test[method] * low, test[method] * high)
    test["cutoff"] = cutoff.date()
    return test


def score_chain(test):
    """Errors per segment (and for all sales) per method, plus the share of sales inside
    each method's 80% range where one is computed."""
    rows = []
    for segment in SEGMENTS + ["all sales"]:
        in_segment = test if segment == "all sales" else test[test["segment"] == segment]
        for method in CHAIN_METHODS:
            scored = in_segment[in_segment[method].notna()]
            if len(scored):
                in_range = f"{method}_in_range"
                rows.append({"segment": segment, "method": method, "sales": len(scored),
                             "coverage": len(scored) / len(in_segment),
                             **_errors(scored["sale_price"], scored[method]),
                             "in_80pct_range": scored[in_range].mean() if in_range in scored else np.nan})
    return pd.DataFrame(rows)


def chain_backtest(cutoff="2026-04-01", end=None):
    """score_chain on one cutoff's chain_predictions."""
    return score_chain(chain_predictions(cutoff, end))


def multi_cutoff_backtest(cutoffs=BACKTEST_CUTOFFS, end=None):
    """chain_predictions for each cutoff, each window ending the day before the next
    cutoff (the last one at `end`, default today), pooled. Returns (pooled scores, median
    % error per cutoff, segment and method, per-sale predictions)."""
    sales = load_sales(resolve_path(load_config(), "features_data"))
    ends = [pd.Timestamp(c) - pd.Timedelta(days=1) for c in cutoffs[1:]] + [end]
    windows = []
    for cutoff, window_end in zip(cutoffs, ends):
        print(f"cutoff {cutoff}", flush=True)
        windows.append(chain_predictions(cutoff, window_end, sales))
    predictions = pd.concat(windows)
    per_cutoff = pd.concat([score_chain(w).assign(cutoff=w["cutoff"].iloc[0]) for w in windows])
    by_cutoff = per_cutoff.pivot_table(index=["segment", "method"], columns="cutoff", values="median_pct_error")
    return score_chain(predictions), by_cutoff, predictions


if __name__ == "__main__" and sys.argv[1:] == ["houses"]:
    houses = check_test_houses()
    shown = ["group", "address", "starting_value_source", "ratio_method", "path_as_expected", "county_value",
             "starting_value", "ratio", "sales_used", "market_estimate", "range_low", "range_high",
             "change_vs_unchanged"]
    print(houses[shown].to_string(index=False, float_format=lambda x: f"{x:,.4g}" if x < 10 else f"{x:,.0f}"))
    if not houses["path_as_expected"].all():
        print("\nSome houses took a different path than `expect` in config/config.yaml - check why before updating it.")
    save_dataframe(houses, resolve_path(load_config(), "reports"), "market_test_houses.csv")
elif __name__ == "__main__" and sys.argv[1:] == ["calibrate"]:
    build_out_of_fold_estimates()
elif __name__ == "__main__" and sys.argv[1:] == ["chain"]:
    results = chain_backtest()
    print(results.round(4).to_string(index=False))
    save_dataframe(results, resolve_path(load_config(), "reports"), "market_chain_backtest.csv")
elif __name__ == "__main__" and sys.argv[1:] == ["chain-multi"]:
    pooled, by_cutoff, predictions = multi_cutoff_backtest()
    reports_path = resolve_path(load_config(), "reports")
    print(pooled.round(4).to_string(index=False))
    print()
    print(by_cutoff.round(3).to_string())
    save_dataframe(pooled, reports_path, "market_chain_backtest_pooled.csv")
    save_dataframe(by_cutoff.reset_index(), reports_path, "market_chain_backtest_by_cutoff.csv")
    save_dataframe(predictions[["parcel_id", "cutoff", "sale_date", "sale_price", "segment", "total_value",
                                "model_estimate", *CHAIN_METHODS]], reports_path, "market_chain_predictions.csv")
elif __name__ == "__main__":
    results = backtest()
    print(results.round(4).to_string())
    save_dataframe(results.reset_index(names="method"), resolve_path(load_config(), "reports"),
                   "market_backtest.csv")

    from src.components.search import lookup_property

    for address in load_config()["test_addresses"]:
        house = lookup_property(address)[0]
        print(f"\n{address} (2023 assessed ${house['total_value']:,.0f})")
        print(market_estimate(house["total_value"], house["neighborhood_code"], house["year_built"]))
        print(market_value(house))
