"""Estimate today's market price from the 2023 assessed value with a sales ratio, and
backtest it.

Self-contained: nothing else in the project imports this module, so it can be removed
without touching the rest of the pipeline (its example run uses search.lookup_property).

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

Houses that sell are not a random sample (renovated houses and flips are
overrepresented), so ratios lean a little high; medians and the price/sqft floor limit
that.
"""

from datetime import datetime

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


def range_factors(sales, as_of):
    """(low, high) multipliers for an estimate's range: RANGE_QUANTILES of sale price /
    estimate for sales in the RANGE_CALIBRATION_MONTHS before `as_of`, estimated with
    ratios as of the start of that period."""
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.DateOffset(months=RANGE_CALIBRATION_MONTHS)
    ratios = neighborhood_ratios(sales, start)
    later = sales[(sales["sale_date"] >= start) & (sales["sale_date"] < as_of)
                  & sales["neighborhood_code"].isin(ratios.index)
                  & (sales["year_built"] < ASSESSMENT_YEAR)]  # houses that get an estimate
    misses = later["sale_price"] / (later["total_value"] * later["neighborhood_code"].map(ratios["median"]))
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


if __name__ == "__main__":
    results = backtest()
    print(results.round(4).to_string())
    save_dataframe(results.reset_index(names="method"), resolve_path(load_config(), "reports"),
                   "market_backtest.csv")

    from src.components.search import lookup_property

    for address in load_config()["test_addresses"]:
        house = lookup_property(address)[0]
        print(f"\n{address} (2023 assessed ${house['total_value']:,.0f})")
        print(market_estimate(house["total_value"], house["neighborhood_code"], house["year_built"]))
