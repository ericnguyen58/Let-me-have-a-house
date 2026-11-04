import numpy as np
import pandas as pd
import pytest

from src.components.feature_engineering import (
    MIN_MOMENTUM_SALES,
    add_adjusted_aggregates,
    add_interactions,
    add_neighborhood_aggregates,
    add_price_momentum,
    add_price_per_sqft,
    nearest_distance,
    school_scores,
)


def test_nearest_distance_in_miles():
    miles, idx = nearest_distance([35.0, 36.0], [-80.0, -80.0], [35.0, 50.0], [-80.0, -80.0])
    assert miles[0] == pytest.approx(0)
    assert miles[1] == pytest.approx(69.1, abs=0.1)  # one degree of latitude
    assert idx.tolist() == [0, 0]


def test_price_per_sqft_only_for_arms_length_sales():
    df = pd.DataFrame(
        {"arms_length_sale": [True, False, True], "heated_area": [2000, 2000, 0], "sale_price": [400_000] * 3}
    )
    assert add_price_per_sqft(df)["price_per_sqft"].tolist()[0] == 200
    assert df["price_per_sqft"].iloc[1:].isna().all()


def test_neighborhood_aggregates_come_from_source_rows_only():
    train = pd.DataFrame({"county": "M", "neighborhood_code": ["n1", "n1", "n1"], "price_per_sqft": [100, 200, 300]})
    test = pd.DataFrame({"county": "M", "neighborhood_code": ["n1"], "price_per_sqft": [10_000]})
    out = add_neighborhood_aggregates(test, source=train)
    assert out["neighborhood_median_price_per_sqft"].tolist() == [200]
    assert out["neighborhood_sale_count"].tolist() == [3]


def test_adjusted_price_moves_old_sales_to_the_level_before_the_assessment():
    # the county sold at 100/sqft in early 2016 and 200 in late 2022, so the neighborhood
    # that sold at 2016 prices is as expensive as the one that sold at 2022 prices
    sale = {"county": "M", "area": "A", "heated_area": 2000, "year_built": 2000}
    train = pd.DataFrame(
        [
            {**sale, "neighborhood_code": "old", "sale_date": "2016-02-01", "price_per_sqft": 100},
            {**sale, "neighborhood_code": "old", "sale_date": "2016-03-01", "price_per_sqft": 100},
            {**sale, "neighborhood_code": "cheap", "sale_date": "2016-02-01", "price_per_sqft": 50},
            {**sale, "neighborhood_code": "new", "sale_date": "2022-11-01", "price_per_sqft": 200},
            {**sale, "neighborhood_code": "new", "sale_date": "2022-12-01", "price_per_sqft": np.nan},
        ]
    )
    test = pd.DataFrame([{**sale, "neighborhood_code": n, "price_per_sqft": 10_000} for n in ["old", "cheap", "new"]])
    out = add_adjusted_aggregates(test, source=train).set_index("neighborhood_code")
    assert out["neighborhood_adjusted_price_per_sqft"].to_dict() == pytest.approx({"old": 200, "cheap": 100, "new": 200})
    assert out.loc["old", "neighborhood_sale_years"] == pytest.approx(-6.9, abs=0.05)
    assert out.loc["new", "neighborhood_sale_years"] == pytest.approx(-0.17, abs=0.01)
    assert out["area_sale_count"].tolist() == [4, 4, 4]  # sales with a price only


def test_interactions_compare_the_house_with_its_neighborhoods_sold_houses():
    df = pd.DataFrame(
        {
            "heated_area": [3000, 0],
            "year_built": [2010, 1990],
            "acreage": [0.5, 0.5],
            "neighborhood_adjusted_price_per_sqft": [200, 200],
            "neighborhood_heated_area": [1500, 1500],
            "neighborhood_year_built": [2000, 2000],
        }
    )
    out = add_interactions(df)
    assert out["neighborhood_value_estimate"].iloc[0] == 600_000
    assert out["relative_heated_area"].iloc[0] == 2
    assert out["relative_year_built"].tolist() == [10, -10]
    assert out["lot_sqft_per_heated_sqft"].iloc[0] == pytest.approx(7.26)
    assert np.isnan(out["lot_sqft_per_heated_sqft"].iloc[1])  # no heated area


def _sales(neighborhood, area, n, yearly_growth):
    t = np.linspace(0, 2.9, n)
    return pd.DataFrame(
        {
            "county": "M",
            "neighborhood_code": neighborhood,
            "area": area,
            "sale_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(t * 365.25, unit="D"),
            "price_per_sqft": 100 * (1 + yearly_growth) ** t,
        }
    )


def test_price_momentum_uses_neighborhood_growth_then_the_areas():
    busy = _sales("busy", "A", MIN_MOMENTUM_SALES, 0.10)
    quiet = _sales("quiet", "A", 5, 0.50)  # too few sales for its own trend
    out = add_price_momentum(pd.concat([busy, quiet], ignore_index=True))
    assert out.loc[out["neighborhood_code"] == "busy", "price_momentum"].iloc[0] == pytest.approx(0.10, abs=1e-3)
    area_growth = out.loc[out["neighborhood_code"] == "quiet", "price_momentum"].iloc[0]
    assert 0.10 < area_growth < 0.50


def test_school_scores_prefer_the_closest_name():
    cards = pd.DataFrame(
        {
            "name": ["J. M. Alexander Middle", "Alexander Graham Middle", "Alexander Elementary"],
            "spg_score": [70, 80, 90],
            "category_code": ["M", "M", "E"],
        }
    )
    scores = school_scores(pd.Series(["Alexander", "Unknown", None]), cards, ["M"])
    assert scores.iloc[0] == 70
    assert scores.iloc[1:].isna().all()
