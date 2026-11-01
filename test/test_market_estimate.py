import pandas as pd
import pytest

from src.components.market_estimate import ASSESSMENT_YEAR, apply_ratios, market_estimate, neighborhood_ratios

AS_OF = pd.Timestamp("2026-06-01")


def _sale(neighborhood, months_ago, ratio):
    return {"neighborhood_code": neighborhood, "sale_date": AS_OF - pd.DateOffset(months=months_ago), "ratio": ratio}


def test_neighborhood_ratios_widen_the_window_until_enough_sales():
    sales = pd.DataFrame(
        [_sale("full", m, r) for m, r in [(1, 1.1), (2, 1.2), (3, 1.3), (4, 1.2), (5, 1.2)]]
        + [_sale("slow", m, 1.5) for m in [2, 8, 18, 20]]
        + [_sale("empty", m, 2.0) for m in [3, 30]]
        + [_sale("full", -1, 9.0)]  # after as_of, never used
    )
    ratios = neighborhood_ratios(sales, AS_OF)
    assert ratios.loc["full"].tolist() == [1.2, 5, 12]
    assert ratios.loc["slow"].tolist() == [1.5, 4, 24]
    assert "empty" not in ratios.index


def test_apply_ratios_falls_back_in_order():
    def table(key, ratio):
        return pd.DataFrame({"median": [ratio], "count": [10], "months": [12]}, index=[key])

    tables = {
        "neighborhood": table("n1", 1.1),
        "neighborhood_adjusted": table("n2", 1.2),
        "area": table("MATTHEWS", 1.3),
        "zillow_county": 1.4,
    }
    rows = pd.DataFrame(
        {"neighborhood_code": ["n1", "n2", "n3", "n4"], "area": ["MATTHEWS", "MATTHEWS", "MATTHEWS", "DAVIDSON"]}
    )
    ratio, method, _, _ = apply_ratios(tables, rows)
    assert ratio.tolist() == [1.1, 1.2, 1.3, 1.4]
    assert method.tolist() == ["neighborhood", "neighborhood_adjusted", "area", "zillow_county"]


def test_new_houses_get_no_ratio_estimate():
    # returns before reading any data
    result = market_estimate(500_000, "n1", year_built=ASSESSMENT_YEAR)
    assert result["market_estimate"] is None
    assert "after the January" in result["reason"]


@pytest.mark.parametrize("year_built", [None, 1990])
def test_market_estimate_scales_the_assessed_value_by_the_neighborhood_ratio(year_built, monkeypatch):
    from src.components import market_estimate as module

    sales = pd.DataFrame([_sale("n1", m, 1.25) for m in range(1, 7)]).assign(
        sale_price=500_000, total_value=400_000, year_built=1990
    )
    monkeypatch.setattr(module, "load_sales", lambda path: sales)
    result = market_estimate(400_000, "n1", year_built=year_built, as_of=AS_OF)
    assert result["market_estimate"] == pytest.approx(500_000)
    assert result["sales_used"] == 6
