import numpy as np
import pandas as pd

from src.components.data_preprocessing import (
    is_individual_owner,
    median_split,
    nearest_label,
    normalize_land_use,
    sale_price_outlier_mask,
)


def test_normalize_land_use_maps_known_codes_and_keeps_missing():
    raw = pd.Series(["R", "CONDOMINIUM HIGH RISE", "TOWNHOUSE", "BARN", None])
    assert normalize_land_use(raw).tolist()[:4] == ["single_family", "condo", "townhouse", "other"]
    assert pd.isna(normalize_land_use(raw).iloc[4])


def test_sale_price_outliers_are_judged_on_log_scale():
    prices = pd.Series([*np.linspace(200_000, 500_000, 20), 250_000, 450_000, 5.0, 2e9, 0.0, np.nan])
    flagged = sale_price_outlier_mask(prices)
    assert flagged.tolist() == [False] * 22 + [True, True, False, False]


def test_individual_owner_filter():
    owners = pd.Series(["SMITH JOHN", "ACME HOMES LLC", "", None, "SMITH FAMILY TRUST", "FIRST BANK"])
    # family trusts are removed too (see Known issues in the README)
    assert is_individual_owner(owners).tolist() == [True, False, True, True, False, False]


def test_median_split_keeps_every_neighborhood_and_respects_max_size():
    nbhds = pd.DataFrame(
        {"latitude": [35.0, 35.1, 35.2, 35.3, 35.4], "longitude": [-80.0] * 5, "n": [10, 10, 10, 10, 10]},
        index=["a", "b", "c", "d", "e"],
    )
    parts = median_split(nbhds, max_size=20)
    assert all(part["n"].sum() <= 20 for part in parts)
    assert sorted(code for part in parts for code in part.index) == list(nbhds.index)


def test_nearest_label_takes_the_majority_nearby():
    known = pd.DataFrame(
        {"latitude": [35.0, 35.001, 35.002, 36.0], "longitude": [-80.0, -80.0, -80.0, -81.0], "city": list("AABB")}
    )
    query = pd.DataFrame({"latitude": [35.0005], "longitude": [-80.0]})
    assert nearest_label(known, "city", query, k=3).tolist() == ["A"]
