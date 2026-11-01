import pytest

from src.components import search
from src.components.training import FEATURE_COLUMNS


@pytest.fixture
def houses(features, search_on):
    features.loc[0, ["situs_street_num", "situs_street_name", "situs_city"]] = [3612, "ABBEY HILL LN", "CHARLOTTE"]
    features.loc[1, ["situs_street_num", "situs_street_name", "situs_city"]] = [3612, "ABBEY RD", "MATTHEWS"]
    search_on(features)
    return features


def test_lookup_by_address_matches_number_and_first_street_word(houses):
    matches = search.lookup_property("3612 Abbey Hill Lane, Charlotte NC")
    assert [m["parcel_id"] for m in matches] == [1.0]
    assert len(search.lookup_property("3612 Abbey")) == 2  # no city: both towns


def test_lookup_by_parcel_id_returns_every_model_feature(houses):
    (house,) = search.lookup_property(parcel_id="5")
    assert set(FEATURE_COLUMNS) <= set(house)


def test_lookup_leaves_out_the_houses_own_sale(houses):
    houses.loc[0, "price_per_sqft"] = 100_000  # an absurd sale would move its own neighborhood median
    (house,) = search.lookup_property(parcel_id="1")
    others = houses[(houses["neighborhood_code"] == houses.loc[0, "neighborhood_code"]) & (houses["parcel_id"] != 1)]
    assert house["neighborhood_median_price_per_sqft"] == pytest.approx(others["price_per_sqft"].median())


def test_lookup_needs_a_house_number(houses):
    with pytest.raises(ValueError, match="house number"):
        search.lookup_property("Abbey Hill Ln")
    with pytest.raises(ValueError, match="address or a parcel_id"):
        search.lookup_property()


def test_area_stats_and_typical_features(houses):
    stats = search.area_stats()
    assert {row["area"] for row in stats} == set(houses["area"])
    assert (
        search.typical_features("MATTHEWS")["heated_area"]
        == houses.loc[houses["area"] == "MATTHEWS", "heated_area"].median()
    )
    with pytest.raises(ValueError, match="unknown area"):
        search.typical_features("ATLANTIS")


def test_find_houses_rejects_an_unknown_sort():
    with pytest.raises(ValueError, match="unknown sort_by"):
        search.find_houses(sort_by="cheapest")
