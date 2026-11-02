"""Checks against the real feature table and trained models. Skipped when they are
missing (as in CI); run locally after the pipeline, e.g. `uv run pytest -m data`."""

import pytest

from src.components import market_estimate, predict, search
from src.components.data_preprocessing import load_mecklenburg
from src.utils.common import read_dataframe, resolve_path
from src.utils.data_checks import check
from src.utils.load_config import load_config

pytestmark = pytest.mark.data


@pytest.mark.parametrize("address", load_config()["test_addresses"])
def test_each_test_address_is_one_house(address):
    assert len(search.lookup_property(address)) == 1


@pytest.mark.parametrize("address", load_config()["test_addresses"][:2])
def test_the_model_is_close_to_the_county_value(address):
    house = search.lookup_property(address)[0]
    assert predict.predict_total_value(house) == pytest.approx(house["total_value"], rel=0.2)


def test_market_value_takes_the_expected_path_for_each_test_house():
    houses = market_estimate.check_test_houses()
    unexpected = houses.loc[~houses["path_as_expected"], ["address", "starting_value_source", "ratio_method"]]
    assert unexpected.empty, f"update config market_test_houses only after checking why:\n{unexpected}"


def test_the_county_export_and_the_feature_table_pass_their_checks():
    cfg = load_config()
    load_mecklenburg(resolve_path(cfg, "raw_data"))  # checks as it reads
    check(read_dataframe(resolve_path(cfg, "features_data"), "properties_features.csv"), "features")
