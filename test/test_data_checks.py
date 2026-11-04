"""The data checks (src/utils/data_checks.py, config/schema.yaml) against the ways an
export can change without anything failing on its own."""

import numpy as np
import pandas as pd
import pytest

from src.components import training
from src.components.data_preprocessing import meck_to_common
from src.utils.data_checks import DataCheckError, check, load_schema, problems

TODAY = pd.Timestamp("2026-11-01")


def county_spec():
    return {**load_schema()["county_raw"], "min_rows": 0}


def raw_frame(n=50):
    """A small county export that passes every check."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "parcelid": np.arange(n, dtype=float) + 100_000,
            "ownrlstnme": "SMITH",
            "ownrfrstnme": "JOHN",
            "streetnumber": rng.integers(100, 9999, n).astype(float),
            "streetname": "MAIN ST",
            "loccity": "CHARLOTTE",
            "saleprice": 300_000.0,
            "saledate": "10/12/24 00:00",
            "naldesc": "QUALIFIED",
            "deed_book": 1234.0,
            "deed_page": 56.0,
            "totlandval": 100_000.0,
            "totalbldgval": 200_000.0,
            "totalvalue": 300_000.0,
            "gisacres": 0.25,
            "landuse_description": "SINGLE FAMILY RESIDENTIAL",
            "neighborhood": "A01",
            "yearbuilt": 1990.0,
            "heatedarea": 2000.0,
            "bedrooms": 3.0,
            "fullbath": 2.0,
            "halfbath": 1.0,
            "xcoord": 35.2,
            "ycoord": -80.8,
        }
    )


def test_a_good_export_passes_and_the_schema_lists_every_column_preprocessing_reads():
    raw = raw_frame()
    assert set(raw.columns) == set(county_spec()["columns"])
    assert check(raw, "county_raw", {"county_raw": county_spec()}, TODAY) is raw
    meck_to_common(raw)  # a KeyError here: preprocessing reads a column the schema doesn't check


def test_a_changed_decimal_separator_is_caught():
    raw = raw_frame().assign(gisacres="0,25")  # read as text when the separator changes
    assert problems(raw, county_spec(), TODAY) == [
        "gisacres: read as str, not numbers (first values: '0,25', '0,25', '0,25')"
    ]


def test_swapped_coordinates_are_caught():
    raw = raw_frame()
    raw["xcoord"], raw["ycoord"] = raw["ycoord"], raw["xcoord"]  # an export that fixed the swap
    found = problems(raw, county_spec(), TODAY)
    assert [p.split(":")[0] for p in found] == ["xcoord", "ycoord"]
    assert "outside [34.95, 35.6] in 50 rows, 100.00%" in found[0]


def test_a_new_date_format_and_future_dates_are_caught():
    assert problems(raw_frame().assign(saledate="2024-12-10"), county_spec(), TODAY)[0].startswith(
        "saledate: not in the date format '%d/%m/%y %H:%M' in 50 rows"
    )
    raw = raw_frame()
    raw.loc[:9, "saledate"] = "10/12/55 00:00"  # 1955 read as 2055
    assert problems(raw, county_spec(), TODAY) == ["saledate: after today in 10 rows, 20.00%; at most 0.20% allowed"]


def test_an_unknown_town_and_too_many_missing_values_are_caught():
    raw = raw_frame()
    raw.loc[0, "loccity"] = "WAXHAW"
    raw.loc[:9, "totalvalue"] = np.nan
    assert problems(raw, county_spec(), TODAY) == [
        "loccity: unexpected values in 1 row, 2.00% ('WAXHAW')",
        "totalvalue: missing in 10 rows, 20.00%; at most 3.00% allowed",
    ]


def test_every_problem_is_reported_at_once():
    raw = raw_frame().drop(columns="heatedarea").assign(xcoord=-80.8)
    with pytest.raises(DataCheckError, match="county_raw failed 2 checks") as error:
        check(raw.head(10), "county_raw", {"county_raw": {**county_spec(), "min_rows": 0}}, TODAY)
    assert "heatedarea: column missing" in str(error.value) and "xcoord: outside" in str(error.value)
    assert problems(raw_frame(), {"min_rows": 100, "columns": {}}) == ["50 rows, at least 100 expected"]


def test_the_feature_checks_cover_every_column_training_uses():
    columns = load_schema()["features"]["columns"]
    assert {training.TARGET, *training.FEATURE_COLUMNS} <= set(columns)
