"""Check a data file against its rules in config/schema.yaml before it is used.

    from src.utils.data_checks import check
    df = check(pd.read_csv(...), "county_raw")   # returns df, or raises DataCheckError

Every problem is collected and reported at once, so a changed export shows everything that
moved in one run rather than one KeyError at a time.
"""

import functools

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

from src.utils.load_config import PROJECT_PATH

SCHEMA_PATH = PROJECT_PATH / "config" / "schema.yaml"


class DataCheckError(ValueError):
    """A data file failed its checks; the message lists every problem."""


@functools.cache
def load_schema():
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)


def _share(mask):
    return float(mask.mean()) if len(mask) else 0.0


def _rows(mask):
    n = int(mask.sum())
    return f"{n:,} row{'s' * (n != 1)}, {_share(mask):.2%}"


def column_problems(series, rules, today):
    """What is wrong with one column under its rules, as a list of sentences."""
    found = []
    if _share(series.isna()) > rules.get("missing", 0):
        found.append(f"missing in {_rows(series.isna())}; at most {rules.get('missing', 0):.2%} allowed")
    present = series.dropna()

    if (rules.get("numeric") or "range" in rules) and not is_numeric_dtype(series):
        sample = ", ".join(repr(v) for v in present.head(3))
        return [*found, f"read as {series.dtype}, not numbers (first values: {sample})"]
    if "range" in rules:
        low, high = rules["range"]
        outside = (present < low) | (present > high)
        if _share(outside) > rules.get("outside", 0):
            found.append(
                f"outside [{low:,}, {high:,}] in {_rows(outside)} (values from {present.min():,} to {present.max():,}); "
                f"at most {rules.get('outside', 0):.2%} allowed"
            )
    if "values" in rules:
        other = ~present.isin(rules["values"])
        if _share(other) > rules.get("outside", 0):
            examples = ", ".join(repr(v) for v in pd.unique(present[other])[:5])
            found.append(f"unexpected values in {_rows(other)} ({examples})")
    if "date" in rules:
        dates = pd.to_datetime(present, format=rules["date"], errors="coerce")
        if _share(dates.isna()) > rules.get("unparsed", 0):
            examples = ", ".join(repr(v) for v in present[dates.isna()].head(3))
            found.append(f"not in the date format {rules['date']!r} in {_rows(dates.isna())} (for example {examples})")
        if _share(dates > today) > rules.get("future", 0):
            found.append(f"after today in {_rows(dates > today)}; at most {rules.get('future', 0):.2%} allowed")
    return found


def problems(df, spec, today=None):
    """Every way `df` breaks `spec` (one entry of the schema), as a list of sentences."""
    today = today or pd.Timestamp.today()
    found = []
    if len(df) < spec.get("min_rows", 0):
        found.append(f"{len(df):,} rows, at least {spec['min_rows']:,} expected")
    for column, rules in spec["columns"].items():
        if column not in df:
            found.append(f"{column}: column missing")
            continue
        found += [f"{column}: {p}" for p in column_problems(df[column], rules or {}, today)]
    return found


def check(df, name, schema=None, today=None):
    """`df` if it passes the schema's checks for `name`, else DataCheckError listing why."""
    found = problems(df, (schema or load_schema())[name], today)
    if found:
        raise DataCheckError(
            f"{name} failed {len(found)} check{'s' * (len(found) > 1)} (rules in config/schema.yaml):\n  - "
            + "\n  - ".join(found)
        )
    return df
