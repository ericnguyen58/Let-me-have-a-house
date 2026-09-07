import os

import pandas as pd


def resolve_path(cfg, key):
    """Join the project root with a configured path, e.g. resolve_path(cfg, 'raw_data')."""
    return cfg["paths"]["root"] + cfg["paths"][key]


def save_dataframe(df, path, filename):
    """Write `df` to `path/filename` as csv, creating `path` if needed."""
    os.makedirs(path, exist_ok=True)
    out_file = os.path.join(path, filename)
    df.to_csv(out_file, index=False)
    print(f"wrote {len(df)} rows to {out_file}")
    return out_file


def read_dataframe(path, filename, **kwargs):
    """Read `path/filename` as csv. Extra kwargs are forwarded to pd.read_csv."""
    return pd.read_csv(os.path.join(path, filename), **kwargs)
