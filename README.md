# Let Me Have a House

A pipeline that turns raw NC county property records into a modeling-ready
dataset for house-value prediction, deal-finding, and similar-house search.

## Status

| Stage | File | Status |
|---|---|---|
| 1. Ingest raw county data + places | `src/components/data_ingestor.py` | ✅ implemented |
| 2. Merge counties + clean | `src/components/data_preprocessing.py` | ✅ implemented |
| 3. Feature engineering | `src/components/feature_engineering.py` | ✅ implemented |
| 4. Train sale-price / assessed-value models | `src/components/training.py` | 📝 stub |
| 5. Evaluate models | `src/components/evaluating.py` | 📝 stub |
| 6. Predict / score new listings | `src/components/predict.py` | 📝 stub |
| 7. API | `api/main.py` | 📝 stub |

Stages 4–7 are deliberately left unimplemented as practice.

## Pipeline

```
data_ingestor.py                data_preprocessing.py                feature_engineering.py
  county sources (ArcGIS/CSV/xlsx)  ─┐
  Overture Maps "place" POIs        ─┤→ merge_all_counties() → clean_properties() → build_features()
                                     │        (properties_clean.csv)         (properties_features.csv)
  nc_places_map()  ───────────────────────────────────────────────────────────↗
                                                                    (nc_places_map.csv, used for `access`)
```

- **Ingest** (`data_ingestor.py`): pages through each county's ArcGIS
  `FeatureServer` endpoint (Guilford, Buncombe, Buncombe building
  characteristics) and downloads NC-wide Overture Maps "place" POIs
  (`overturemaps download`, bbox in `config/config.yaml`). Mecklenburg and
  Wake are supplied as static exports (CAMA CSV / xlsx) rather than fetched.
- **Merge + clean** (`data_preprocessing.py`): one loader/mapper per county
  (`meck_to_common`, `bun_to_common`, `gui_to_common`, `wake_to_common`) onto
  a shared `COMMON_COLUMNS` schema, concatenated into
  `properties_combined.csv`. `clean_properties()` then:
  - nulls out `sale_date` outside `[1900, current year]` and `year_built`
    placeholders (`0` or in the future),
  - nulls `sale_price == 0`,
  - flags `arms_length_sale` (price in `[$10k, $20M]` and sold within the
    last 10 years) — the filter the sale-price model is meant to train on,
  - normalizes `land_use_class` into `single_family` / `condo` / `townhouse`
    / `vacant_land` / `other` via a best-effort county-taxonomy mapping.
  - `nc_places_map()` separately flattens the Overture places parquet into
    `nc_places_map.csv` (`lat`, `lon`, `name`, `cat`).
  - Output: `data/processed/properties_clean.csv`.
- **Feature engineering** (`feature_engineering.py`): adds, on top of the
  clean table:
  - `property_age` = current year − `year_built`.
  - `price_per_sqft` = `sale_price / heated_area`, arms-length sales only.
  - Per-`(county, neighborhood_code)` aggregates: median `price_per_sqft`,
    sale count. Computed on the full dataset rather than a train-only split
    — a known mild leakage caveat, to revisit once `training.py` exists.
  - `access`: an accessibility score (0–1) — a weighted average, across a
    curated set of place categories (grocery, park, school, hospital,
    pharmacy, restaurant, transit; see `ACCESSIBILITY_CATEGORIES`), of
    `exp(-distance_to_nearest / τ)`, distance computed via a haversine
    `BallTree`. `NaN` for houses missing coordinates (not imputed).
  - Output: `data/features/properties_features.csv`.

## Setup

```bash
uv sync
```

Requires Python ≥3.14. `overturemaps` (the places downloader) is a separate
CLI invoked via `subprocess`, not a Python API — installed as a dependency,
callable directly.

## Running a stage

```bash
uv run python -m src.components.data_ingestor       # → data/raw/**
uv run python -m src.components.data_preprocessing  # → data/processed/*.csv
uv run python -m src.components.feature_engineering # → data/features/properties_features.csv
```

Every stage reads paths from `config/config.yaml` (`paths.root` is resolved
to the repo root at import time in `src/utils/load_config.py`; every other
path in the config is relative to it).

## Known caveats

- Neighborhood price aggregates are computed on the full dataset (see
  above) — leakage against a future train/test split.
- Wake County rows have no coordinates in the current extract, so they're
  excluded from `access` and from any lat/lon-based feature, though they
  still carry sale/assessment data.
- `land_use_class` normalization is a best-effort mapping across four
  different county taxonomies, not a guaranteed 1:1 crosswalk.
- `ACCESSIBILITY_CATEGORIES`' weights and decay ranges (τ) are initial
  estimates, not fitted/validated against anything yet.

## Repo layout

```
src/components/   pipeline stages (ingest, preprocess, feature engineering, training*, predict*, evaluating*)
src/utils/        shared helpers (config loading, dataframe I/O)
src/pipeline/     (placeholder, not yet used)
api/              FastAPI app*
config/           config.yaml (paths, data sources, bbox), schema.yaml*
data/raw/         per-source downloads
data/processed/   merged + cleaned properties, places map
data/features/    modeling-ready feature table
notebooks/        exploratory notebooks (eda.ipynb validated the county loaders now promoted into data_preprocessing.py)
output/models/    trained model artifacts*

* not yet populated
```
