# House-value pipeline: cleaning → features → models → deal-finder → RAG search → API

> **Build status**: sections 1–3 are implemented in this pass. Sections
> 4–10 (training, evaluating, predict, deal-finder, RAG index, API,
> ML dependencies) are left unimplemented on purpose — the user wants to
> build those themselves as practice. This document stays in the repo as
> the spec to build against.

## Context

The repo already has raw ingestion (`src/components/data_ingestor.py`) and one-off
EDA (`notebooks/eda.ipynb`) that merges four NC county property sources
(Mecklenburg CAMA csv, Buncombe/Guilford ArcGIS JSON, Wake xlsx) into
`data/processed/properties_combined.csv` (1.26M rows). But that merge logic only
exists inline in a notebook, the `src/components/*.py` pipeline stubs
(`data_preprocessing.py`, `training.py`, `predict.py`, `evaluating.py`,
`src/utils/common.py`) are all empty, and the combined data still has obvious
junk (sale dates from 1807/2068, a $9.6B "sale"). The goal of this pass is to
turn that into a real, reusable pipeline: clean data → engineered features →
two trained regressors → a "good deal" scorer → a RAG-based similar-house
search → all served from one FastAPI app. Map/trend-dashboard visualization
is deliberately deferred to a follow-up once there's scored data to
visualize (confirmed with user).

Decisions made with the user:
- **Two prediction targets**: a sale-price model (market value) and an
  assessed-value model (`total_value`), both exposed.
- **Model**: `sklearn.ensemble.HistGradientBoostingRegressor` (native
  categorical support, one new dependency: `scikit-learn`).
- **Scope**: backend through a working API endpoint; no map/dashboard yet.
- **Similar-house search**: real RAG, not plain KNN — each property becomes
  a short text description, embedded locally (`sentence-transformers`) and
  indexed in FAISS, so the API can answer both "find houses like this
  parcel" and free-text queries. Retrieval only for now, no LLM-generated
  summary layered on top yet. Indexed only over rows with complete-enough
  features (heated_area, beds, baths, land_use_class all present) so every
  match is actually comparable — this excludes Buncombe (it has none of
  those fields) from similarity search, though it stays in the value
  models.

## Files

### 1. `src/components/data_preprocessing.py` — ✅ implemented now
Promote `notebooks/eda.ipynb`'s already-validated loaders/mappers into real
functions (reuse its logic exactly — it's been checked against the raw data):
- `load_arcgis_json(path)` — flattens Buncombe/Guilford ArcGIS payloads.
- `load_mecklenburg()` — semicolon-delimited CAMA csv, comma decimals.
- `load_wake()` — xlsx via `openpyxl`, collapse building-card grain
  (`CARD_NUMBER`) to one row per `REAL_ESTATE_ID` (keep card 1 / sum
  heated area, matching the note already in the notebook).
- Per-county mapper → `COMMON_COLUMNS` schema (same list as
  `properties_combined.csv`'s header), `pd.concat` into one frame.

New cleaning on top (this is the actual gap versus the notebook):
- `sale_date`: parse; null out anything outside `[1900, current_year]`
  (kills the 1807/2068 rows).
- `sale_price`: null out `0` and add a boolean `arms_length_sale` column —
  `True` when `sale_price` is between $10,000 and $20M **and** `sale_date`
  is within the last 10 years. This is the filter the sale-price model
  trains on, and it's also what makes `9.6e9` harmless (excluded, not
  clipped/imputed).
- `land_use_class`: normalize the free-text/coded values (`R`, `100`,
  `SINGLE FAMILY RESIDENTIAL`, `RESIDENTIAL`, `TOWN HOUSE SFR`, `V`,
  `CONDOMINIUM`, …) into a small canonical set via an explicit mapping
  dict — `single_family`, `condo`, `townhouse`, `vacant_land`, `other`.
  Documented inline as a best-effort heuristic since county land-use
  taxonomies don't line up exactly.
- Drop exact duplicate rows.

Entry point mirrors `data_ingestor.py`'s `if __name__ == "__main__"` style.
Output: `data/processed/properties_clean.csv` (keeps
`properties_combined.csv` untouched as the raw-merge artifact).

### 2. `src/components/feature_engineering.py` (new) — ✅ implemented now
Reads `properties_clean.csv`, adds:
- `property_age` = reference year − `year_built`.
- `price_per_sqft` = `sale_price / heated_area` (arms-length rows only).
- Neighborhood aggregates per `(county, neighborhood_code)`: median
  `price_per_sqft`, sale count — computed on the full clean dataset for v1
  (documented as a mild leakage caveat, acceptable at this stage).
Output: `data/features/properties_features.csv` (matches
`config["paths"]["features_data"]`).

### 3. `src/utils/common.py` — ✅ implemented now
Small shared helpers used by 2–7 below, e.g. `resolve_path(cfg, key)` (joins
`paths.root` + `paths[key]`, the pattern every notebook/`data_ingestor.py`
repeats by hand) and `save_dataframe`/`read_dataframe` wrappers.

### 4. `src/components/training.py` — 📝 practice (not implemented)
- Loads `properties_features.csv`.
- One `ColumnTransformer` + `HistGradientBoostingRegressor` pipeline,
  built twice:
  - **Model A (sale price)**: rows where `arms_length_sale`, target
    `log1p(sale_price)`, predictors = heated_area, beds, baths, acreage,
    property_age, land_use_class, county, neighborhood aggregates, lat/long.
  - **Model B (assessed value)**: full clean dataset, target
    `total_value`, same predictor set.
- Train/test split (`random_state` fixed for reproducibility), fit, save
  via `joblib.dump` to `output/models/sale_price_model.joblib` and
  `assessed_value_model.joblib` (`config["paths"]["models"]`), plus a
  small `model_metadata.json` (feature list, training date, row counts).

### 5. `src/components/evaluating.py` — 📝 practice (not implemented)
Loads each saved model + its held-out test split, reports MAE, RMSE, R²
(and MAPE for the sale-price model) overall and broken out by `county`.
Writes a short report to `reports/` (`config["paths"]["reports"]`).

### 6. `src/components/predict.py` — 📝 practice (not implemented)
`predict_sale_price(features: dict)` and `predict_assessed_value(features:
dict)` loading the joblib models once (module-level cache); a
`predict_for_parcel(county, parcel_id)` convenience that looks up the row
in `properties_features.csv`.

### 7. `src/components/deal_finder.py` (new) — 📝 practice (not implemented)
For every arms-length-sale row: `predicted_sale_price` (Model A) vs actual
`sale_price` → `deal_score = (predicted - actual) / predicted` (positive =
underpriced). Also carries `total_value` vs Model B's prediction for
context. Ranks within `(county, neighborhood_code)`. Output:
`data/processed/deal_scores.csv`.

### 8. `src/components/rag_index.py` (new) — 📝 practice (not implemented)
- Reads `properties_features.csv`, filters to rows with non-null
  `heated_area`, `bedrooms`, `bathrooms`, `land_use_class` (per the
  coverage decision above).
- `describe_property(row) -> str` builds a short text blurb, e.g.
  `"3-bed 2-bath single_family home in Cornelius, Mecklenburg County.
  1767 sqft, built 2006, 0.13 acres, assessed at $386,500."` — plain
  string formatting, no LLM call needed to generate these.
- Embeds every description with `sentence-transformers`
  (`all-MiniLM-L6-v2` — small, fast, 384-dim, standard default choice).
- Builds a FAISS `IndexFlatIP` over L2-normalized embeddings (cosine
  similarity via inner product; flat/exact search is fast enough at this
  row count and avoids the tuning an IVF index would need).
- Saves the index to `output/models/rag_index.faiss` and a row-aligned
  sidecar (`output/models/rag_index_metadata.csv`: parcel_id, county,
  description text, key display fields) so a FAISS row id maps straight
  back to a parcel.

### 9. `api/main.py` — 📝 practice (not implemented)
FastAPI app (dependency already in `pyproject.toml`):
- Startup event loads both value models, `properties_features.csv`, the
  FAISS index, its metadata sidecar, and the sentence-transformer model
  once, into memory.
- `GET /health`
- `POST /predict` — Pydantic request body (sqft, beds, baths, year_built,
  acreage, land_use_class, county, neighborhood_code, lat/long optional) →
  `{predicted_sale_price, predicted_assessed_value}`.
- `GET /parcel/{county}/{parcel_id}` — stored features + both predictions
  + deal score if present.
- `GET /deals?county=&neighborhood=&limit=` — top underpriced parcels from
  `deal_scores.csv`.
- `GET /similar/{county}/{parcel_id}?k=5` — looks up that parcel's stored
  embedding, FAISS-searches for its `k` nearest neighbors (excluding
  itself), returns their parcel_id/description/key fields.
- `POST /search` — body `{query: str, k: int}` → embeds the free-text
  query with the same sentence-transformer model, FAISS-searches, returns
  the top-k matching properties. This is the actual RAG entry point (free
  text → retrieval); no generation step on top yet.

### 10. `pyproject.toml` — 📝 practice (not implemented; steps 1–3 need no new dependencies)
Add `scikit-learn`, `sentence-transformers`, and `faiss-cpu` (via `uv add`).
`joblib` comes in transitively through scikit-learn and is imported
directly.

## Verification

1. `uv run python -m src.components.data_preprocessing` → inspect
   `data/processed/properties_clean.csv`: no `sale_date` outside
   1900–2026, no `sale_price` of 0 or the 9.6e9 outlier, `land_use_class`
   collapsed to the canonical set.
2. `uv run python -m src.components.feature_engineering` → confirm
   `data/features/properties_features.csv` has `property_age`,
   `price_per_sqft`, neighborhood aggregate columns.
3. `uv run python -m src.components.training` → confirm both `.joblib`
   files land in `output/models/`.
4. `uv run python -m src.components.evaluating` → sanity-check R² (sale
   price model should land somewhere reasonable, e.g. > 0.5) and that
   per-county breakdowns aren't wildly skewed by one county.
5. `uv run python -m src.components.deal_finder` → spot-check the top and
   bottom of `deal_scores.csv` look sane (e.g. top "deals" aren't just
   rows with missing heated_area).
6. `uv run python -m src.components.rag_index` → confirm
   `output/models/rag_index.faiss` and the metadata sidecar are written,
   and the indexed row count matches the "complete features" filter
   (should exclude ~all of Buncombe).
7. `uv run fastapi dev api/main.py`, then `curl`:
   - `/health`
   - `POST /predict` with a sample payload
   - `/deals?county=Wake&limit=5`
   - `/similar/Wake/<a real parcel_id>?k=5` — check neighbors are
     plausibly similar (comparable beds/baths/sqft/county)
   - `POST /search` with a free-text query like `"3 bedroom house in
     Charlotte under 400k"` — check results are topically on-target.
