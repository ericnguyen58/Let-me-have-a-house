# Let Me Have a House

A pipeline for Mecklenburg County, North Carolina property records. It loads
county parcel data, cleans it, assigns each house to an area (a town, or a
part of Charlotte), adds house, neighborhood and accessibility features, and
trains models that estimate a property's assessed value (`total_value`). The
estimate is meant as a reference number people can use when judging a market
price.

## Research question

How much of a home's assessed value in Mecklenburg County is explained by the
house itself, the area it is in, and its access to education, health care,
emergency services and activities, and how does the value of the same house
differ between areas?

## What the data is

The data comes from the Mecklenburg County tax assessor. Each row is one
parcel as the county has it on file today. It is not a sales history and not a
listing feed.

Each row contains:

- the property as it is now: heated area, bedrooms, bathrooms, year built,
  lot size, location
- the county's assessed value (`total_value`, and its land and building parts),
  from the 2023 countywide revaluation
- only the most recent recorded sale (`sale_price`, `sale_date`). Many rows
  have no usable sale.

The data does not contain:

- houses currently for sale or their asking prices
- earlier sales of the same house
- the condition of the house at the time it sold
- the type of sale (market sale, family transfer, foreclosure, and so on)

What this means:

- The models estimate the 2023 assessed value, not today's market price.
  Whether to adjust estimates to today's prices is an open question.
- `arms_length_sale` is an estimate, because the data does not record the
  type of sale.
- Nothing in this project can find a house that is for sale now. That would
  need a source of active listings.

Places (schools, clinics, parks, fire stations and so on) come from Overture
Maps. Its categories are noisy, which is handled in `feature_engineering.py`
as described below.

## Status

| Stage | File | Status |
|---|---|---|
| Config loading | `src/utils/load_config.py` | Implemented |
| Shared helpers | `src/utils/common.py` | Implemented |
| 1. Ingest raw data | `src/components/data_ingestor.py` | Implemented |
| 2. Clean and assign areas | `src/components/data_preprocessing.py` | Implemented |
| 3. Feature engineering | `src/components/feature_engineering.py` | Implemented. School quality not yet added. |
| 4. Training | `src/components/training.py` | Implemented |
| 5. Evaluation | `src/components/evaluating.py` | 10-house validation implemented. Other planned tests not yet complete. |
| 5b. Retrain best model | `src/components/finetune.py` | Implemented |
| 6. Prediction | `src/components/predict.py` | Implemented |
| 7. Search (similar houses, past sales below estimate) | `src/components/search.py` | Implemented |
| 8. Assistant (Claude tool use) | `src/components/assistant.py` | Implemented |
| Pipeline orchestration | `src/pipeline/` | Not yet complete (empty package) |
| API | `api/main.py` | Not yet complete (empty file) |
| Docker | `Dockerfile` | Not yet complete (IDE placeholder that runs `top`) |

## Current models

Only random forest is in use for now (`training.MODEL_TYPES`). It scored
better than XGBoost on both feature sets in the last full training run. The
XGBoost setup is kept in `training.py` and can be switched back on.

| File in `output/models/` | State | Test set (about 59,000 houses) |
|---|---|---|
| `with_neighborhood_price_random_forest` (default) | Retrained by `finetune.py`, uses `price_momentum` | RMSE $74,390, MAE $29,571, R squared 0.957, average error 5.50% |
| `without_neighborhood_price_random_forest` | From the last full training run, ignores `price_momentum` | RMSE $76,400, MAE $31,269, R squared 0.955, average error 5.81% |
| `*_xgboost` | Not used. The with-neighborhood-price one no longer fits its shared preprocessor. | |

A full training run (`uv run python -m src.components.training`) retrains both
random forests with every current feature and tunes their settings again.

## Data flow

```
data_ingestor.py
  Overture Maps places     -> data/raw/NC parquet/nc_places.parquet
  (the Mecklenburg CSV is placed in data/raw/merklenburg/ by hand)
        |
        v
data_preprocessing.py
  nc_places_map()                   -> data/processed/nc_places_map.csv
  merge_all_counties()
    -> clean_properties()
    -> assign_area()                -> data/processed/properties_clean.csv
        |
        v
feature_engineering.py
  build_features()                  -> data/features/properties_features.csv
        |
        v
training.py
  2 feature sets x random forest    -> output/models/*.joblib
  metrics                           -> reports/training_metrics.csv,
                                       reports/best_hyperparameters.json
        |
        v
predict.py, search.py               (load the saved models and feature table)
        |
        v
assistant.py                        (Claude calls predict/search as tools)
```

## Setup

```bash
uv sync
```

Requires Python 3.14 or newer. `overturemaps` is used as a command-line tool
(called through `subprocess`), not as a Python import. The assistant needs
`ANTHROPIC_API_KEY`, either in the environment or in a `.env` file (see
`.env.example`).

## Running each stage

Run from the repository root:

```bash
uv run python -m src.components.data_ingestor        # download places data
uv run python -m src.components.data_preprocessing   # clean and assign areas
uv run python -m src.components.feature_engineering  # add features
uv run python -m src.components.training             # train models
uv run python -m src.components.evaluating           # validate 10 houses
uv run python -m src.components.finetune             # retrain the best model on current features
uv run python -m src.components.predict              # example estimates
uv run python -m src.components.search               # example comps and deals
uv run python -m src.components.assistant "your question here"
```

Each stage reads its inputs from the previous stage's output files, so they
must be run in order the first time. Preprocessing and feature engineering
each take about 20 seconds. Training takes much longer, because it runs a
hyperparameter search for 2 random forests on about 295,000 houses.
`finetune.py`, which retrains one model with known settings, takes about a
minute.

## Configuration

All paths and settings live in `config/config.yaml`:

- `paths`: directories for raw data, processed data, features, reports and
  models. All are relative to the project root.
- `sources` (commented out), `params`: ArcGIS download settings for the other
  counties (Guilford, Buncombe). Not used while the project is
  Mecklenburg-only. Removing the `# ` from the `sources` block turns them back
  on.
- `nc_places`: the bounding box used to download Overture Maps places. It
  covers North Carolina and the northern part of South Carolina, so houses
  near the state line see places on both sides.
- `training`: target column, test split size, random seed, and the number of
  hyperparameter search iterations and cross-validation folds.

---

## src/utils

### `load_config.py`

`load_config()` reads `config/config.yaml` and returns it as a dictionary.
It overwrites `paths.root` with the absolute path of the project root,
found from the location of this file (two directories up). This means the
code works no matter which directory it is run from, and every other path in
the config can be written relative to the root.

### `common.py`

Small helpers used by every stage:

- `resolve_path(cfg, key)`: joins the project root with one of the configured
  paths, for example `resolve_path(cfg, "raw_data")`.
- `save_dataframe(df, path, filename)`: writes a DataFrame to CSV without the
  index, creates the folder if needed, and prints the row count.
- `read_dataframe(path, filename, **kwargs)`: reads a CSV with
  `low_memory=False` (avoids mixed-type column warnings). Extra arguments are
  passed to `pd.read_csv`.

---

## src/components

### `data_ingestor.py`

Downloads raw data. Contains one class, `data_ingestor`.

- `fetch_places(path, bbox)`: runs `overturemaps download --type=place` for the
  configured bounding box and saves the result as GeoParquet. If the file
  already exists, the download is skipped.
- `ingest()`: downloads the places file with `fetch_places`, then each ArcGIS
  source listed under `sources` in the config. The sources are commented out
  while the project is Mecklenburg-only, so currently only the places file is
  downloaded.
- `_fetch_all(source)`: requests an ArcGIS source page by page, using
  `resultOffset` and `resultRecordCount`, until a page comes back smaller than
  the page size, and combines the pages into one response.
- `health_check(response)`: returns whether a response has status 200. It is
  not used anywhere.

The Mecklenburg file is not downloaded by this module. It must be placed at
`data/raw/merklenburg/Cama_Table_transformed.csv` by hand.

### `data_preprocessing.py`

Loads the Mecklenburg records, maps them to a shared schema, cleans them and
assigns each house to an area.

The loaders and mappers for Buncombe, Guilford and Wake are commented out, not
deleted, so the project can go back to several counties later.

**Shared schema.** `COMMON_COLUMNS` lists the columns the data is mapped onto:
county, parcel ID, owner name, address parts, sale price and date, deed book
and page, land / building / total value, acreage, land use class,
neighborhood code, year built, heated area, bedrooms, bathrooms, latitude and
longitude.

**Loading.** `load_mecklenburg` reads the county CSV. `meck_to_common` renames
its columns to the shared schema:

- Owner name is built from last and first name.
- Bathrooms are full baths plus 0.5 per half bath.
- The source has x and y swapped, so `xcoord` is used as latitude and `ycoord`
  as longitude.
- Situs ZIP is not available.

**Places.** `nc_places_map(path, cfg)` reads the Overture places parquet and
writes `data/processed/nc_places_map.csv` with four columns: `lat`, `lon`,
`name` (primary name) and `cat` (primary category).

**Cleaning.** `clean_properties(df)` does the following, in order:

1. Drops exact duplicate rows.
2. Converts the columns in `NUMERIC_COLUMNS` to numbers.
3. Sets `sale_date` to missing if the year is before 1900 or after the
   current year.
4. Sets `sale_price` to missing when it is 0, or when it is an outlier.
   `sale_price_outlier_mask` finds outliers with the 1.5 x IQR rule on
   `log(1 + price)`, because house prices are right-skewed.
5. Sets `year_built` to missing when it is 0 (the placeholder for unknown) or
   in the future.
6. Adds `arms_length_sale`: true when the sale price is between $10,000 and
   $20,000,000 and the sale was within the last 10 years. This is a rough
   filter for normal market sales.
7. Maps `land_use_class` to `single_family`, `condo`, `townhouse`,
   `vacant_land` or `other` using `LAND_USE_MAP`, then keeps only
   `single_family`, `condo` and `townhouse`.
8. Removes rows with heated area outside 300 to 10,000 sq ft. Rows with
   missing heated area are kept.
9. Removes rows whose owner name contains a business or institution keyword
   (LLC, TRUST, BANK, CHURCH, CITY OF, and others in
   `NON_INDIVIDUAL_OWNER_KEYWORDS`). Blank owner names are kept.
10. Drops `owner_name` and `situs_zip`.

**Areas.** `assign_area(df)` adds an `area` column, used to compare the same
house across different parts of the county:

- Each house starts with its town from `situs_city`.
- `UNINC` (unincorporated land between towns) and `STALLINGS` (52 parcels of
  a town that is mostly in Union County) are not used as areas. Each of their
  parcels takes the most common town among its 15 nearest parcels
  (`nearest_label`).
- Charlotte holds about 75% of the houses, so it is split into areas made of
  whole neighborhoods (`neighborhood_code`). `median_split` keeps cutting a
  group of neighborhoods in half by house count, across its longer side, until
  each group has at most 30,000 houses. Groups are named `CHARLOTTE_1` to
  `CHARLOTTE_12`, numbered from north to south.
- Charlotte houses without a neighborhood code take the area of their nearest
  neighbors.
- The other towns keep their natural size.

Resulting areas:

| Area | Houses |
|---|---|
| `CHARLOTTE_1` to `CHARLOTTE_12` | 14,755 to 29,985 each |
| Huntersville | 19,765 |
| Mint Hill | 11,252 |
| Matthews | 8,689 |
| Cornelius | 8,408 |
| Davidson | 4,343 |
| Pineville | 2,796 |

`run()` writes the places map, loads, cleans, assigns areas, and saves
`data/processed/properties_clean.csv`.

### `feature_engineering.py`

Reads `properties_clean.csv` and `nc_places_map.csv`, adds features, and saves
`data/features/properties_features.csv`.

**House and neighborhood features**

- `add_property_age`: current year minus year built.
- `add_price_per_sqft`: sale price divided by heated area. Only filled for
  arms-length sales with heated area above 0; missing otherwise.
- `add_neighborhood_aggregates(df, source=None)`: for each
  `(county, neighborhood_code)` group, the median price per sq ft and the
  number of sales with a price per sq ft. The values are computed from
  `source` rows, which defaults to `df` itself. The feature file uses all
  rows, and `training.py` recomputes them from training rows only.
- `add_price_momentum(df, source=None)`: adds `price_momentum`, described
  below. Computed from `source` rows in the same way.

**Price momentum.** `price_momentum` is how fast prices per sq ft were rising
in the house's neighborhood before the assessment: the yearly growth rate
from 2020 through 2022. It is a measure of what buyers expected, as far as
recorded sales can show it.

- **How it is measured.** For each neighborhood, a trend line is fitted
  through the logarithm of price per sq ft of its arms-length sales against
  sale date (`_growth_by`). The slope is turned into a yearly growth rate,
  for example 0.19 for +19% a year.
- **Neighborhood first, area as fallback.** A neighborhood needs at least 30
  sales in the window (`MIN_MOMENTUM_SALES`) for its own trend. That is true
  for 446 neighborhoods, covering about 46% of houses. Other houses use their
  area's trend. Measured by area only, the feature would take just 18 values,
  which the model could already learn from `area` itself.
- **Why 2020 to 2022.** The target is the January 1, 2023 assessment, so the
  window stops there (`MOMENTUM_START`, `MOMENTUM_END`). The feature carries
  nothing from after the target date.
- **What the values look like.** The median is +19.5% a year, matching the
  roughly 40 to 50% price rise across the county in those years. Cheaper
  areas rose faster than expensive ones (CHARLOTTE_7, the most expensive
  area, has a median of +13%), so momentum is slightly negatively correlated
  with value (-0.13).
- **Limitation.** Each house carries only its most recent sale, so a
  neighborhood's sales in different years are different houses. The trend
  partly reflects which houses happened to sell, not only price change. This
  is the likely reason for a few extreme values (-29% and +74% a year), for
  example neighborhoods where new construction started selling partway
  through the window.

**Accessibility features.** Four groups, each measured in the way that fits
how people use that kind of place:

| Group | Column | Meaning |
|---|---|---|
| Education | `education_nearest_miles` | Distance to the nearest school |
| Health care | `health_care_share` | Share of places within 5 miles that are health care |
| Emergency | `emergency_fire_miles` | Distance to the nearest fire station |
| | `emergency_police_miles` | Distance to the nearest police station |
| | `emergency_medical_miles` | Distance to the nearest emergency department |
| Activity | `activity_share` | Share of places within 5 miles that are activities |
| (control) | `places_count` | Number of places of any kind within 5 miles |

The choices behind these:

- **Distance or count.** Emergency help comes from the closest station, so
  emergency uses distance to the nearest one. For health care and activity,
  having many options nearby matters, so they use places within 5 miles
  (`COUNT_RADIUS_MILES`). A larger radius, such as 20 miles, covers most of
  the county for every house and stops telling houses apart.
- **Shares, not raw counts.** Raw counts of health care and activity places
  had a correlation of 0.91, because both mostly measured how built-up the
  area is. Now `places_count` holds that on its own, and the two shares say
  what kind of places are nearby (correlation -0.27 between them). All seven
  accessibility features have correlations of 0.5 or less with each other.
- **Raw miles, not decay scores.** Tree models split on thresholds, so a
  score such as `exp(-distance / tau)` gives them exactly the same options as
  the distance itself. Raw miles are easier to explain.
- **Distance cap.** Distances are capped at 20 miles
  (`MAX_DISTANCE_MILES`). In practice no house is more than about 8.2 miles
  from the nearest place in any group.
- **Emergency departments are found by name.** Overture's
  `emergency_department` category is mostly individual ER doctors, and real
  emergency departments are filed under many other categories (for example
  `obstetrics_and_gynecology`, `travel_service`, `outpatient_care_facility`,
  or none). `label_emergency_departments` marks a place as a real emergency
  department if its name looks like one (`ED_NAME_PATTERN`, such as
  "Emergency Department", "Emergency Room", or a name ending in ER or ED), or
  if it is on a hand-checked list of acute-care hospitals whose names do not
  say so (`ED_HOSPITAL_NAMES`). This finds 34 emergency departments and
  hospitals in and around Mecklenburg.
- **Hospitals.** A hospital with an emergency department counts as emergency.
  Every other place in the `hospital` category counts as health care.
- **Ambulances are not included.** Medic, the county ambulance service, moves
  ambulances between posts during the day, and Overture's
  `ambulance_or_ems_service` places are mostly clinicians and businesses. Fire
  stations are the better measure of emergency response, since fire crews
  are also sent to medical calls.
- **Education is distance only for now.** School quality should come first
  and distance second. Adding it needs CMS attendance zone boundaries (to know
  which school each house is assigned to) and NC School Report Card grades,
  for the school year that matches the 2023 assessment.

Helper functions:

- `nearest_distance`: nearest-place search with a `BallTree` using haversine
  distance, converted to miles.
- `nearest_place_miles`: capped distance to the nearest place in a list of
  categories.
- `places_within_miles`: number of places within a radius, for a list of
  categories or for all places.
- `add_accessibility`: builds all accessibility columns.

Houses without coordinates get missing values, not guessed ones. All
Mecklenburg houses currently have coordinates.

`build_features(df, places)` runs all of the above in order.

### `training.py`

Trains models that estimate `total_value` (the 2023 assessed value).

**Two feature sets:**

| Feature set | What it is for |
|---|---|
| `with_neighborhood_price` | The best estimate. Includes `neighborhood_median_price_per_sqft`. |
| `without_neighborhood_price` | The same features without that column. The neighborhood price already contains most of the effect of location, amenities and area, so this is the model to read when asking what `area` and the accessibility features are worth. |

Each set is trained with every model type in `MODEL_TYPES`. That is only
`random_forest` for now, so 2 models are trained. `build_search_estimators`
also defines XGBoost, which comes back by adding `"xgboost"` to
`MODEL_TYPES`.

**Features** (`FEATURE_COLUMNS`):

- categorical: `land_use_class`, `area`
- house: acreage, year built, heated area, bedrooms, bathrooms, property age
- location: latitude, longitude
- neighborhood: median price per sq ft, sale count, price momentum
- accessibility: the seven columns from `feature_engineering.py`

Land and building value are left out because `total_value` is their sum.
`neighborhood_code` is left out because its aggregates and `area` are
included.

**Split.** The data is split randomly: 80% for training, 20% for testing. It
is not split by neighborhood. The use case is valuing a house in a
neighborhood that already has sales. With a neighborhood-based split, every
test house would lack its neighborhood aggregates. All features built from
sale prices (`SALE_DERIVED_FEATURES`: the neighborhood aggregates and
`price_momentum`) are recomputed from training rows only
(`split_with_aggregates`), so a test house's own sale never feeds its own
features. About 8.5% of test houses are in neighborhoods with no training
sales. Their neighborhood price is filled with the median.

**Steps:**

- `load_training_data`: keeps rows with a positive target and drops the
  sale-derived features stored in the feature file.
- `split_with_aggregates`: splits the data and recomputes the sale-derived
  features.
- `build_preprocessor(features)`: numeric columns are filled with the median
  and standardized; categorical columns are filled with the most common value
  and one-hot encoded. One preprocessor is fitted per feature set.
- `build_search_estimators`: wraps each model in `RandomizedSearchCV`, which
  tries `search_iter` random hyperparameter combinations with `cv_folds`
  cross-validation folds and picks the one with the lowest RMSE. Random forest
  depth is capped to keep model files a reasonable size. XGBoost uses
  `tree_method="approx"` and `n_jobs=1` because other settings crashed on this
  Python / XGBoost / macOS combination. Parallelism is set on the search
  instead.
- `train_models`: for each feature set, fits the preprocessor on the training
  rows, trains on `log(1 + total_value)`, converts predictions back to
  dollars, and records RMSE, MAE, R squared and MAPE on the test set.
- `run()` saves:
  - `output/models/<feature_set>_preprocessor.joblib`
  - `output/models/<feature_set>_<model>.joblib`
  - `reports/training_metrics.csv`
  - `reports/best_hyperparameters.json`

Models trained before `price_momentum` was added still load, but they ignore
the column. Retrain to use it. Each random forest file is about 1.1 GB, so
loading one takes a few seconds.

### `evaluating.py`

**10-house validation (implemented).** A spot check that shows each model's
error house by house, instead of only as averages.

- `validate_houses(n=10, parcel_ids=None, random_state=0)`:
  - Rebuilds the test split from `training.py`, with the same settings, so
    the houses were not used to train any model. Their neighborhood
    aggregates are computed from training rows only, as in training.
  - Takes 10 random test houses, or the houses given in `parcel_ids`. A
    parcel ID that is not in the test split raises `ValueError`, because a
    house the models were trained on would give a misleadingly good result.
  - Estimates each house with every model in use (each feature set x each
    type in `MODEL_TYPES`) and adds each estimate's percent error against the
    actual `total_value`.
- `summarize(results)`: the mean and worst absolute percent error for each
  model, best first.
- The steps are also available on their own, and `finetune.py` reuses them:
  `load_splits` (the training split), `select_houses`, `add_estimates` (one
  model's estimates and errors) and `print_validation`.
- `run()` prints both tables and saves the full results, including house
  details, sale price and every estimate, to `reports/validation_houses.csv`.

The rebuilt test split only matches the one used in training if
`properties_features.csv` has not changed since the models were trained. After
rerunning feature engineering, retrain before validating.

Ten houses show how the models behave on individual cases. They are too few to
compare models; use `reports/training_metrics.csv` for that.

**Not yet complete.** Planned additions:

- A drop-column test for each accessibility group: train without it and
  compare test error on the same split, repeated over 5 splits.
- A random noise column as a baseline for feature importance.
- A simple baseline model (`heated_area x neighborhood median price per sq
  ft`) to compare the models against.
- The same-house comparison across areas: predict one house's value with its
  `area` and location features set to each area in turn.

The rule for deciding whether a feature matters should be written down before
looking at results. For example: removing it raises test MAE by at least 1% in
most splits, and it ranks above the noise column.

### `finetune.py`

Retrains the last training run's best model on the current features, then
validates it. Use it after adding a feature, instead of a full training run.

- **Which model.** `best_previous_model` reads `reports/training_metrics.csv`
  from the last full training run, picks the feature set and model type in
  use with the lowest RMSE (the measure the
  hyperparameter search optimizes), and reads the settings the search chose
  for it from `reports/best_hyperparameters.json`.
- **Retraining.** The model is trained again from scratch with those settings,
  on the same training split as `training.py`, with every current feature.
  The hyperparameter search is skipped, which makes this much faster than a
  full run. Tree models cannot be updated in place when the feature list
  changes, so this is a full retrain of one model, not incremental training.
  A single random forest uses all CPU cores (`build_model`).
- **Validation against the last run.**
  - Test-set RMSE, MAE, R squared and MAPE, side by side with the last run's
    values for the same model, and the change.
  - The same houses as the last 10-house check
    (`reports/validation_houses.csv`), with the last run's estimate and the
    new one side by side. If that file does not exist, 10 random test houses
    are used.
- **Output.** Nothing from the last run is overwritten:
  - `output/models/finetuned/<feature_set>_preprocessor.joblib` and
    `<feature_set>_<model>.joblib`
  - `reports/finetuned_metrics.csv`
  - `reports/finetuned_validation_houses.csv`
- **Using the result.** `predict.py` loads the models in `output/models/`.
  To switch it to the retrained model, move its two files from
  `output/models/finetuned/` up into `output/models/`. This replaces the
  feature set's shared preprocessor, so any other model type trained on that
  feature set stops fitting it.
- **Comparing again.** The "last run" numbers always come from the last full
  training run's files, so running `finetune.py` twice compares against the
  same baseline.
- **Limitation.** The settings were chosen before the new features existed,
  so they may no longer be the best ones. A full training run tunes them
  again.

### `predict.py`

Estimates `total_value` for one house.

- `predict_total_value(features, feature_set="with_neighborhood_price", model_name="random_forest")`:
  `features` is a dictionary of feature names to values. It returns the
  estimate in dollars.
- Any feature not given is filled the same way as in training (median or most
  common value). For the location features (`area`, latitude, longitude and
  the accessibility columns), that means a typical value for the whole county,
  so estimates are much less accurate without them.
- `feature_set` chooses between the two trained feature sets.
  `with_neighborhood_price` is the default, because it is the one meant for
  estimates. An unknown name raises `ValueError`.
- `model_name` must be a type in `training.MODEL_TYPES` (only
  `random_forest` for now); anything else raises `ValueError`.
- `load_model(feature_set, model_name)` returns the preprocessor and model.
  `load_artifact` caches loaded files, so repeated calls do not reload from
  disk. `search.py` uses the same cache.

### `search.py`

Two lookup functions that reuse the trained preprocessors and models. Both
cover all of Mecklenburg.

- `find_similar_houses(features, k=5, feature_set="with_neighborhood_price")`:
  returns the `k` houses closest to the given features, measured after
  applying the feature set's preprocessor, using scikit-learn
  `NearestNeighbors`. Houses without a `total_value` are left out. Each result
  includes parcel ID, area, basic attributes, total value, neighborhood code
  and the distance. The results come from any area. When the query leaves many
  features out, the missing ones are filled with typical values, and the
  closest houses can be in a different area from the one given.
- `find_deals(model_name="random_forest", feature_set="with_neighborhood_price", top_n=10, min_sale_price=10000, area=None)`:
  despite the name, this does not find houses for sale. It finds past
  arms-length sales that were well below the model's estimate, ranked by
  `deal_score = (predicted_value - sale_price) / predicted_value`. `area`
  (for example `"MATTHEWS"` or `"CHARLOTTE_3"`) limits results to one area. It
  only considers sales that:
  - are at least `min_sale_price`,
  - are at least $50 per sq ft (`MIN_PRICE_PER_SQFT`), to remove family
    transfers and other non-market sales,
  - happened within the last 2 years (`MAX_SALE_AGE_YEARS`), so that price
    growth since the sale is not mistaken for a deal.
- Limitations of `find_deals`:
  - The estimate is the 2023 assessed value, which is usually below market
    price. So `deal_score` mixes the normal gap between assessed and market
    value with actual bargains.
  - The model estimates from the property's current record. If a house was
    bought cheap in poor condition and then renovated, the current record
    describes the renovated house, so the old sale looks like a deal. The data
    has no renovation date or condition field, so this cannot be filtered out.
  - Results should be treated as leads to check, not confirmed deals.
- `_index(feature_set)` caches the feature table and fitted neighbor index for
  each feature set, so repeated queries are fast.

### `assistant.py`

Lets a user ask questions in plain language. Claude decides which functions
to call and writes the answer from their results.

- Three tools are registered with the `@beta_tool` decorator from the
  Anthropic SDK:
  - `predict_total_value`: takes any of the 20 features plus `feature_set`,
    and calls `predict.predict_total_value` with the default random forest.
  - `find_similar_houses`: takes the same features plus `feature_set` and `k`,
    and calls `search.find_similar_houses`.
  - `find_deals`: takes `feature_set`, `top_n` and an optional `area`, and
    calls `search.find_deals`.
- The two feature-based tools share one block of argument descriptions
  (`FEATURE_ARGS_DOC`), which `_with_feature_docs` inserts into each docstring
  before `@beta_tool` turns the docstring into the tool schema.
- Tools return JSON strings. Errors, such as an unknown feature set or models
  that have not been trained yet, are returned as `{"error": ...}` so Claude
  can explain them.
- `SYSTEM_PROMPT` tells Claude:
  - the estimate is the 2023 assessed value, a reference point rather than a
    market price,
  - the list of areas,
  - to ask for the house's area and location when missing, since estimates are
    much weaker without them,
  - to only give dollar amounts that came from a tool,
  - to mention the renovation and assessed-versus-market caveats whenever it
    shows deals.
- `ask(question)`: runs the Anthropic SDK tool runner, which repeats the
  call-tool / return-result loop until Claude gives a final answer, and
  returns that answer's text.
- The model is `claude-opus-5` (`MODEL`). Server-side refusal fallback is
  turned on (`fallbacks="default"`): if the model declines a request, the API
  runs it again on a fallback model chosen by Anthropic. If the whole chain
  declines, `ask` returns "The request was declined."
- Run from the command line with the question as arguments. If no question is
  given, a built-in example question is used.

---

## src/pipeline

Not yet complete. Contains only an empty `__init__.py`. Each stage is
currently run on its own with `python -m`.

## Known issues

- The models estimate the 2023 assessed value, which can differ from today's
  market price.
- The owner-name filter removes names containing "TRUST". Many ordinary
  homeowners hold their house in a family trust, so this likely removes more
  expensive homes than cheap ones.
- `arms_length_sale` is only a price and date range check.
- Overture places data is incomplete. For example, it has no point named for
  the Novant Huntersville and Mint Hill hospitals, so `ED_HOSPITAL_NAMES` uses
  another building on each campus instead. A hospital with no point anywhere on
  its campus cannot be added this way.
- Overture categories are noisy beyond hospitals. For example, some
  `hospital` places are individual doctors or insurance companies. These add a
  little noise to `health_care_share`.
- School quality is not yet included.
- `price_momentum` is partly affected by which houses happened to sell in
  each year (see `feature_engineering.py`).
- Four towns (Matthews, Cornelius, Davidson, Pineville) have fewer than 10,000
  houses, so results for them are less reliable than for the other areas.
- The two random forests are out of step until the next full training run:
  only the default one uses `price_momentum` (see "Current models").
- Parcel IDs are stored as the county's `parcelid`, which has no leading
  zeros (`733316`). The county's own lookup uses 8 digits (`00733316`), so pad
  with zeros when checking a parcel there.
- `config/schema.yaml` is empty, and `Dockerfile` is an IDE placeholder.

## Repository layout

```
src/components/   pipeline stages
src/utils/        config loading and CSV helpers
src/pipeline/     not yet complete
api/              not yet complete (empty main.py)
config/           config.yaml, schema.yaml (empty)
data/raw/         downloaded and hand-placed source files
data/processed/   cleaned properties and places map
data/features/    feature table used for training and search
notebooks/        exploration notebooks
output/models/    trained preprocessors and models (finetuned/ holds retrained ones)
reports/          training metrics, chosen hyperparameters, house validation
test/             no tests yet
Dockerfile        placeholder, not yet complete
```
