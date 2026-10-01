# Let Me Have a House

A pipeline for Mecklenburg County, North Carolina property records. It loads
county parcel data, cleans it, assigns each house to an area (a town, or a
part of Charlotte), adds house, neighborhood, accessibility, school, crime,
flood and transit features, and
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
- only the most recent recorded sale (`sale_price`, `sale_date`), with the
  county's code for whether it was a market sale (`sale_qualification`, from
  the county's `naldesc`: "Qualified", or a reason such as related parties,
  foreclosure or a bank seller). Many rows have no usable sale.

The data does not contain:

- houses currently for sale or their asking prices
- earlier sales of the same house
- the condition of the house at the time it sold

What this means:

- The models estimate the 2023 assessed value, not today's market price.
  Whether to adjust estimates to today's prices is an open question.
- `arms_length_sale` follows the county's own sale code, so it is as good as
  the county's classification.
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
| 3. Feature engineering | `src/components/feature_engineering.py` | Implemented |
| 4. Training | `src/components/training.py` | Implemented |
| 5. Evaluation | `src/components/evaluating.py` | 10-house validation implemented. Other planned tests not yet complete. |
| 5b. Retrain best model | `src/components/finetune.py` | Implemented |
| 6. Prediction | `src/components/predict.py` | Implemented |
| 7. Search (property lookup, area stats, similar houses, past sales below estimate) | `src/components/search.py` | Implemented |
| 8. Assistant (Claude tool use) | `src/components/assistant.py` | Implemented |
| Market estimate (sales ratio) and backtest | `src/components/market_estimate.py` | Implemented, standalone |
| Pipeline orchestration | `src/pipeline/` | Not yet complete (empty package) |
| API | `api/main.py` | Not yet complete (empty file) |
| Docker | `Dockerfile` | Not yet complete (IDE placeholder that runs `top`) |

## Current models

Only histogram gradient boosting (scikit-learn's
`HistGradientBoostingRegressor`) is in use for now (`training.MODEL_TYPES`).
Models are chosen by MAE in dollars: the aim is a smaller typical miss, even
if RMSE or R squared had to give a little. In a comparison on a validation
split carved from the training rows, gradient boosting beat the tuned random
forest on every metric anyway (MAE $29,407 against $33,066), and its model
files are 45 to 70 MB instead of about 1.1 GB. The random forest and XGBoost
setups are kept in `training.py` and can be switched back on.

Both models come from the latest full training run (about 20 minutes). It
uses all 27 features, including the school, crime, flood and
light rail features, and the about 9,200 golf, waterfront and high-rise homes
that were added back (see `data_preprocessing.py`). Market sales are the ones
the county marked "Qualified".

| File in `output/models/` | Test set (about 61,000 houses) |
|---|---|
| `with_neighborhood_price_hist_gradient_boosting` (default) | RMSE $76,440, MAE $29,614, R squared 0.961, average error 5.10% |
| `without_neighborhood_price_hist_gradient_boosting` | RMSE $75,598, MAE $29,778, R squared 0.962, average error 5.14% |
| `*_random_forest` | Previous models, not used. RMSE $86,183, MAE $32,826, R squared 0.950, average error 5.71% (default feature set); $88,283, $34,435, 0.948, 6.00% (without neighborhood price). Their metrics and settings are kept in `reports/random_forest_run/`. |
| `*_xgboost` | Not used, and no longer fit the current preprocessors. |

On the same test set, the switch from random forest lowered MAE by 9.8%
(default) and 13.5% (without neighborhood price). With gradient boosting the
two feature sets are almost equally accurate, so the neighborhood price
column adds little once the other location features are in.

The best settings for both feature sets: `learning_rate` 0.05,
`max_leaf_nodes` 255, `min_samples_leaf` 10, `max_features` 0.5,
`l2_regularization` 0.1. Early stopping chose 1,589 trees (default) and
2,327 trees (without neighborhood price).

**Test houses** (held out of training; `evaluating.validate_houses` with
their parcel IDs):

| House | Assessed | Default model | Without neighborhood price |
|---|---|---|---|
| 4817 Kelly Woods Ln | $1,191,900 | $1,296,627 (+8.8%) | $1,340,598 (+12.5%) |
| 3612 Abbey Hill Ln | $612,600 | $596,105 (−2.7%) | $601,817 (−1.8%) |

These two houses came out better with the random forest (+5.5% and +1.2%).
Two houses are too few to judge by; on the 10-house check the mean error went
from 5.7% to 4.6%.

`predict.py` and the assistant give slightly different numbers for the same
houses ($1,170,346 and $595,113 with the default model). `lookup_property`
computes the neighborhood price features from every other house's sales,
while validation uses training sales only. Both leave out the house's own
sale.

A full training run (`uv run python -m src.components.training`) retrains both
models with every current feature and tunes their settings again.

## Data flow

```
data_ingestor.py
  Overture Maps places     -> data/raw/NC parquet/nc_places.parquet
  Mecklenburg GIS layers   -> data/raw/meck_gis/*.geojson, quality_of_life.csv
  NC school report cards   -> data/raw/school_report_cards/*.xlsx
  Zillow county index      -> data/raw/zillow/county_zhvi.csv
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
  2 feature sets x gradient boosting -> output/models/*.joblib
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
uv run python -m src.components.data_ingestor        # download places, GIS, school and Zillow data
uv run python -m src.components.data_preprocessing   # clean and assign areas
uv run python -m src.components.feature_engineering  # add features
uv run python -m src.components.training             # train models
uv run python -m src.components.evaluating           # validate 10 houses
uv run python -m src.components.finetune             # retrain the best model on current features
uv run python -m src.components.predict              # example estimates
uv run python -m src.components.search               # example comps and deals
uv run python -m src.components.assistant "your question here"
uv run python -m src.components.assistant             # multi-turn chat
uv run python -m src.components.market_estimate       # backtest today's-price methods
```

Each stage reads its inputs from the previous stage's output files, so they
must be run in order the first time. Preprocessing and feature engineering
each take about 20 seconds. Training takes much longer (about 20 minutes), because it runs a
hyperparameter search for 2 gradient boosting models on about 304,000 houses.
`finetune.py`, which retrains one model with known settings, takes about 2.5
minutes.

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
- `meck_gis`: the Mecklenburg County GIS layers to download (school zones,
  neighborhood profile areas, floodplain, Blue Line stations), and which
  Quality of Life indicators and year to keep.
- `report_cards`: the two NC School Report Card zip files and which files to
  extract from them.
- `zillow_zhvi`: the Zillow county home value index CSV.
- `test_addresses`: the two houses every example run and check uses:
  4817 Kelly Woods Ln (28277) and 3612 Abbey Hill Ln (28210). The example runs
  of `predict.py`, `search.py` and `market_estimate.py`, and the SHAP
  notebook's test-house section, read them from here. `training.py` always
  puts them in the test set, so no model is trained on them.
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
- `fetch_meck_layers()`: downloads each layer in `meck_gis.layers` from
  Mecklenburg County GIS as GeoJSON (latitude/longitude), and the chosen
  Quality of Life indicators for one year as `quality_of_life.csv`. Only the
  indicator values are downloaded; the neighborhood shapes come from the
  `NeighborhoodProfileAreas` layer.
- `fetch_report_cards()`: downloads the two NC School Report Card zip files
  (about 270 MB together) and keeps only `rcd_acc_spg2.xlsx` (school
  performance grades) and `rcd_location.xlsx` (school names and codes).
  `dpi.nc.gov` refuses requests without a browser-like User-Agent, so one is
  sent.
- `fetch_zillow()`: downloads Zillow's monthly home value index by county.
- `ingest()`: downloads the places file with `fetch_places`, then runs the three
  functions above, then each ArcGIS source listed under `sources` in the
  config. The sources are commented out while the project is
  Mecklenburg-only. Every download is skipped if its file already exists.
- `_fetch_all(url, page_size, params)`: requests an ArcGIS layer page by page, using
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
6. Adds `arms_length_sale`: true when the county marked the sale
   "Qualified" (`sale_qualification`, `QUALIFIED_SALE`), the sale price is
   between $10,000 and $20,000,000, and the sale was within the last 10
   years. About 23% of sales that pass the price and date check are not
   Qualified: sales of several parcels in one deed, related-party sales,
   sales by banks or government, foreclosures and others. Before the county
   code was used, those counted as market sales. For example, 4817 Kelly Woods
   Ln's 2016 sale for $470,000 was a sale by a bank and pulled its
   neighborhood's median price down.
7. Maps `land_use_class` to `single_family`, `condo`, `townhouse`,
   `vacant_land` or `other` using `LAND_USE_MAP`, then keeps only
   `single_family`, `condo` and `townhouse`. The county's labels with a
   suffix count as homes too: `SINGLE FAMILY RESIDENTIAL - GOLF`,
   `- WATERFRONT` and `RURAL HOMESITE` as single family, `CONDOMINIUM WATER
   FRONTAGE` and `CONDOMINIUM HIGH RISE` as condo. Before they were mapped,
   about 9,200 of these homes, mostly on golf courses and Lake Norman, were
   dropped as "other". `MULTI FAMILY TOWNHOUSE` is still left out, since it is
   unclear whether those are individual homes.
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
  `CHARLOTTE_15`, numbered from north to south. Charlotte has about 238,000
  houses, so most groups end up near 15,000; `CHARLOTTE_2` is the one group
  that stayed just under 30,000 without being split.
- Charlotte houses without a neighborhood code take the area of their nearest
  neighbors.
- The other towns keep their natural size.

Resulting areas:

| Area | Houses |
|---|---|
| `CHARLOTTE_1` to `CHARLOTTE_15` | 14,704 to 15,849 each, except `CHARLOTTE_2` with 29,570 |
| Huntersville | 20,688 |
| Mint Hill | 11,520 |
| Cornelius | 10,897 |
| Matthews | 8,690 |
| Davidson | 5,335 |
| Pineville | 2,796 |

`run()` writes the places map, loads, cleans, assigns areas, and saves
`data/processed/properties_clean.csv`.

### `feature_engineering.py`

Reads `properties_clean.csv`, `nc_places_map.csv` and the downloaded GIS,
school report card files, adds features, and saves
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
  area's trend. Measured by area only, the feature would take just 21 values,
  which the model could already learn from `area` itself.
- **Why 2020 to 2022.** The target is the January 1, 2023 assessment, so the
  window stops there (`MOMENTUM_START`, `MOMENTUM_END`). The feature carries
  nothing from after the target date.
- **What the values look like.** The median is +19.3% a year, matching the
  roughly 40 to 50% price rise across the county in those years. Cheaper
  areas rose faster than expensive ones (CHARLOTTE_9, the most expensive
  area, has a median of +14%), so momentum is slightly negatively correlated
  with value (-0.17).
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
- **Education distance is kept alongside school quality.** The assigned
  school's score (below) matters more to buyers, but distance to the nearest
  school stays as a feature.

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

**Location context** (`add_location_context`). Each house's coordinates are
matched to Mecklenburg GIS shapes with geopandas:

| Feature | What it is | Source |
|---|---|---|
| `elementary_school_score`, `middle_school_score`, `high_school_score` | NC School Performance Grade score (0-100) of the house's assigned CMS school | CMS school zones + NC School Report Cards |
| `violent_crime_rate`, `property_crime_rate` | crimes per 1,000 residents in the house's neighborhood profile area (NPA) | Quality of Life Explorer |
| `in_floodplain` | 1 if the house is inside the FEMA floodplain, else 0 | `FEMAFloodplain` |
| `light_rail_miles` | miles to the nearest LYNX Blue Line station, capped at 20 | `CATSLynxBlueLineStations` |

- **School names.** `elementary_school`, `middle_school` and `high_school`
  hold the assigned schools' names (for example Eastover, Sedgefield, Myers
  Park). They are for display in `lookup_property`, not model features.
- **Year.** School grades and crime rates are from 2022 (for schools, the
  2021-22 school year), the last full year before the January 2023 assessment.
- **Matching schools to grades.** The zone layers use short names
  ("J.W. Grier") and the report cards use full names ("Joseph W Grier
  Academy"). `school_scores` matches a zone name to a CMS school at the same
  level whose name contains all its words, ignoring initials and words such
  as "Elementary" or "Academy". If several match, it takes the one with the
  fewest extra words. K-8 schools can serve elementary and middle zones.
- **Missing values.**
  - Schools that opened after 2022 (Knights View, Governors Village, Grove
    Park, Esperanza Global, Mint Hill elementary, Cato Ridge middle, Palisades
    and Ballantyne Ridge high) have no grade. For example, 96% of Pineville
    houses have no high school score.
  - Crime rates come from CMPD, which only polices Charlotte, so houses in
    the six towns have none.
  - Missing values are filled the same way as other features (median) when
    the model is trained.
- **School zones are today's boundaries**, not the 2022 ones. CMS has redrawn
  some zones since then.

`build_features(df, places, raw_path)` runs all of the above in order.

### `training.py`

Trains models that estimate `total_value` (the 2023 assessed value).

**Two feature sets:**

| Feature set | What it is for |
|---|---|
| `with_neighborhood_price` | The best estimate. Includes `neighborhood_median_price_per_sqft`. |
| `without_neighborhood_price` | The same features without that column. The neighborhood price already contains most of the effect of location, amenities and area, so this is the model to read when asking what `area` and the accessibility features are worth. |

Each set is trained with every model type in `MODEL_TYPES`. That is only
`hist_gradient_boosting` for now, so 2 models are trained.
`build_search_estimators` also defines random forest and XGBoost, which come
back by adding `"random_forest"` or `"xgboost"` to `MODEL_TYPES`.

**Features** (`FEATURE_COLUMNS`):

- categorical: `land_use_class`, `area`
- house: acreage, year built, heated area, bedrooms, bathrooms, property age
- location: latitude, longitude
- neighborhood: median price per sq ft, sale count, price momentum
- accessibility: the seven columns from `feature_engineering.py`

Land and building value are left out because `total_value` is their sum.
`neighborhood_code` is left out because its aggregates and `area` are
included.

**Split.** The data is split randomly: 80% for training, 20% for testing.
The houses in `test_addresses` (see "Configuration") are then moved to the
test set if the random split put them in training (`test_address_parcels`).
It is not split by neighborhood. The use case is valuing a house in a
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
  cross-validation folds and picks the one with the lowest MAE in dollars
  (`dollar_mae`, which converts the log predictions back before scoring;
  `SELECTION_METRIC` names the metric for `finetune.py` too). Gradient
  boosting uses early stopping on an internal 10% split, so `max_iter` (3,000)
  is only an upper bound. Random forest depth is capped to keep model files a
  reasonable size. XGBoost uses `tree_method="approx"` and `n_jobs=1` because other settings crashed on this
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
the column. Retrain to use it. Each gradient boosting file is 45 to 70 MB;
the old random forest files are about 1.1 GB each.

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
  use with the lowest MAE (`training.SELECTION_METRIC`, the measure the
  hyperparameter search optimizes), and reads the settings the search chose
  for it from `reports/best_hyperparameters.json`.
- **Retraining.** The model is trained again from scratch with those settings,
  on the same training split as `training.py`, with every current feature.
  The hyperparameter search is skipped, which makes this much faster than a
  full run. Tree models cannot be updated in place when the feature list
  changes, so this is a full retrain of one model, not incremental training.
  The model uses all CPU cores (`build_model`).
- **Validation against the last run.**
  - Test-set RMSE, MAE, R squared and MAPE, side by side with the last run's
    values for the same model, and the change.
  - The same houses as the last 10-house check
    (`reports/validation_houses.csv`), with the last run's estimate and the
    new one side by side. If that file does not exist, 10 random test houses
    are used. If it was written for a different model type, the same houses
    are used without the last run's columns.
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

- `predict_total_value(features, feature_set="with_neighborhood_price", model_name="hist_gradient_boosting")`:
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
  `hist_gradient_boosting` for now); anything else raises `ValueError`.
- `load_model(feature_set, model_name)` returns the preprocessor and model.
  `load_artifact` caches loaded files, so repeated calls do not reload from
  disk. `search.py` uses the same cache.
- `market_adjustment()`: returns the factor to move an estimate from the
  assessment date to today, and the month it is based on. It is Zillow's
  Mecklenburg home value index in its latest month divided by December 2022.
  It only reflects the county-wide average change in prices, so the result is
  still not a market price for a specific house.

### `search.py`

Lookup functions over the feature table. `find_similar_houses` and
`find_deals` also reuse the trained preprocessors and models. All cover all of
Mecklenburg.

- `lookup_property(address=None, parcel_id=None, limit=5)`: returns the county
  record for a property: address, neighborhood code, total, land and building
  value, last sale, assigned school names, and every model feature. The result can be passed straight
  into `predict_total_value` or `find_similar_houses`, so no feature has to be
  filled with a typical value.
  - The county stores street names abbreviated (`ALTONDALE AV`), so an address
    matches on its house number plus the first word of the street name. "Ave"
    and "Avenue" both match.
  - If a town name from the matches appears in the address, only that town's
    matches are kept.
  - It can return several rows (the same street name in different towns, or
    several units at one number) or none.
  - `parcel_id` is compared as a number, so the county's 8-digit form with
    leading zeros (`00733316`) also works.
  - Missing values are returned as `None`, not `NaN`.
  - The neighborhood price features (`SALE_DERIVED_FEATURES`) are recomputed
    without the matched houses' own sales, the same way `training.py` treats
    test houses, so an estimate never uses the house's own sale price.
- `area_stats(area=None)`: one row per area (or only the given area) with the
  house count, median `total_value`, heated area, year built,
  `price_momentum`, distance to the nearest school and emergency department,
  places count, and median $/sq ft of arms-length sales. An unknown area
  raises `ValueError`.

- `find_similar_houses(features, k=5, feature_set="with_neighborhood_price")`:
  returns the `k` houses closest to the given features, measured after
  applying the feature set's preprocessor, using scikit-learn
  `NearestNeighbors`. Houses without a `total_value` are left out. Each result
  includes parcel ID, area, basic attributes, total value, neighborhood code
  and the distance. The results come from any area. When the query leaves many
  features out, the missing ones are filled with typical values, and the
  closest houses can be in a different area from the one given.
- `find_deals(model_name="hist_gradient_boosting", feature_set="with_neighborhood_price", top_n=10, min_sale_price=10000, area=None)`:
  despite the name, this does not find houses for sale. It finds past
  arms-length sales that were well below the model's estimate, ranked by
  `deal_score = (predicted_value - sale_price) / predicted_value`. `area`
  (for example `"MATTHEWS"` or `"CHARLOTTE_3"`) limits results to one area. It
  only considers sales that:
  - are at least `min_sale_price`,
  - are at least $50 per sq ft (`MIN_PRICE_PER_SQFT`), a second check
    against non-market sales the county code missed,
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

- Five tools are registered with the `@beta_tool` decorator from the
  Anthropic SDK, listed in `TOOLS`:
  - `lookup_property`: takes an `address` or `parcel_id`, and calls
    `search.lookup_property`.
  - `area_stats`: takes an optional `area`, and calls `search.area_stats`.
  - `predict_total_value`: takes any of the 27 features plus `feature_set`,
    and calls `predict.predict_total_value` with the default model.
    It also returns `adjusted_to_today`, the estimate times
    `predict.market_adjustment()`, when the Zillow file has been downloaded.
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
  - when given an address or parcel ID, to call `lookup_property` first and
    pass its feature values into the estimate and comps tools, and to ask
    which house was meant if there are several matches,
  - for a house described without an address, to use `area_stats` for typical
    values or ask for the address, since estimates are much weaker without the
    house's location,
  - to only give dollar amounts that came from a tool,
  - to mention the renovation and assessed-versus-market caveats whenever it
    shows deals.
- `_run(messages)`: runs the Anthropic SDK tool runner, which repeats the
  call-tool / return-result loop until Claude gives a final answer, and
  returns that final message.
- `ask(question)`: answers one question and returns the answer's text.
- `chat()`: multi-turn chat in the terminal, so Claude can ask follow-up
  questions such as which of several address matches was meant. Only Claude's
  final answer from each turn is kept in the history. If Claude needs an
  earlier tool result again, it calls the tool again. An empty line or Ctrl-D
  quits.
- The model is `claude-opus-5` (`MODEL`). Server-side refusal fallback is
  turned on (`fallbacks="default"`): if the model declines a request, the API
  runs it again on a fallback model chosen by Anthropic. If the whole chain
  declines, `ask` returns "The request was declined."
- Run from the command line with the question as arguments to use `ask`. With
  no arguments it starts `chat`.

---

## notebooks/shap_explain.ipynb

Explains the model with SHAP in three stages, each adding features to the
previous one: the house only, then the neighborhood, then accessibility,
school, crime, flood and light rail. It uses the same train/test split as
`training.py`, with smaller forests (depth 18, 100 trees) so SHAP runs in a
few minutes. It runs in about 6 minutes, and ends with a breakdown of each
test house.

| Stage | Test MAE | Average error |
|---|---|---|
| House only | $87,404 | 15.01% |
| + Neighborhood | $35,324 | 6.15% |
| + Everything else | $34,258 | 5.96% |

The neighborhood's median price per sq ft gives the largest gain. The new
location features add a small gain. Of those, health care share, elementary
school score and violent crime rate rank highest. See the notebook's last
section for the full reading and its limits.

### `market_estimate.py`

Moves a 2023 assessed value to today's market price with a **sales ratio**,
the check assessors use on their own values. Standalone: no other module
imports it, so deleting the file (and this section) removes it.

- `load_sales`: arms-length sales with a price, an assessed value, at least
  $50 per sq ft, and a sale date no later than today. The county file has a
  few sales dated in the future, which are dropped. Each sale's ratio is
  `sale_price / total_value`.
- `neighborhood_ratios(sales, as_of)`: each neighborhood's median ratio,
  from its own sales only. It uses the last 12 months; if the neighborhood
  had fewer than 3 sales (`MIN_NEIGHBORHOOD_SALES`), it looks back 24 and
  then 36 months (`RATIO_WINDOWS_MONTHS`). Another neighborhood's or the
  area's ratio is never used: inside one area, a golf-course neighborhood
  and the starter homes next to it can move very differently.
- `market_estimate(assessed_value, neighborhood_code)`: assessed value times
  the neighborhood's ratio, with an 80% range (`range_low`, `range_high`),
  the number of sales and the window used. There is no estimate
  (`market_estimate` is `None`, with the reason) when the neighborhood has
  fewer than 3 sales in 36 months, or when the house was built in 2023 or
  later (`year_built`, `ASSESSMENT_YEAR`). A house built after the January
  2023 revaluation doesn't have a 2023 value of the finished house the way
  older houses do; in the backtest its median error was 14.1%, against 6.5%
  for older houses. `assessed_value` can be the county's `total_value` or the
  model's estimate of it.
- `range_factors(sales, as_of)`: the range comes from how far real sales
  landed from this method's estimates. It reruns the method as of 6 months
  ago and takes the 10th and 90th percentiles of sale price / estimate for
  the sales since (houses built before 2023 only), so about 80% of sales
  should fall inside. It updates as new sales come in. The spread of the
  neighborhood's own ratios is not used: in the backtest only 33% of sales
  fell inside its middle half.
- `backtest(cutoff="2026-04-01")`: ratios and the Zillow index use only data
  before the cutoff, and each method predicts the sale price of every later
  sale from the house's 2023 assessed value. All methods are scored on the
  same sales, those that get an estimate. Results go to
  `reports/market_backtest.csv`.

Backtest on county-qualified sales from April to September 2026: 4,383
sales, of which 4,191 (95.6%) are in a neighborhood with a ratio, and 540 of
those are houses built in 2023 or later, which are left out. That leaves
3,651 sales; 83% of them used the 12-month window, the rest 24 or 36 months.

| Method | MAE | Median error | Within 10% |
|---|---|---|---|
| No adjustment | $126,748 | 15.2% | 32.6% |
| Zillow county index | $114,457 | 13.1% | 39.6% |
| Sales ratio (own neighborhood) | $62,755 | 6.5% | 66.8% |

Before new builds were left out, on 4,191 sales, the own-neighborhood ratio
had a 7.0% median error, against 8.0% for the earlier version that fell back
to the area's ratio when a neighborhood had fewer than 10 sales. The
neighborhood ratio wins even when it comes from only 3 to 5 sales (6.6%
median error, against 8.4% for the area ratio on the same sales).

The 80% range, calibrated only on sales before April, was -12.1% to +13.8% of
the estimate, and 76.2% of the later sales fell inside it. As of today it is
-10.6% to +19.7%: recent sales missed high more often than low, mostly
renovated houses selling above their neighborhood's norm.

Test houses:

| House | Assessed | Neighborhood ratio | Market estimate | 80% range |
|---|---|---|---|---|
| 4817 Kelly Woods Ln | $1,191,900 | 1.259 (W540, 4 sales, 12 months) | $1,500,904 | $1,342,171 to $1,796,961 |
| 3612 Abbey Hill Ln | $612,600 | 1.364 (U921, 6 sales, 12 months) | $835,309 | $746,968 to $1,000,075 |

**Check against Realtor.com.** Realtor.com estimates were looked up by hand
for 20 houses, one per area, none sold in the last 2 years
(`reports/realtor_check.csv`), plus the two test houses. Two of the 20 were
built after 2023 and get no estimate (445 Main St in Matthews and 2214
Transatlantic Ave), so 18 are compared:

- 17 of 18 Realtor.com estimates fall inside our 80% range, and both test
  houses' do (19 of 20).
- Our estimate is within 4.6% of Realtor.com's for the typical house (median
  absolute gap), and within 10% for 15 of 18. The 2023 assessed value alone
  is 17.4% below Realtor.com's for the typical house.
- The one outside the range, 6301 Saddlebrook Ct (ours 22% lower): its
  neighborhood ratio comes from only 3 sales in 24 months, and Realtor.com's
  $699,999 may be a list price rather than an estimate.
- Before the new-build check, 445 Main St (built 2024) was the worst miss:
  ours was 26% above Realtor.com's.

Realtor.com is also an estimate, so this shows the two agree, not that
either is right.

Limits:

- Houses that sell are not a random sample (renovated houses and flips are
  overrepresented), so ratios lean a little high.
- A ratio from 3 to 6 sales can be moved by one unusual sale. The response
  shows `sales_used` so the reader can judge how much to trust it.
- The range is the same width for every house. It doesn't get wider for a
  neighborhood with few sales.
- No estimate for about 4% of houses (neighborhood had fewer than 3 market
  sales in 3 years) and for houses built in 2023 or later, about 4% of houses
  and 13% of recent sales.
- The result is a market price estimate. The county's next revaluation will
  likely come in below it: right after the 2023 revaluation, sales ran about
  7.6% above assessed values.

## src/pipeline

Not yet complete. Contains only an empty `__init__.py`. Each stage is
currently run on its own with `python -m`.

## Known issues

- The models estimate the 2023 assessed value, which can differ from today's
  market price. `market_adjustment` moves it by the county-wide price change
  only.
- The owner-name filter removes names containing "TRUST". Many ordinary
  homeowners hold their house in a family trust, so this likely removes more
  expensive homes than cheap ones.
- Overture places data is incomplete. For example, it has no point named for
  the Novant Huntersville and Mint Hill hospitals, so `ED_HOSPITAL_NAMES` uses
  another building on each campus instead. A hospital with no point anywhere on
  its campus cannot be added this way.
- Overture categories are noisy beyond hospitals. For example, some
  `hospital` places are individual doctors or insurance companies. These add a
  little noise to `health_care_share`.
- School zones are today's boundaries, while grades are from 2022, and new
  schools have no grade.
- Crime rates cover Charlotte only.
- `price_momentum` is partly affected by which houses happened to sell in
  each year (see `feature_engineering.py`).
- Three towns (Matthews, Davidson, Pineville) have fewer than 10,000
  houses, so results for them are less reliable than for the other areas.
- Parcel IDs are stored as the county's `parcelid`, which has no leading
  zeros (`733316`). The county's own lookup uses 8 digits (`00733316`), so pad
  with zeros when checking a parcel there.
- The assistant is told to base every dollar amount on a tool result, but it
  can still add explanations no tool gave. In one test it named the
  neighborhood and guessed a reason for the value that no tool had returned.
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
notebooks/        exploration notebooks; shap_explain.ipynb explains the model stage by stage
output/models/    trained preprocessors and models (finetuned/ holds retrained ones)
reports/          training metrics, chosen hyperparameters, house validation
test/             no tests yet
Dockerfile        placeholder, not yet complete
```
