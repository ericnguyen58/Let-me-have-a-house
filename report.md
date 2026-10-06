# What drives a house's assessed value

Mecklenburg County, NC. Analyses run on 2026-10-06.

| Analysis | Source | Output |
|---|---|---|
| SHAP by scenario | `notebooks/shap_explain.ipynb` | in the notebook |
| Drop-column test | `evaluating.drop_column_test` | `reports/drop_column.csv` |
| Same house in different areas | `evaluating.same_house_by_area`, `notebooks/area_effect.ipynb` | `reports/same_house_by_area.csv`, `reports/area_premium_decomposition.csv` |

## Question

How much of a home's 2023 assessed value (`total_value`) is explained by the house itself, by the area it is in,
and by its access to education, health care, emergency services and activities? And how does the value of the
same house differ between areas?

## Answer in brief

1. **The house explains most of the spread but misses by 15% on average.** Location takes the error from 15% to
   about 5%.
2. **Accessibility and schools carry the location effect.** Accessibility alone gets 96% of the way the
   neighborhood price does. Without the neighborhood price, the model is as accurate, and the credit moves to
   accessibility and schools. The `area` label itself adds little once those are known.
3. **The same house is assessed more than twice as high in the most expensive area as in the cheapest**: a typical
   2,125 sq ft single-family house would be about $325,000 in CHARLOTTE_3 and about $713,000 in CHARLOTTE_9.
   - The premiums of central Charlotte come mostly from accessibility.
   - Those of the lake towns and southern suburbs come mostly from schools.
4. **Crime, light rail and the flood zone add little** in percent error. The light rail and flood zone effects
   that show up are location in disguise. In dollars the drop-column test is less clear for crime and the flood
   zone (section 2).

## 1. How much each part explains (SHAP scenarios)

Test-set error of each scenario (60,839 test houses):

| Scenario | Features | MAE | Average error | R squared |
|---|---|---|---|---|
| 1. House only | house | $87,566 | 15.19% | 0.764 |
| 1b. House + accessibility | house, accessibility | $33,067 | 5.62% | 0.953 |
| 1c. House + flood zone | house, `in_floodplain` | $87,337 | 15.15% | 0.764 |
| 2. + Neighborhood | house, location, neighborhood price | $30,682 | 5.25% | 0.957 |
| 3. + Everything | all 27 features | $29,413 | 5.08% | 0.963 |
| 3b. Everything except neighborhood price | 26 features | $29,862 | 5.18% | 0.963 |

Differences of about 0.1 points are within noise (see section 2).

### Findings

1. **The house alone** misses by 15% on average. Heated area is by far the strongest feature (mean |SHAP| 0.288),
   then bathrooms (0.076).
2. **Location is the biggest improvement** (15.2% to 5.3%).
3. **Accessibility on its own gets 96% of the neighborhood's gain** (scenario 1b, 15.2% to 5.6%). With no other
   location information, it carries a mean |SHAP| of 0.206 (about 23% of value):

   | Feature | Mean \|SHAP\| in 1b |
   |---|---|
   | `places_count` | 0.098 |
   | `health_care_share` | 0.089 |
   | `emergency_medical_miles` | 0.038 |
   | `emergency_police_miles` | 0.037 |

   These features say how central and built-up the surroundings are, which is most of what location means for
   value.
4. **The flood zone adds almost nothing overall** (scenario 1c, 15.19% to 15.15%). Only 0.6% of houses are in a
   floodplain. For the 388 floodplain houses in the test set, the house + flood zone model raises the estimate by
   about 10%. Once location is known (scenario 3), the effect is about 0, so the 10% is the location of floodplain
   houses, not the flood zone.
5. **With everything known, credit is split like this** (mean |SHAP| in scenarios 3 and 3b):

   | Group | 3. Everything | 3b. Without neighborhood price |
   |---|---|---|
   | house | 0.297 | 0.296 |
   | neighborhood price | 0.117 | — |
   | accessibility | 0.057 | 0.115 |
   | school | 0.056 | 0.076 |
   | location (area, coordinates, sales, momentum) | 0.029 | 0.043 |
   | crime | 0.021 | 0.021 |
   | light rail | 0.007 | 0.015 |
   | flood zone | 0.0004 | 0.0004 |

   - Without the neighborhood price, its credit goes mostly to accessibility (doubled) and schools. The error
     barely changes (5.08% to 5.18%).
   - In scenario 3, `elementary_school_score` is the 5th most important feature. In 3b, `places_count` is 2nd and
     `elementary_school_score` 3rd.
   - Directions are as expected. Higher school scores and more health care nearby raise the estimate: the
     correlation between each feature and its SHAP value is +0.91 and +0.90. More violent crime lowers it (−0.74).
6. **Test houses** (held out of training; scenario 3 model; SHAP in log value):

   | House | Assessed | Estimate | What drives it |
   |---|---|---|---|
   | 4817 Kelly Woods Ln | $1,191,900 | $1,283,815 (+7.7%) | Size (5,249 sq ft) +0.66, bathrooms +0.16. The neighborhood price ($198 per sq ft) adds about nothing, because many of its sales are from 2016-2019. |
   | 3612 Abbey Hill Ln | $612,600 | $612,302 (−0.0%) | Size +0.17. Neighborhood price ($164 per sq ft) −0.06. Nearby health care +0.04. |
   | 7631 Seton House Ln | $1,000,000 | $978,827 (−2.1%) | Size (4,056 sq ft) +0.44, bathrooms +0.13, neighborhood price ($239 per sq ft) +0.12. |

## 2. What each group adds on top of everything else (drop-column test)

The production model was retrained without one group at a time and scored on the full test set. The table shows
the change in average error against the full model, in percentage points, with a 95% bootstrap interval over
test houses.

- **Seed noise** is the range of the full model's error across 4 random seeds: 0.066 points with the
  neighborhood price, 0.046 without it.
- **"Beyond noise"** means the interval excludes 0 and the change is larger than the seed noise.

**Without the neighborhood price** (full model 5.13%):

| Dropped | Change | 95% interval | Beyond noise? |
|---|---|---|---|
| location (area, coordinates, sales, momentum) | +0.21 | +0.18 to +0.23 | yes |
| school | +0.11 | +0.09 to +0.13 | yes |
| accessibility | +0.09 | +0.06 to +0.11 | yes |
| `area` alone | +0.07 | +0.05 to +0.08 | yes |
| crime | +0.04 | +0.02 to +0.06 | no |
| `places_count` alone | +0.03 | +0.01 to +0.05 | no |
| light rail | +0.02 | +0.01 to +0.04 | no |
| flood zone | +0.01 | −0.00 to +0.03 | no |
| `health_care_share` alone | +0.01 | −0.01 to +0.03 | no |

**With the neighborhood price** (full model 5.12%), nothing is clearly needed:
- Dropping the neighborhood price itself costs only +0.02 points. Dropping location costs +0.02, accessibility
  +0.06 and crime +0.06, all within the seed noise.
- Dropping school, light rail, `area`, `places_count` or `health_care_share` *lowers* the error slightly (−0.03 to
  −0.05), also within noise.
- The flood zone shows +0.085, just past the seed noise. This is most likely noise:
  - its SHAP value is about 0;
  - the same drop without the neighborhood price is not significant (+0.01);
  - each drop was fit with one seed, and a 4-seed range underestimates the noise.

**In dollars (MAE), the with-price results look different.**
- Dropping accessibility raises MAE by 2.4%, the flood zone by 1.9% and crime by 1.5%, with bootstrap intervals
  that exclude 0. That passes the README's planned rule (MAE up by at least 1%).
- Dollar errors are dominated by expensive houses, so these groups may help with expensive homes. Floodplain
  houses may include waterfront lots (not checked).
- Seed noise was only measured for percent error, and the bootstrap leaves out seed noise. So these three can't
  be told from noise either.
- It is unlikely that the flood zone matters only when the neighborhood price is present: without it, the same
  drop costs +0.2%. That points to noise, but only the matched-seed rerun below can settle it.

**Without the neighborhood price, in MAE**, the result is the same as in percent error:

| Dropped | MAE change | 95% interval |
|---|---|---|
| location | +4.5% | +3.8% to +5.2% |
| accessibility | +2.0% | +1.3% to +2.7% |
| school | +1.7% | +1.2% to +2.3% |
| `area` | +1.1% | +0.6% to +1.6% |
| everything else | under 1% | |

**Against the plan in the README.** The README planned this test over 5 splits, with a feature counting as
important when removing it raises MAE by at least 1% in most splits. This run used one split, with seed noise and
a bootstrap instead, so it does not meet that plan. Without the neighborhood price, location, accessibility, school
and `area` pass the 1% bar on this split.

**What this means:**
- The location information is highly redundant. With the neighborhood price in, any one group can be removed and
  the others cover for it.
- Without it, location, schools and accessibility each hold information nothing else has, and `area` holds a
  little on its own.
- Crime, light rail, the flood zone and the single accessibility features don't.

## 3. The same house in different areas

A reference house keeps its own house features, and in each area is given the location of 300 real parcels of
the same type. A location is every non-house feature, taken together from one parcel, so each combination exists.
The **premium** compares the house's mean log estimate in the area with the same house given 3,000 county-wide
parcels' locations.

**Typical single-family house.** The real house closest to the county median: 2,125 sq ft, 3 bedrooms, 2.5 baths,
built 1996, 0.27 acres, assessed $465,500 in Matthews. Model without the neighborhood price:

| Area | Estimate | Premium | Raw data |
|---|---|---|---|
| CHARLOTTE_9 | $713,000 | +70% | +55% |
| CHARLOTTE_8 | $658,000 | +66% | +55% |
| CHARLOTTE_7 | $683,000 | +40% | +78% |
| DAVIDSON | $558,000 | +29% | +38% |
| CORNELIUS | $493,000 | +26% | +17% |
| CHARLOTTE_12 | $512,000 | +19% | +19% |
| CHARLOTTE_15 | $491,000 | +14% | +15% |
| CHARLOTTE_14 | $493,000 | +14% | +16% |
| MATTHEWS | $419,000 | −2% | +2% |
| HUNTERSVILLE | $417,000 | −3% | +3% |
| CHARLOTTE_13 | $386,000 | −8% | −2% |
| CHARLOTTE_6 | $363,000 | −9% | −7% |
| MINT HILL | $399,000 | −9% | −0% |
| PINEVILLE | $394,000 | −10% | +8% |
| CHARLOTTE_10 | $377,000 | −10% | −0% |
| CHARLOTTE_11 | $371,000 | −11% | −4% |
| CHARLOTTE_2 | $359,000 | −17% | −8% |
| CHARLOTTE_4 | $336,000 | −18% | −7% |
| CHARLOTTE_1 | $354,000 | −18% | −11% |
| CHARLOTTE_5 | $351,000 | −19% | −6% |
| CHARLOTTE_3 | $325,000 | −25% | −12% |

The estimate is the median over the area's 300 locations. "Raw data" is the area's median assessed value per sq ft
for houses of the same type within 20% of the size, against the county's, with no model involved.

### Findings

1. **The same house is worth 2.2 times as much in CHARLOTTE_9 as in CHARLOTTE_3.** Eight areas are above the
   county and thirteen below.
2. **The model agrees with the raw data.** The correlation between the two premiums is 0.88 to 0.95 for every
   reference house. Where they differ:
   - The model's spread is a little wider at the low end.
   - It is narrower for CHARLOTTE_7, a mixed area whose estimates range from $346,000 to $922,000.
   - Pineville is the one area where they disagree in sign.
3. **Both production models give almost the same premiums.** For the typical house, the largest gap between the
   with- and without-neighborhood-price models is 3.6 points (Cornelius).
4. **Where the premium comes from** (Shapley shares in log value, typical single-family house, model without the
   neighborhood price):

   | Kind of area | Main source | Shares |
   |---|---|---|
   | Central Charlotte (CHARLOTTE_7, 8, 9), +40% to +70% | accessibility | +0.30 to +0.33; schools up to +0.11 |
   | Lake towns and southern suburbs (Davidson, Cornelius, CHARLOTTE_12, 14, 15), +14% to +29% | schools | +0.06 to +0.19; accessibility about 0 |
   | Below-county areas | accessibility | −0.02 to −0.12 (all but CHARLOTTE_6). Schools pull most Charlotte areas and Pineville down further (−0.02 to −0.09) but lift Matthews, Huntersville and Mint Hill (+0.02 to +0.04) |

   - **The location group itself is small** (−0.07 to +0.11). The area label and coordinates add little once
     accessibility and schools are known.
   - **Crime is at most ±0.05**, and the flood zone 0.
   - **Light rail gives Cornelius +0.08, but Cornelius has no light rail.** Cornelius and Davidson are the farthest
     from it (13 miles), so the distance stands in for distance from the city or for the lake.
5. **The premium depends on the type of house.** CHARLOTTE_9 is +70% for the typical single-family house but +3%
   for the typical condo, while Davidson is +43% for the condo. Each type is compared with its own typical county
   location.
6. **For single-family houses, the premium is a similar percent at different sizes.** Abbey Hill (2,596 sq ft),
   Seton House (4,056) and Kelly Woods (5,249) have premiums close to the typical house's in most areas. For Kelly
   Woods, 9 of the below-county areas have few houses that large, so its premiums there are extrapolations.

## How we got there

**Data.** Mecklenburg County tax assessor parcels. Each row is one parcel as it is on file today: the house's
characteristics, the 2023 assessed value, and only its most recent recorded sale. Features come from
`feature_engineering.py`. In `evaluating.FEATURE_GROUPS`, shared by all three analyses, they are grouped into:

| Group | Features |
|---|---|
| house | type, size, rooms, age, lot |
| location | area, coordinates, neighborhood sale count, price momentum |
| neighborhood price | median price per sq ft |
| accessibility | 7 features |
| school | 3 features |
| crime | 2 features |
| flood | 1 feature |
| light rail | 1 feature |

**Split.** The same as `training.py`: 80% training and 20% test, `random_state` 42, giving 243,340 training and
60,839 test houses. The config test addresses are always in the test set. Neighborhood price features come from
training sales only, so no sale price leaks into the test set.

**Target.** `log(1 + total_value)`. A SHAP value or a share is a change in log value. For small values, it is
roughly a percent: +0.10 is about +10.5%.

**Models.** `HistGradientBoostingRegressor` with the production settings from `reports/best_hyperparameters.json`:
- learning rate 0.05
- 255 leaves
- at least 10 houses per leaf
- half the features per split
- L2 regularization 0.1

Early stopping is the same as in `training.py`: up to 3,000 trees, stopping after 50 rounds without improvement on
a 10% split.

| Analysis | Models used |
|---|---|
| SHAP scenarios | one new fit per scenario |
| Drop-column test | one new fit per drop, plus 3 seed refits of the full model |
| Same house by area | the saved production models in `output/models/` |

**SHAP.** `shap.TreeExplainer` on 500 sampled test houses per scenario. The one-hot columns are summed back into
one value per feature. Group importance is the mean absolute value of a group's summed SHAP values per house.

**Drop-column test.** For each group, and for `area`, `places_count` and `health_care_share` alone, the model is
refit without it on the training rows and scored on all test houses.
- The 95% interval is a paired bootstrap: 1,000 resamples of the test houses' error differences.
- The seed noise is the range of the full model's error across seeds 42, 1, 2 and 3.
- It took about 80 minutes.

**Same house by area.** The steps:
1. **Reference houses:** the 3 test addresses, and for each type (single-family, townhouse, condo) the real house
   closest to that type's median house.
2. **Locations:** for each area, 300 random parcels of the reference house's type. Each has every non-house
   feature, with neighborhood features from training sales.
3. **Premium:** exp(mean log estimate in the area − mean log estimate with 3,000 county-wide parcels of the same
   type) − 1.
4. **Size coverage:** the share of an area's locations whose neighborhood has a house of the same type within 25%
   of the reference size. Below 50% is flagged.
5. **Decomposition** (typical houses, model without the neighborhood price): exact Shapley values over 6 location
   groups. Each of 200 area parcels is paired with a county parcel, and every combination of groups is evaluated
   (64 per pair). The shares add up to the log premium.

## Changes in methodology

### In this analysis

| Change | Before | Now | Why |
|---|---|---|---|
| Model | Random forests (100 trees, depth 18), smaller than production | Gradient boosting with the production settings | Production switched to gradient boosting. The analysis now explains the model actually used. |
| Side scenarios | Three cumulative stages | Added 1b (house + accessibility), 1c (house + flood zone) and 3b (everything but the neighborhood price) | The cumulative stages show these features only *after* the neighborhood price, which hides what they explain on their own. |
| Feature groups | `neighborhood` and `flood_and_rail` | `location` and `neighborhood_price` split; `flood` and `light_rail` split; one shared definition in `evaluating.FEATURE_GROUPS` | To test the neighborhood price and the flood zone on their own, and to use the same groups in every analysis. |
| Unique value | Not measured | Drop-column test with bootstrap intervals and seed noise | SHAP splits credit between correlated features; the drop test shows what each group adds that nothing else has. |
| Same house by area | Not measured | Location swap with real parcels, raw-data check and Shapley decomposition | Answers the second half of the research question directly. |
| Test houses | Kelly Woods, Abbey Hill | Seton House Ln added | Added to `config.yaml`. It is forced into the test set, so the split moved by one house. |

**Random forest against gradient boosting** (average error):

| Scenario | Random forest | Gradient boosting |
|---|---|---|
| 1. House only | 15.01% | 15.19% |
| 1b. House + accessibility | 7.03% | 5.62% |
| 1c. House + flood zone | 15.00% | 15.15% |
| 2. + Neighborhood | 6.15% | 5.25% |
| 3. + Everything | 5.96% | 5.08% |

Gradient boosting gets more out of the accessibility and school features: in scenario 3, accessibility goes from
0.038 to 0.057 and school from 0.026 to 0.056. With the random forest, house + accessibility got 90% of the
neighborhood's gain; with gradient boosting it gets 96%.

### In the project, affecting how to read these results

- **The production model is gradient boosting** (`HistGradientBoostingRegressor`), selected by MAE in dollars
  instead of RMSE. The default model's MAE fell from $32,826 (random forest) to $29,614 (5.71% to 5.10%).
- **The training data now includes about 9,200 golf-course, waterfront and high-rise homes** that were
  previously excluded, so errors are not directly comparable with older runs.
- **Market sales follow the county's own sale code** ("Qualified"), and a house's own sale is left out of its
  neighborhood price.

## Limits

- **Attribution:**
  - SHAP and the Shapley decomposition show what the model relies on, not what causes value.
  - The decomposition mixes groups from two parcels, which gives combinations no real house has.
- **Noise:**
  - Differences of about 0.1 points in average error are within noise. Reordering the features alone moved
    scenarios 2 and 3 by 0.02 points.
  - The drop-column test fits each drop with one seed.
  - Area premiums come from 300 locations per area, so gaps of a few points between areas are not meaningful.
- **Data:**
  - The neighborhood price is a median over 10 years of sales. It understates neighborhoods whose prices rose a lot.
  - Crime rates cover Charlotte only. The six towns get the county median, so their crime share is about 0 by
    construction.
  - School zones are today's, while grades are from 2022.
  - The flood zone results rest on 0.6% of houses.
- **Target:** everything here is the 2023 assessed value, not today's market price.

## Next steps

- **Firm up the drop-column test.** Fit each drop with the same 4 seeds as the full model and compare seed by
  seed, in both percent error and MAE, or repeat over the 5 splits the README planned. That would settle crime,
  the flood zone and accessibility in the with-price set, at about 4 to 5 times the run time (5 to 7 hours).
- **Add the README's other planned baselines:** a random noise column, and the simple `heated_area` x neighborhood
  price per sq ft model.
- **Separate the flood zone from location.** Compare floodplain houses with nearby houses outside the floodplain.
- **Go below the area level.** Run the same house by neighborhood for the areas with the widest ranges, like
  CHARLOTTE_7 ($346,000 to $922,000 for the same house).
- **Market prices.** Repeat the area comparison with `market_estimate.market_value` to get today's prices instead
  of assessed values.
