"""Score a change to the model behind the market estimate, the same way for every
experiment, and compare it with experiments/baseline.json.

An experiment is a model spec run on the code of the branch it is run from (so a branch
that adds features or a model family is scored with them):

    {"model": a training.build_search_estimators name, "params": its settings,
     "feature_set": a training.FEATURE_SETS name}

`--spec` gives only what changes from BASELINE_SPEC, the production run's model
(2026-11-01_155733). Its params are merged into the baseline's for the same model and
replace them for another model.

Two scores:
- test_mae: MAE in dollars of the 2023 assessed value (total_value) on training.py's test
  split, from one fit on its training split - the model's own out-of-sample error (the
  production run's: $29,439). The screen: one fit, a few minutes.
- the chain backtest (market_estimate.multi_cutoff_backtest, windows ending on END): at
  each of the 9 quarterly cutoffs the model is refit without the houses that sold from
  42 months before it on, and the chain (model x calibrated neighborhood ratio, else x
  Zillow) predicts every sale up to the next cutoff. A cutoff's score is the chain's MAE
  in dollars on those sales; mean_mae is the mean of the 9. Ten fits, about half an hour
  on a free machine.

A change is accepted only if its mean_mae is at least MIN_IMPROVEMENT below the
baseline's and no cutoff's MAE is more than MAX_CUTOFF_INCREASE above the baseline's at
that cutoff.

    uv run python -m experiments.backtest baseline                 # writes experiments/baseline.json
    uv run python -m experiments.backtest run NAME [--spec JSON] [--quick] [--note TEXT]
    uv run python -m experiments.backtest compare [FILE.jsonl ...]  # the leaderboard

`run` appends one line to experiments/<branch>.jsonl. Nothing here writes to
output/models/ or reports/. Set OMP_NUM_THREADS when several runs share the machine.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.components import market_estimate, training
from src.components.finetune import build_model
from src.utils.common import resolve_path
from src.utils.load_config import load_config

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(EXPERIMENTS_DIR, "baseline.json")
# The production run's best model: reports/best_hyperparameters.json of 2026-11-01_155733
BASELINE_SPEC = {
    "model": "hist_gradient_boosting",
    "params": {
        "min_samples_leaf": 10,
        "max_leaf_nodes": 255,
        "max_features": 0.5,
        "learning_rate": 0.05,
        "l2_regularization": 0.1,
    },
    "feature_set": "with_neighborhood_price",
}
# The last day of the last window, fixed so later runs score the same sales (the feature
# file is from November 1, 2026).
END = "2026-11-01"
MIN_IMPROVEMENT = 0.02
MAX_CUTOFF_INCREASE = 0.05


def full_spec(changes=None):
    changes = changes or {}
    spec = {**BASELINE_SPEC, **changes}
    if spec["model"] == BASELINE_SPEC["model"]:
        spec["params"] = {**BASELINE_SPEC["params"], **changes.get("params", {})}
    return spec


def test_scores(spec):
    """One fit on training.py's split: MAE and MAPE of total_value on its test rows."""
    cfg = load_config()
    df = training.load_training_data(resolve_path(cfg, "features_data"))
    train, test = training.split_with_aggregates(df, cfg["training"])
    features = training.FEATURE_SETS[spec["feature_set"]]
    preprocessor = training.build_preprocessor(features)
    model = build_model(spec["model"], spec["params"], cfg["training"])
    model.fit(preprocessor.fit_transform(train[features]), np.log1p(train[training.TARGET]))
    predicted = np.expm1(model.predict(preprocessor.transform(test[features])))
    return {
        "test_mae": mean_absolute_error(test[training.TARGET], predicted),
        "test_mape": mean_absolute_percentage_error(test[training.TARGET], predicted),
    }


def backtest_scores(spec):
    """The chain's MAE and median % error on all sales at each cutoff, and their mean."""
    _, _, predictions = market_estimate.multi_cutoff_backtest(end=END, spec=spec)
    cutoffs = {}
    for cutoff, window in predictions.groupby("cutoff"):
        window = window[window["chain"].notna()]
        miss = (window["chain"] - window["sale_price"]).abs()
        cutoffs[str(cutoff)] = {
            "mae": miss.mean(),
            "median_pct_error": (miss / window["sale_price"]).median(),
            "sales": len(window),
        }
    return {"mean_mae": float(np.mean([c["mae"] for c in cutoffs.values()])), "cutoffs": cutoffs}


def verdict(result, baseline):
    """The mean MAE change, the worst cutoff's change, and whether the change is accepted."""
    if result.get("cutoffs") is None or baseline is None:
        return None
    change = result["mean_mae"] / baseline["mean_mae"] - 1
    worst = max(result["cutoffs"][c]["mae"] / b["mae"] - 1 for c, b in baseline["cutoffs"].items())
    return {
        "mean_mae_change": change,
        "worst_cutoff_change": worst,
        "accepted": change <= -MIN_IMPROVEMENT and worst <= MAX_CUTOFF_INCREASE,
    }


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=EXPERIMENTS_DIR).stdout.strip()


def load_baseline():
    if not os.path.exists(BASELINE_FILE):
        return None
    with open(BASELINE_FILE) as f:
        return json.load(f)


def score(name, spec, quick=False, note=None):
    started = time.time()
    result = {
        "name": name,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        # the experiment logs themselves don't count
        "dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no", "--", ":/", ":(top,exclude)experiments/*.json*")
        ),
        "started": datetime.now().isoformat(timespec="seconds"),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "note": note,
        "spec": spec,
    }
    print(f"{name}: test split", flush=True)
    result.update(test_scores(spec))
    print(f"{name}: test MAE ${result['test_mae']:,.0f}", flush=True)
    result.update({"mean_mae": None, "cutoffs": None} if quick else backtest_scores(spec))
    result["seconds"] = round(time.time() - started)
    result["verdict"] = verdict(result, load_baseline())
    return result


def run(name, changes=None, quick=False, note=None):
    result = score(name, full_spec(changes), quick, note)
    path = os.path.join(EXPERIMENTS_DIR, f"{result['branch']}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps({k: result[k] for k in ["name", "test_mae", "mean_mae", "verdict", "seconds"]}, indent=2))
    print(f"appended to {path}")
    return result


def leaderboard(files=None):
    """Every experiment line against the baseline, best mean MAE first, then the screens
    by test MAE."""
    baseline = load_baseline()
    rows = []
    for path in files or sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if baseline is not None:
        rows.insert(0, {**baseline, "name": "baseline"})

    def row(r):
        v = verdict(r, baseline) or {}
        return {
            "name": r["name"],
            "branch": r.get("branch"),
            "test_mae": r["test_mae"],
            "test_change": r["test_mae"] / baseline["test_mae"] - 1 if baseline else np.nan,
            "mean_mae": r.get("mean_mae"),
            "mean_change": v.get("mean_mae_change"),
            "worst_cutoff": v.get("worst_cutoff_change"),
            "accepted": v.get("accepted"),
        }

    table = pd.DataFrame([row(r) for r in rows])
    return table.sort_values(["mean_mae", "test_mae"], na_position="last").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline")
    one = sub.add_parser("run")
    one.add_argument("name")
    one.add_argument("--spec", type=json.loads, default=None, help="changes from BASELINE_SPEC, as JSON")
    one.add_argument("--quick", action="store_true", help="test_mae only")
    one.add_argument("--note")
    compare = sub.add_parser("compare")
    compare.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    if args.command == "baseline":
        result = score("baseline", full_spec())
        result["verdict"] = None
        with open(BASELINE_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {BASELINE_FILE}: test MAE ${result['test_mae']:,.0f}, mean MAE ${result['mean_mae']:,.0f}")
    elif args.command == "run":
        run(args.name, args.spec, args.quick, args.note)
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(
                leaderboard(args.files or None).to_string(
                    float_format=lambda x: f"{x:,.4f}" if abs(x) < 1 else f"{x:,.0f}"
                )
            )


if __name__ == "__main__":
    sys.exit(main())
