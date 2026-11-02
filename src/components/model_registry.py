"""Training runs and the production model.

Every training run (training.py, finetune.py) is saved to its own folder,
output/models/runs/{run_id}/, and is never overwritten:

- {feature_set}_preprocessor.joblib, {feature_set}_{model_name}.joblib
- training_metrics.csv, best_hyperparameters.json
- held_out_parcels.csv: parcels with no row in the run's training split
- run.json: when and from what it was made (git commit, the feature file's sha256 and
  row count, package versions), its test metrics, and whether it was promoted and why

output/models/production.json names the production run: the one predict.py (and with it
search.py, the assistant and the API) loads. Promoting a run points production.json at
it and copies its training_metrics.csv and best_hyperparameters.json to reports/, where
finetune.py, evaluating.py and the calibrate stage read them. A new run is promoted only
if it passes training.check_and_promote's comparison with the production run.
output/models/last_run.json names the newest run, promoted or not.

    uv run python -m src.components.model_registry                 # list runs
    uv run python -m src.components.model_registry promote RUN_ID  # promote by hand, without the check
    uv run python -m src.components.model_registry rollback        # back to the run production replaced
    uv run python -m src.components.model_registry adopt           # one time: record output/models/*.joblib as a run

A process that has loaded a model keeps it: restart the API after a promotion.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from importlib import metadata

import joblib
import pandas as pd

from src.utils.common import resolve_path
from src.utils.load_config import PROJECT_PATH, load_config

RUNS_DIR = "runs/"
PRODUCTION_FILE = "production.json"
LAST_RUN_FILE = "last_run.json"
RUN_FILE = "run.json"
HELD_OUT_FILE = "held_out_parcels.csv"
# copied to reports/ on promotion
REPORT_FILES = ["training_metrics.csv", "best_hyperparameters.json"]
PACKAGES = ["scikit-learn", "pandas", "numpy", "joblib"]


def run_path(models_path, run_id):
    return os.path.join(models_path, RUNS_DIR, run_id) + "/"


def new_run(models_path, base_run=None, started=None):
    """(run_id, folder) of a new, empty run, named by the time it starts (`started`, a
    datetime, or now). With `base_run`, the folder starts as a copy of that run's models
    and reports, for a run that replaces only some of them (finetune.py)."""
    run_id = name = (started or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    n = 1
    while os.path.exists(run_path(models_path, run_id)):
        n += 1
        run_id = f"{name}_{n}"
    path = run_path(models_path, run_id)
    os.makedirs(path)
    if base_run is not None:
        source = run_path(models_path, base_run)
        for name in os.listdir(source):
            if name.endswith(".joblib") or name in REPORT_FILES:
                shutil.copy2(source + name, path + name)
    return run_id, path


def _git():
    """The commit the code was at, and whether tracked files had uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_PATH, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_PATH,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except OSError, subprocess.CalledProcessError:  # no git, or not a repository (the Docker image)
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def file_fingerprint(path):
    """sha256, size and modification time of a data file, or None when it doesn't exist."""
    if path is None or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": os.path.relpath(path, PROJECT_PATH),
        "sha256": digest.hexdigest(),
        "bytes": os.path.getsize(path),
        "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
    }


def environment():
    return {"python": platform.python_version(), **{p: metadata.version(p) for p in PACKAGES}}


def run_info(run_id, features_file=None, source="training", base_run=None, created=None):
    """The start of a run's run.json: what it was made from and with."""
    return {
        "run_id": run_id,
        "created": (created or datetime.now()).isoformat(timespec="seconds"),
        "source": source,
        "base_run": base_run,
        "git": _git(),
        "data": file_fingerprint(features_file),
        "environment": environment(),
    }


def write_json(path, data):
    """Write through a temporary file, so a reader never sees half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def read_info(models_path, run_id):
    with open(run_path(models_path, run_id) + RUN_FILE) as f:
        return json.load(f)


def write_info(models_path, run_id, info):
    write_json(run_path(models_path, run_id) + RUN_FILE, info)


def held_out(train, test):
    """Parcels of `test` with no row in `train` (a parcel can have more than one row)."""
    return set(test["parcel_id"]) - set(train["parcel_id"])


def save_held_out(path, parcels):
    """Write the run's held-out parcels and return how many; None (unknown) writes nothing."""
    if parcels is None:
        return None
    pd.Series(sorted(parcels), name="parcel_id").to_frame().to_csv(path + HELD_OUT_FILE, index=False)
    return len(parcels)


def held_out_parcels(models_path, run_id):
    """The run's held-out parcels as a set, or None when the run has no record of them."""
    path = run_path(models_path, run_id) + HELD_OUT_FILE
    if not os.path.exists(path):
        return None
    return set(pd.read_csv(path)["parcel_id"])


def production_run(models_path):
    """The production run's id, or None before any run has been promoted."""
    path = os.path.join(models_path, PRODUCTION_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["run_id"]


def production_path(models_path):
    """Folder of the production run's models."""
    run_id = production_run(models_path)
    if run_id is None:
        raise FileNotFoundError(
            f"no production model: {os.path.join(models_path, PRODUCTION_FILE)} doesn't exist. Train one with "
            "`python -m src.components.training`, or record models saved before runs were kept with "
            "`python -m src.components.model_registry adopt`"
        )
    return run_path(models_path, run_id)


def promote(models_path, reports_path, run_id, reason):
    """Make `run_id` the production run and copy its reports to reports/."""
    path = run_path(models_path, run_id)
    if not os.path.exists(path + RUN_FILE):
        raise SystemExit(f"no run {run_id!r} in {os.path.join(models_path, RUNS_DIR)}")
    os.makedirs(reports_path, exist_ok=True)
    for name in REPORT_FILES:
        if os.path.exists(path + name):
            shutil.copy2(path + name, os.path.join(reports_path, name))
    current = production_run(models_path)
    write_json(
        os.path.join(models_path, PRODUCTION_FILE),
        {
            "run_id": run_id,
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "previous": current if current != run_id else _previous(models_path),
        },
    )


def _previous(models_path):
    with open(os.path.join(models_path, PRODUCTION_FILE)) as f:
        return json.load(f).get("previous")


def rollback(models_path, reports_path):
    """Promote the run the production run replaced. Rolling back twice returns to where
    it started; `promote` goes to any run."""
    current = production_run(models_path)
    previous = _previous(models_path) if current else None
    if previous is None:
        raise SystemExit("no earlier production run to roll back to")
    promote(models_path, reports_path, previous, f"rollback from {current}")
    return previous


def record_check(models_path, run_id, promoted, reason):
    """Store the promotion decision in the run's run.json and in last_run.json."""
    info = read_info(models_path, run_id)
    info["check"] = {"promoted": promoted, "reason": reason}
    write_info(models_path, run_id, info)
    write_json(os.path.join(models_path, LAST_RUN_FILE), {"run_id": run_id, **info["check"]})


def list_runs(models_path):
    """One row per run, oldest first, with each model's test MAE (on that run's own test
    split, so runs on different data aren't directly comparable)."""
    runs_dir = os.path.join(models_path, RUNS_DIR)
    production = production_run(models_path)
    rows = []
    for run_id in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        if not os.path.exists(run_path(models_path, run_id) + RUN_FILE):
            continue
        info = read_info(models_path, run_id)
        git = info.get("git") or {}
        row = {
            "run_id": run_id,
            "source": info.get("source"),
            "commit": (git.get("commit") or "")[:7] + ("+changes" if git.get("dirty") else ""),
            "rows": (info.get("data") or {}).get("rows"),
            **{f"mae {name}": m["mae"] for name, m in info.get("metrics", {}).items()},
            "promoted": (info.get("check") or {}).get("promoted"),
            "production": "<-" if run_id == production else "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def adopt(cfg=None):
    """Record the models training.py saved to output/models/ before runs were kept as a
    run, and make it production. The held-out parcels are recomputed from the feature
    file when it is older than the models (so it is the file they were trained on);
    otherwise they are left out, and the first trained run is promoted by hand."""
    from src.components import training

    cfg = cfg or load_config()
    models_path, reports_path = resolve_path(cfg, "models"), resolve_path(cfg, "reports")
    if production_run(models_path) is not None:
        raise SystemExit(f"{PRODUCTION_FILE} exists already: runs are being recorded")
    names = [f"{fs}_preprocessor.joblib" for fs in training.FEATURE_SETS] + [
        f"{fs}_{m}.joblib" for fs in training.FEATURE_SETS for m in training.MODEL_TYPES
    ]
    missing = [n for n in names if not os.path.exists(models_path + n)]
    if missing:
        raise SystemExit(f"nothing to adopt: {', '.join(missing)} missing from {models_path}")
    trained = max(os.path.getmtime(models_path + n) for n in names)

    run_id, path = new_run(models_path, started=datetime.fromtimestamp(trained))
    for name in names:
        shutil.copy2(models_path + name, path + name)
    for name in REPORT_FILES:
        if os.path.exists(reports_path + name):
            shutil.copy2(reports_path + name, path + name)

    features_file = resolve_path(cfg, "features_data") + "properties_features.csv"
    info = run_info(run_id, features_file, source="adopted", created=datetime.fromtimestamp(trained))
    data = info["data"]
    info.update(
        git={"commit": None, "dirty": None},  # the code they were trained with wasn't recorded
        training=cfg["training"],
        feature_sets={
            fs: list(joblib.load(f"{path}{fs}_preprocessor.joblib").feature_names_in_) for fs in training.FEATURE_SETS
        },
        model_types=training.MODEL_TYPES,
        metrics={},
    )
    if os.path.exists(path + "training_metrics.csv"):
        metrics = pd.read_csv(path + "training_metrics.csv")
        info["metrics"] = {
            f"{r.feature_set}/{r.model}": {k: float(r[k]) for k in ["rmse", "mae", "r2", "mape"]}
            for _, r in metrics.iterrows()
        }
    if data is not None and os.path.getmtime(features_file) < trained:
        df = training.load_training_data(resolve_path(cfg, "features_data"))
        train, test = training.split_rows(df, cfg["training"])
        info["data"]["rows"], info["train_rows"], info["test_rows"] = len(df), len(train), len(test)
        info["held_out_parcels"] = save_held_out(path, held_out(train, test))
        note = "held-out parcels recomputed from the feature file, unchanged since training"
    else:
        note = "the feature file changed after training, so the held-out parcels are unknown"
    write_info(models_path, run_id, info)
    reason = f"adopted: the models in {os.path.relpath(models_path, PROJECT_PATH)}/ from before runs were kept; {note}"
    promote(models_path, reports_path, run_id, reason)
    record_check(models_path, run_id, True, reason)
    # dated to when the models were trained, so the pipeline doesn't take the adoption
    # for a new model and rerun what follows train
    for name in [PRODUCTION_FILE, LAST_RUN_FILE]:
        os.utime(os.path.join(models_path, name), (trained, trained))
    return run_id, reason


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.components.model_registry", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="list runs (the default)")
    promote_parser = sub.add_parser("promote", help="make a run production, without the check")
    promote_parser.add_argument("run_id")
    sub.add_parser("rollback", help="back to the run production replaced")
    sub.add_parser("adopt", help="record output/models/*.joblib, saved before runs were kept, as a run")
    args = parser.parse_args(argv)

    cfg = load_config()
    models_path, reports_path = resolve_path(cfg, "models"), resolve_path(cfg, "reports")
    if args.command == "promote":
        promote(models_path, reports_path, args.run_id, "promoted by hand")
        print(f"production is now {args.run_id}")
    elif args.command == "rollback":
        print(f"production is now {rollback(models_path, reports_path)}")
    elif args.command == "adopt":
        run_id, reason = adopt(cfg)
        print(f"production is now {run_id}: {reason}")
    else:
        runs = list_runs(models_path)
        print(runs.to_string(index=False, float_format=lambda x: f"{x:,.0f}") if len(runs) else "no runs yet")


if __name__ == "__main__":
    main()
