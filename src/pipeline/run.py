"""Run the pipeline stages in order, skipping the ones that are up to date.

    uv run python -m src.pipeline.run                 # every stage that is out of date
    uv run python -m src.pipeline.run --dry-run       # what would run, and why, without running it
    uv run python -m src.pipeline.run --from train    # rerun train even if up to date, then what follows
    uv run python -m src.pipeline.run --to features   # stop after features
    uv run python -m src.pipeline.run --force         # rerun every selected stage

A stage is up to date when all its outputs exist and none of its inputs is newer than
its oldest output, the way make decides. Only data files count as inputs, not code: after
changing a stage's code, rerun from that stage with --from. Each stage still runs on its
own with `python -m src.components.<module>`, as before.

The Mecklenburg CSV (data/raw/merklenburg/Cama_Table_transformed.csv) is placed by hand,
so no stage makes it; a missing one stops the run before preprocess.
"""

import argparse
import importlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.utils.common import resolve_path
from src.utils.load_config import load_config


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[], object]
    inputs: list[str]
    outputs: list[str]


def _call(module, function="run"):
    """A stage's run function, imported when the stage runs, so --dry-run stays quick."""
    return lambda: getattr(importlib.import_module(f"src.components.{module}"), function)()


def _ingest():
    from src.components.data_ingestor import data_ingestor

    data_ingestor().ingest()


def stages(cfg):
    """The pipeline's stages in run order, with the files each reads and writes."""
    from src.components.market_estimate import OOF_FILE
    from src.components.training import FEATURE_SETS, MODEL_TYPES

    raw, processed, features, models, reports = (
        resolve_path(cfg, key) for key in ["raw_data", "processed_data", "features_data", "models", "reports"]
    )
    places = raw + "NC parquet/nc_places.parquet"
    zillow = raw + "zillow/county_zhvi.csv"
    location = [
        *(f"{raw}meck_gis/{layer}.geojson" for layer in cfg["meck_gis"]["layers"]),
        raw + "meck_gis/quality_of_life.csv",
        *(raw + "school_report_cards/" + name for name in cfg["report_cards"]["files"]),
    ]
    clean, places_map = processed + "properties_clean.csv", processed + "nc_places_map.csv"
    feature_table = features + "properties_features.csv"
    model_files = [models + f"{fs}_preprocessor.joblib" for fs in FEATURE_SETS] + [
        models + f"{fs}_{name}.joblib" for fs in FEATURE_SETS for name in MODEL_TYPES
    ]
    training_reports = [reports + "training_metrics.csv", reports + "best_hyperparameters.json"]
    out_of_fold = models + OOF_FILE
    return [
        Stage("ingest", _ingest, [], [places, *location, zillow]),
        Stage(
            "preprocess",
            _call("data_preprocessing"),
            [raw + "merklenburg/Cama_Table_transformed.csv", places],
            [clean, places_map],
        ),
        Stage("features", _call("feature_engineering"), [clean, places_map, *location], [feature_table]),
        Stage("train", _call("training"), [feature_table], model_files + training_reports),
        Stage(
            "calibrate",
            _call("market_estimate", "build_out_of_fold_estimates"),
            [feature_table, *training_reports],
            [out_of_fold],
        ),
        Stage("validate", _call("evaluating"), [feature_table, *model_files], [reports + "validation_houses.csv"]),
        Stage(
            "check_houses",
            _call("market_estimate", "run_houses"),
            [feature_table, *model_files, out_of_fold, zillow],
            [reports + "market_test_houses.csv"],
        ),
    ]


def staleness(stage):
    """Why `stage` needs to run, judged from its files now, or None when it is up to date."""
    missing = [p for p in stage.outputs if not os.path.exists(p)]
    if missing:
        return f"{_short(missing[0])} is missing" + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else "")
    oldest = min(os.path.getmtime(p) for p in stage.outputs)
    newer = [p for p in stage.inputs if os.path.exists(p) and os.path.getmtime(p) > oldest]
    if newer:
        return f"{_short(newer[0])} is newer than its outputs"
    return None


def _short(path):
    root = load_config()["paths"]["root"]
    return path.removeprefix(root)


def select(all_stages, start=None, end=None):
    names = [s.name for s in all_stages]
    for name in [start, end]:
        if name is not None and name not in names:
            raise SystemExit(f"unknown stage {name!r}, expected one of {names}")
    first = names.index(start) if start else 0
    last = names.index(end) if end else len(names) - 1
    if first > last:
        raise SystemExit(f"--from {start} comes after --to {end}")
    return all_stages[first : last + 1]


def run_pipeline(all_stages, start=None, end=None, force=False, dry_run=False, log=print):
    """Run the stages from `start` to `end` (names; default every stage) that are out of
    date, or all of them with `force`; `start` itself always runs. With `dry_run`, only
    report what would run. Returns [(stage name, "run" | "skip", reason)].

    A stage whose input is missing stops the run, naming the stage that makes it (or
    saying it is placed by hand)."""
    selected = select(all_stages, start, end)
    makers = {path: s.name for s in all_stages for path in s.outputs}
    will_change = set()  # outputs of stages that run before this one (dry run only)
    plan = []
    for stage in selected:
        if force or stage.name == start:
            reason = "forced"
        else:
            reason = staleness(stage)
            if dry_run and reason is None:
                upstream = next((p for p in stage.inputs if p in will_change), None)
                reason = upstream and f"{makers[upstream]} runs first and rewrites {_short(upstream)}"
        if reason is None:
            plan.append((stage.name, "skip", "up to date"))
            log(f"[skip] {stage.name}: up to date")
            continue

        missing = [p for p in stage.inputs if not os.path.exists(p) and p not in will_change]
        if missing:
            maker = makers.get(missing[0])
            hint = f"run the {maker} stage first" if maker else "it is placed by hand (see README)"
            raise SystemExit(f"{stage.name}: missing input {_short(missing[0])} - {hint}")

        plan.append((stage.name, "run", reason))
        log(f"[run]  {stage.name}: {reason}")
        if dry_run:
            will_change.update(stage.outputs)
            continue
        started = time.monotonic()
        stage.run()
        log(f"[done] {stage.name} in {time.monotonic() - started:,.0f}s")
    return plan


def main(argv=None):
    all_stages = stages(load_config())
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.run", description=__doc__.split("\n")[0])
    names = [s.name for s in all_stages]
    parser.add_argument("--from", dest="start", choices=names, help="first stage; it runs even if up to date")
    parser.add_argument("--to", dest="end", choices=names, help="last stage")
    parser.add_argument("--force", action="store_true", help="rerun every selected stage")
    parser.add_argument("--dry-run", action="store_true", help="show what would run, and why, without running it")
    args = parser.parse_args(argv)
    return run_pipeline(all_stages, args.start, args.end, args.force, args.dry_run)


if __name__ == "__main__":
    main()
