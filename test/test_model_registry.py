"""Runs, the production model and the check that decides promotion (model_registry.py,
training.save_run / check_and_promote), on models trained on the synthetic table."""

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.components import model_registry, predict, training

CFG = {"test_size": 0.25, "random_state": 0, "search_iter": 2, "cv_folds": 2}  # as in synthetic_training


@pytest.fixture(scope="module")
def model_sets(synthetic_training):
    """(train, test, {"good": the trained models, "bad": models that predict every house at
    the median value})."""
    _, train, test, (results, models, preprocessors, params) = synthetic_training
    bad_models = {
        key: DummyRegressor(strategy="median").fit(train[[]], np.log1p(train["total_value"])) for key in models
    }
    bad_results = {
        key: training.eval_metrics(test["total_value"], np.expm1(m.predict(test[[]]))) for key, m in bad_models.items()
    }
    return (
        train,
        test,
        {"good": (results, models, preprocessors, params), "bad": (bad_results, bad_models, preprocessors, params)},
    )


@pytest.fixture
def registry(tmp_path, model_sets, monkeypatch):
    """Save runs into a temporary models folder: save(kind, test=None, **info) -> run_id.
    The shared-house minimum is lowered to fit the synthetic table's 150 test houses."""
    monkeypatch.setattr(training, "MIN_SHARED_HOUSES", 50)
    models_path, reports_path = f"{tmp_path}/models/", f"{tmp_path}/reports/"
    train, test, sets = model_sets

    def save(kind, test_rows=None, **info):
        results, models, preprocessors, params = sets[kind]
        rows = test if test_rows is None else test_rows
        metrics = training.metrics_table(results)
        return training.save_run(models_path, train, rows, preprocessors, models, metrics, params, CFG, **info)

    def check(run_id, test_rows=None):
        return training.check_and_promote(models_path, reports_path, run_id, test if test_rows is None else test_rows)

    return models_path, reports_path, save, check


def test_the_first_run_is_promoted_and_predict_loads_it(registry, model_sets, monkeypatch, tmp_path):
    models_path, reports_path, save, check = registry
    run_id = save("good")
    promoted, reason = check(run_id)
    assert promoted and "first run" in reason
    assert model_registry.production_run(models_path) == run_id
    assert os.path.exists(reports_path + "training_metrics.csv")  # production's reports go to reports/

    monkeypatch.setattr(predict, "load_config", lambda: {"paths": {"root": f"{tmp_path}/", "models": "models/"}})
    house = model_sets[1].iloc[0].to_dict()
    assert predict.predict_total_value(house) == pytest.approx(house["total_value"], rel=0.3)


def test_a_worse_run_is_kept_but_not_promoted(registry):
    models_path, reports_path, save, check = registry
    good = save("good")
    check(good)
    good_report = open(reports_path + "training_metrics.csv").read()

    bad = save("bad")
    promoted, reason = check(bad)
    assert not promoted and "more than 1% worse" in reason
    assert model_registry.production_run(models_path) == good
    assert open(reports_path + "training_metrics.csv").read() == good_report
    assert model_registry.read_info(models_path, bad)["check"]["promoted"] is False
    with open(models_path + model_registry.LAST_RUN_FILE) as f:
        assert json.load(f)["run_id"] == bad  # what the pipeline's train stage is judged by


def test_an_equally_good_run_is_promoted_and_rollback_returns_to_the_one_before(registry):
    models_path, reports_path, save, check = registry
    first = save("good")
    check(first)
    second = save("good")
    assert second != first
    promoted, reason = check(second)
    assert promoted and "(+0.00%)" in reason
    assert model_registry.rollback(models_path, reports_path) == first
    assert model_registry.production_run(models_path) == first
    assert model_registry.rollback(models_path, reports_path) == second  # and back


def test_without_enough_shared_held_out_houses_the_decision_is_left_to_a_person(registry, model_sets):
    models_path, _, save, check = registry
    check(save("good"))
    test = model_sets[1]
    elsewhere = test.assign(parcel_id=test["parcel_id"] + 10_000)  # none of production's held-out houses
    promoted, reason = check(save("good", elsewhere), elsewhere)
    assert not promoted and "compare by hand" in reason


def test_held_out_parcels_leave_out_parcels_with_a_row_in_training():
    train = pd.DataFrame({"parcel_id": [1, 2, 3]})
    test = pd.DataFrame({"parcel_id": [3, 4, 5]})  # parcel 3 has a row on each side
    assert model_registry.held_out(train, test) == {4, 5}


def test_a_run_records_what_it_was_made_from(registry, tmp_path):
    models_path, _, save, _ = registry
    features_file = tmp_path / "features.csv"
    features_file.write_text("parcel_id\n1\n")
    info = model_registry.read_info(models_path, save("good", features_file=str(features_file)))
    assert set(info["git"]) == {"commit", "dirty"}
    assert len(info["data"]["sha256"]) == 64
    assert info["environment"]["scikit-learn"]
    assert set(info["feature_sets"]) == set(training.FEATURE_SETS)
    assert set(info["metrics"]) == {f"{fs}/{m}" for fs in training.FEATURE_SETS for m in training.MODEL_TYPES}
    assert info["held_out_parcels"] > 0


def test_a_run_based_on_another_copies_its_models_and_keeps_only_parcels_both_held_out(registry, model_sets):
    models_path, _, save, _ = registry
    base = save("good")
    test = model_sets[1]
    later = save("good", test.iloc[: len(test) // 2], base_run=base)
    names = sorted(n for n in os.listdir(model_registry.run_path(models_path, later)) if n.endswith(".joblib"))
    assert names == sorted(n for n in os.listdir(model_registry.run_path(models_path, base)) if n.endswith(".joblib"))
    assert model_registry.held_out_parcels(models_path, later) < model_registry.held_out_parcels(models_path, base)


def test_without_a_production_run_loading_a_model_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="no production model"):
        model_registry.production_path(f"{tmp_path}/")


def test_list_runs_marks_production(registry):
    models_path, _, save, check = registry
    first = save("good")
    check(first)
    check(save("bad"))
    runs = model_registry.list_runs(models_path)
    assert runs["run_id"].tolist()[0] == first
    assert runs.set_index("run_id")["production"].to_dict()[first] == "<-"
    assert runs["promoted"].tolist() == [True, False]


def test_adopt_records_models_saved_before_runs_were_kept(tmp_path, model_sets, monkeypatch):
    train, test, sets = model_sets
    monkeypatch.setattr(training, "test_address_parcels", lambda: [])
    cfg = {
        "paths": {"root": f"{tmp_path}/", "models": "models/", "reports": "reports/", "features_data": "features/"},
        "training": CFG,
    }
    os.makedirs(tmp_path / "features")
    pd.concat([train, test]).to_csv(tmp_path / "features" / "properties_features.csv", index=False)
    _, models, preprocessors, params = sets["good"]
    training.save_artifacts(f"{tmp_path}/models/", preprocessors, models)  # the old layout, written after the data
    trained = os.path.getmtime(tmp_path / "features" / "properties_features.csv") + 60
    for name in os.listdir(tmp_path / "models"):
        os.utime(tmp_path / "models" / name, (trained, trained))

    run_id, reason = model_registry.adopt(cfg)
    models_path = f"{tmp_path}/models/"
    assert model_registry.production_run(models_path) == run_id
    assert "recomputed" in reason and model_registry.held_out_parcels(models_path, run_id)
    # dated like the models, so the pipeline doesn't rerun the stages after train
    assert os.path.getmtime(models_path + model_registry.PRODUCTION_FILE) == trained
    with pytest.raises(SystemExit, match="exists already"):
        model_registry.adopt(cfg)
