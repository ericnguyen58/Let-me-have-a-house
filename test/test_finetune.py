"""finetune.py on the synthetic table: it retrains production's best model and saves it as
a run based on the production run, which goes through the same check as a training run."""

import os

import pytest

from src.components import finetune, model_registry, training


@pytest.fixture
def production(synthetic_training, tmp_path, monkeypatch):
    """A production run in a temporary models folder, with finetune.py pointed at it and at
    the synthetic split: (models path, reports path, production run id)."""
    cfg, train, test, (results, models, preprocessors, params) = synthetic_training
    paths = {"root": f"{tmp_path}/", "models": "models/", "reports": "reports/", "features_data": "features/"}
    models_path, reports_path = f"{tmp_path}/models/", f"{tmp_path}/reports/"
    run_id = training.save_run(
        models_path, train, test, preprocessors, models, training.metrics_table(results), params, cfg
    )
    training.check_and_promote(models_path, reports_path, run_id, test)
    monkeypatch.setattr(finetune, "load_config", lambda: {"paths": paths, "training": cfg})
    monkeypatch.setattr(finetune, "load_splits", lambda: (train, test))
    monkeypatch.setattr(training, "MIN_SHARED_HOUSES", 50)  # the synthetic table has 150 test houses
    return models_path, reports_path, run_id


def test_finetune_saves_a_run_based_on_production_and_checks_it(production):
    models_path, reports_path, base = production
    metrics, _ = finetune.run()

    run_id = model_registry.production_run(models_path)
    assert run_id != base  # the same settings on the same split: no worse, so promoted
    info = model_registry.read_info(models_path, run_id)
    assert (info["source"], info["base_run"], info["check"]["promoted"]) == ("finetune", base, True)
    # the same split as production, so the same held-out parcels
    assert model_registry.held_out_parcels(models_path, run_id) == model_registry.held_out_parcels(models_path, base)
    files = os.listdir(model_registry.run_path(models_path, run_id))
    assert sorted(f for f in files if f.endswith(".joblib")) == sorted(
        f for f in os.listdir(model_registry.run_path(models_path, base)) if f.endswith(".joblib")
    )
    assert metrics["change"].abs().max() < 1e-6
    assert os.path.exists(reports_path + "finetuned_metrics.csv")


def test_finetune_needs_a_production_run(tmp_path, monkeypatch):
    paths = {"root": f"{tmp_path}/", "models": "models/", "reports": "reports/"}
    monkeypatch.setattr(finetune, "load_config", lambda: {"paths": paths, "training": {}})
    with pytest.raises(SystemExit, match="no production run"):
        finetune.run()
