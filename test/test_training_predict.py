"""Train on the synthetic table with training.py's own code, save the artifacts, and load
them back through predict.py, as the real pipeline does."""

import sys

import numpy as np
import pytest

from src.components import predict, training


@pytest.fixture(scope="module")
def training_run(module_features, tmp_path_factory):
    """Train once for the module: (results, test rows, root directory of the saved models)."""
    root = tmp_path_factory.mktemp("run")
    cfg = {"test_size": 0.25, "random_state": 0, "search_iter": 2, "cv_folds": 2}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(training, "test_address_parcels", lambda: [])  # those need the real file
        features = module_features.drop(columns=training.SALE_DERIVED_FEATURES)
        train, test = training.split_with_aggregates(features, cfg)
    results, models, preprocessors, _ = training.train_models(train, test, cfg)
    training.save_artifacts(str(root / "models"), preprocessors, models)
    return results, test, root


@pytest.fixture
def trained(training_run, monkeypatch):
    results, test, root = training_run
    monkeypatch.setattr(predict, "load_config", lambda: {"paths": {"root": f"{root}/", "models": "models/"}})
    return results, test


def test_models_learn_the_signal(trained):
    results, _ = trained
    assert len(results) == len(training.FEATURE_SETS) * len(training.MODEL_TYPES)
    assert all(metrics["r2"] > 0.8 for metrics in results.values())


def test_saved_models_predict_through_predict_py(trained):
    _, test = trained
    house = test.iloc[0].to_dict()
    for feature_set in training.FEATURE_SETS:
        estimate = predict.predict_total_value(house, feature_set=feature_set)
        assert estimate == pytest.approx(house["total_value"], rel=0.25)


def test_missing_features_are_imputed(trained):
    estimate = predict.predict_total_value({"area": "MATTHEWS", "heated_area": 2000})
    assert np.isfinite(estimate) and estimate > 0


def test_unknown_feature_set_or_model_is_rejected():
    with pytest.raises(ValueError, match="unknown feature_set"):
        predict.load_model("everything")
    with pytest.raises(ValueError, match="not in use"):
        predict.load_model(model_name="random_forest")


def test_dollar_mae_reads_log_values():
    assert training.dollar_mae(np.log1p([100.0, 200.0]), np.log1p([110.0, 180.0])) == pytest.approx(15)


def test_only_the_model_types_in_use_are_built():
    searches = training.build_search_estimators(0, {"search_iter": 1, "cv_folds": 2})
    assert list(searches) == training.MODEL_TYPES
    assert "xgboost" not in sys.modules  # an optional dependency, imported only when asked for
