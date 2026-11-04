"""experiments/backtest.py's spec merging and acceptance rule, without data: the runs
themselves need the local feature file."""

from experiments.backtest import BASELINE_SPEC, full_spec, verdict


def test_a_spec_changes_only_what_it_names():
    spec = full_spec({"params": {"learning_rate": 0.1}})
    assert spec["params"] == {**BASELINE_SPEC["params"], "learning_rate": 0.1}
    assert spec["feature_set"] == BASELINE_SPEC["feature_set"]
    # another model's params replace the baseline's instead of merging into them
    assert full_spec({"model": "lightgbm", "params": {"n_estimators": 10}})["params"] == {"n_estimators": 10}
    assert full_spec() == BASELINE_SPEC


def scores(*maes):
    return {"mean_mae": sum(maes) / len(maes), "cutoffs": {str(i): {"mae": m} for i, m in enumerate(maes)}}


def test_a_change_needs_a_2pct_lower_mean_and_no_cutoff_5pct_worse():
    baseline = scores(100, 100)
    assert verdict(scores(97, 98), baseline)["accepted"]  # -2.5%
    assert not verdict(scores(99, 98), baseline)["accepted"]  # -1.5%
    assert not verdict(scores(90, 105.5), baseline)["accepted"]  # -2.25%, but one cutoff +5.5%
    assert verdict({"test_mae": 1, "cutoffs": None}, baseline) is None  # a screen
