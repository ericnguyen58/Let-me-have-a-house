import os

import pytest

from src.pipeline.run import Stage, run_pipeline, stages
from src.utils.load_config import load_config


@pytest.fixture
def chain(tmp_path):
    """raw (placed by hand) -> a -> b, where each stage writes its output and logs that it ran."""
    raw, a, b = (str(tmp_path / name) for name in ["raw.csv", "a.csv", "b.csv"])
    ran = []

    def writer(name, path):
        def run():
            ran.append(name)
            with open(path, "w") as f:
                f.write(name)

        return run

    open(raw, "w").close()
    pipeline = [Stage("a", writer("a", a), [raw], [a]), Stage("b", writer("b", b), [a], [b])]
    return pipeline, ran, {"raw": raw, "a": a, "b": b}


def _age(path, seconds):
    os.utime(path, (os.path.getmtime(path) - seconds,) * 2)


def quiet(*args):
    pass


def test_runs_missing_outputs_then_skips_them(chain):
    pipeline, ran, _ = chain
    run_pipeline(pipeline, log=quiet)
    assert ran == ["a", "b"]
    plan = run_pipeline(pipeline, log=quiet)
    assert ran == ["a", "b"]
    assert [action for _, action, _ in plan] == ["skip", "skip"]


def test_a_newer_input_reruns_its_stage_and_what_follows(chain):
    pipeline, ran, files = chain
    run_pipeline(pipeline, log=quiet)
    for name, seconds in [("raw", 300), ("a", 200), ("b", 100)]:
        _age(files[name], seconds)
    run_pipeline(pipeline, log=quiet)
    assert ran == ["a", "b"]  # still up to date
    os.utime(files["raw"])  # the hand-placed file is replaced
    run_pipeline(pipeline, log=quiet)
    assert ran == ["a", "b", "a", "b"]


def test_from_forces_its_stage_and_skips_earlier_ones(chain):
    pipeline, ran, files = chain
    run_pipeline(pipeline, log=quiet)
    plan = run_pipeline(pipeline, start="b", log=quiet)
    assert ran == ["a", "b", "b"]
    assert plan == [("b", "run", "forced")]


def test_dry_run_runs_nothing_and_follows_upstream_changes(chain):
    pipeline, ran, files = chain
    run_pipeline(pipeline, log=quiet)
    plan = run_pipeline(pipeline, start="a", dry_run=True, log=quiet)
    assert ran == ["a", "b"]
    assert [(name, action) for name, action, _ in plan] == [("a", "run"), ("b", "run")]
    assert "a runs first" in plan[1][2]


def test_a_missing_hand_placed_input_stops_the_run(chain):
    pipeline, ran, files = chain
    os.remove(files["raw"])
    with pytest.raises(SystemExit, match="placed by hand"):
        run_pipeline(pipeline, log=quiet)
    assert ran == []


def test_a_missing_input_names_the_stage_that_makes_it(chain):
    pipeline, ran, _ = chain
    with pytest.raises(SystemExit, match="run the a stage first"):
        run_pipeline(pipeline, start="b", log=quiet)


def test_unknown_or_reversed_stages_are_rejected(chain):
    pipeline, _, _ = chain
    with pytest.raises(SystemExit, match="unknown stage"):
        run_pipeline(pipeline, start="c", log=quiet)
    with pytest.raises(SystemExit, match="comes after"):
        run_pipeline(pipeline, start="b", end="a", log=quiet)


def test_every_real_input_is_made_by_an_earlier_stage_or_placed_by_hand():
    made = set()
    for stage in stages(load_config()):
        for path in stage.inputs:
            assert path in made or path.endswith("merklenburg/Cama_Table_transformed.csv"), (stage.name, path)
        made.update(stage.outputs, stage.may_write)
