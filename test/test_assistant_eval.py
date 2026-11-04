"""The assistant eval's grading (evals/assistant.py), without a model: the eval itself
needs one and the local data, so it runs by hand, but its checks are tested here."""

import json

import pytest
from anthropic import beta_tool

from evals.assistant import grade, has_count, load_cases, matches, money_amounts, numbers_in, recorded_tool_calls
from src.components import assistant


@pytest.mark.parametrize(
    "written, value, ok",
    [
        ("$1,529,788", 1_529_788.33, True),
        ("about $1.53 million", 1_529_788.33, True),  # rounded
        ("$1.52 million", 1_529_788.33, True),  # cut off
        ("$1.5M", 1_529_788.33, True),
        ("$1.4 million", 1_529_788.33, False),
        ("$1,530,000", 1_529_788.33, True),  # trailing zeros as rounding
        ("$500,000", 509_000, True),
        ("$500,000", 530_000, False),  # rounding to two significant digits at most
        ("$870k", 870_545, True),
        ("$5,446 less", -5_446.2, True),  # the sign is in the words
        ("$235 per sq ft", 235.0, True),
    ],
)
def test_a_written_figure_matches_the_value_it_rounds(written, value, ok):
    (amount,) = money_amounts(written)
    assert matches(amount, value) is ok


def test_a_range_written_with_one_unit_applies_it_to_both_ends():
    low, high = money_amounts("from $1.38 to $1.82 million")
    assert matches(low, 1_375_766) and matches(high, 1_821_142)
    assert money_amounts("$1.38 per share, $2 million")[0][0] == 1.38  # not a range
    assert money_amounts("between $822,000 and $1.10 million")[0][0] == 822_000  # a whole figure


def test_counts_are_found_as_digits_or_words_but_not_inside_other_numbers():
    assert has_count("based on 3 recent sales", 3) and has_count("three sales", 3)
    assert not has_count("$3,500 and a ratio of 1.3", 3)


def test_every_number_in_a_tool_output_is_collected():
    output = json.dumps({"a": 1.5, "b": [{"c": 2}, "text", True], "d": None})
    assert sorted(numbers_in(output)) == [1.5, 2.0]
    assert numbers_in("not json: 1,250 and 3.5") == [1250.0, 3.5]


def test_tool_calls_are_recorded_and_the_tool_class_is_restored():
    @beta_tool
    def double(x: int) -> str:
        """Double a number.

        Args:
            x: The number.
        """
        return json.dumps({"y": 2 * x})

    original = type(double).call
    with recorded_tool_calls() as calls:
        double.call({"x": 2})
    assert calls == [{"tool": "double", "args": {"x": 2}, "output": '{"y": 4}'}]
    assert type(double).call is original


CALLS = [
    {"tool": "lookup_property", "args": {"address": "1 Main St"}, "output": json.dumps([{"parcel_id": 123.0}])},
    {
        "tool": "estimate_market_value",
        "args": {"parcel_id": "123"},
        "output": json.dumps({"market_estimate": 1_529_788.33, "range_low": 1_375_766.5, "sales_used": 3}),
    },
]
CASE = {
    "question": "What would 1 Main St sell for?",
    "tools": ["lookup_property", "estimate_market_value"],
    "forbid": ["find_deals"],
    "args": {"estimate_market_value": {"parcel_id": "@parcel"}},
    "facts": [
        {"tool": "estimate_market_value", "field": "market_estimate"},
        {"tool": "estimate_market_value", "field": "range_low"},
        {"tool": "estimate_market_value", "field": "sales_used", "kind": "count"},
    ],
    "mentions": [["sales", "sold"]],
}


def test_a_good_answer_passes_every_check():
    answer = "About $1.53 million (from $1,376,000), based on 3 recent sales."
    checks = grade(CASE, answer, CALLS, parcel=123.0)
    assert {name: ok for name, (ok, _) in checks.items()} == {
        "tools": True,
        "forbid": True,
        "args:estimate_market_value": True,
        "fact:market_estimate": True,
        "fact:range_low": True,
        "fact:sales_used": True,
        "mentions:1": True,
        "grounded": True,
    }


def test_a_bad_answer_fails_the_checks_it_breaks():
    answer = "It is worth $2,000,000."
    checks = grade(CASE, answer, CALLS[:1] + [{**CALLS[1], "args": {"parcel_id": "999"}}], parcel=123.0)
    failed = {name: detail for name, (ok, detail) in checks.items() if not ok}
    assert set(failed) == {
        "args:estimate_market_value",
        "fact:market_estimate",
        "fact:range_low",
        "fact:sales_used",
        "mentions:1",
        "grounded",
    }
    assert failed["grounded"] == "not from a tool: $2,000,000"


def test_a_difference_of_two_tool_figures_is_noted_as_derived_not_failed():
    calls = [{"tool": "area_stats", "args": {}, "output": json.dumps([{"a": 606_900.0}, {"b": 493_300.0}])}]
    ok, detail = grade({"question": "Compare them."}, "Davidson is about $114k higher.", calls)["grounded"]
    assert ok and detail == "derived: $114k = $606,900 - $493,300"


def test_an_off_topic_answer_must_be_the_fixed_reply_without_tools():
    case = {"question": "Weather?", "no_tools": True, "reply": "OFF_TOPIC_REPLY"}
    assert all(ok for ok, _ in grade(case, assistant.OFF_TOPIC_REPLY, []).values())
    checks = grade(case, "It will rain.", CALLS[:1])
    assert not checks["no_tools"][0] and not checks["reply"][0]


def test_the_cases_name_real_tools_and_replies():
    tools = {t.name for t in assistant.TOOLS}
    keys = {"id", "question", "address", "tools", "forbid", "no_tools", "args", "facts", "mentions", "reply", "history"}
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        assert set(case) <= keys, case["id"]
        named = set(case.get("tools", [])) | set(case.get("forbid", [])) | set(case.get("args", {}))
        named |= {f["tool"] for f in case.get("facts", [])}
        assert named <= tools, case["id"]
        assert {f["tool"] for f in case.get("facts", [])} <= set(case.get("tools", [])), case["id"]
        if "reply" in case:
            assert isinstance(getattr(assistant, case["reply"]), str)
        if "@parcel" in json.dumps(case.get("args", {})):
            assert case.get("address"), case["id"]
