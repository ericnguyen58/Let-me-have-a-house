"""Run the assistant on questions with checkable answers, and grade the answers by code.

    uv run python -m evals.assistant                       # every case in evals/assistant_cases.yaml
    uv run python -m evals.assistant --only market_value what_if_bathroom
    uv run python -m evals.assistant --repeat 3            # each case 3 times: answers vary between runs

It runs on ASSISTANT_MODEL, the local Ollama model by default (free, needs `ollama serve`).
A Claude model is billed per question, so it needs --paid. Not part of CI: it needs a
model and the local data.

No second model judges the answers. A case passes when every one of its checks does: the
tools called and their arguments, the numbers the answer gives against what the tools
returned (allowing for rounding as written), the caveats the system prompt asks for, and
that every dollar figure in the answer came from a tool or the question. Results go to
reports/assistant_eval.csv, and with every answer and tool call to reports/assistant_eval.json.
"""

import argparse
import contextlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from anthropic.lib.tools._beta_functions import BetaFunctionTool

from src.components import assistant, search
from src.utils.common import resolve_path
from src.utils.load_config import load_config

CASES_PATH = Path(__file__).with_name("assistant_cases.yaml")
# A tool output kept in the JSON report is cut to this many characters
OUTPUT_CHARS = 3_000
# Slack on top of the rounding a written figure allows, as a share of the figure
EXTRA_TOLERANCE = 0.001

MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?(million|thousand|mn|[mk])\b)?", re.IGNORECASE)
RANGE_JOIN = re.compile(r"\s*(?:to|-|–|—|and)\s*")
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def money_amounts(text):
    """Dollar figures in `text` as (value, tolerance, as written). The tolerance is one unit
    of the last digit given, so "$1.53 million" stands for anything from about $1.52 to
    $1.54 million (rounded or cut off), and trailing zeros count as rounding to two
    significant digits at most: "$1,530,000" is to the $10,000, and so is "$500,000"."""
    found = []
    found_matches = list(MONEY.finditer(text))
    for i, match in enumerate(found_matches):
        digits, unit = match.group(1).rstrip(".,"), (match.group(2) or "").lower()
        following = found_matches[i + 1] if i + 1 < len(found_matches) else None
        if (
            not unit
            and float(digits.replace(",", "")) < 1000  # a bare "$1.38", not "$822,000"
            and following
            and following.group(2)
            and RANGE_JOIN.fullmatch(text[match.end() : following.start()])
        ):
            unit = following.group(2).lower()  # "$1.38 to $1.82 million": the unit is written once
        scale = {"": 1, "k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6}[unit]
        plain = digits.replace(",", "")
        if "." in plain:
            step = 10 ** -len(plain.split(".")[1])
        else:  # trailing zeros as rounding, to two significant digits at most
            zeros = len(plain) - len(plain.rstrip("0")) if plain.strip("0") else 0
            step = 10 ** max(0, min(zeros, len(plain) - 2))
        value = float(plain) * scale
        found.append((value, step * scale + EXTRA_TOLERANCE * value, match.group(0)))
    return found


def matches(amount, value):
    """Whether a written dollar figure (from money_amounts) can stand for `value`; signs
    are ignored, since "-$5,446" is often written "$5,446 less"."""
    written, tolerance, _ = amount
    return abs(written - abs(value)) <= tolerance


def numbers_in(output):
    """Every number in a tool's output (JSON, or else any numbers in the text)."""
    try:
        data = json.loads(output)
    except TypeError, ValueError:
        return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", str(output))]
    found, stack = [], [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            found.append(float(item))
    return found


def has_count(text, n):
    words = re.findall(r"[a-z]+", text.lower())
    return re.search(rf"(?<![\d.,]){n}(?![\d.,]*\d)", text) is not None or (
        0 <= n < len(NUMBER_WORDS) and NUMBER_WORDS[n] in words
    )


@contextlib.contextmanager
def recorded_tool_calls():
    """Record every tool call while open: [{tool, args, output}]. Both the Ollama loop and
    Claude's tool runner run a tool through its .call()."""
    calls, original = [], BetaFunctionTool.call

    def call(self, input):
        output = original(self, input)
        calls.append({"tool": self.name, "args": dict(input or {}), "output": output})
        return output

    BetaFunctionTool.call = call
    try:
        yield calls
    finally:
        BetaFunctionTool.call = original


def _same(expected, actual):
    if expected == "*":
        return actual is not None
    if actual is None:
        return False
    try:
        return math.isclose(float(expected), float(actual), rel_tol=0.005)
    except TypeError, ValueError:
        return str(expected).strip().lower() == str(actual).strip().lower()


def _result(calls, tool):
    """The tool's last output without an error, parsed; a list gives its first element."""
    for call in reversed(calls):
        if call["tool"] != tool:
            continue
        try:
            data = json.loads(call["output"])
        except TypeError, ValueError:
            continue
        if isinstance(data, list):
            data = data[0] if data else None
        if isinstance(data, dict) and "error" not in data:
            return data
    return None


def grade(case, answer, calls, parcel=None):
    """{check: (passed, detail)} for one answer and the tool calls behind it."""
    checks = {}
    called = [c["tool"] for c in calls]
    if case.get("tools"):
        missing = [t for t in case["tools"] if t not in called]
        checks["tools"] = (not missing, f"not called: {', '.join(missing)}" if missing else "")
    if case.get("forbid"):
        used = [t for t in case["forbid"] if t in called]
        checks["forbid"] = (not used, f"called: {', '.join(used)}" if used else "")
    if case.get("no_tools"):
        checks["no_tools"] = (not calls, f"called: {', '.join(called)}" if calls else "")
    for tool, expected in (case.get("args") or {}).items():
        expected = {k: (parcel if v == "@parcel" else v) for k, v in expected.items()}
        good = any(all(_same(v, c["args"].get(k)) for k, v in expected.items()) for c in calls if c["tool"] == tool)
        seen = [c["args"] for c in calls if c["tool"] == tool]
        checks[f"args:{tool}"] = (good, "" if good else f"wanted {expected}, got {seen or 'no call'}")
    amounts = money_amounts(answer)
    for fact in case.get("facts") or []:
        result = _result(calls, fact["tool"])
        value = None if result is None else result.get(fact["field"])
        if value is None:
            checks[f"fact:{fact['field']}"] = (False, f"{fact['tool']} returned no {fact['field']}")
        elif fact.get("kind") == "count":
            n = int(round(float(value)))
            checks[f"fact:{fact['field']}"] = (has_count(answer, n), f"answer doesn't say {n}")
        else:
            good = any(matches(a, float(value)) for a in amounts)
            checks[f"fact:{fact['field']}"] = (good, "" if good else f"no figure for ${float(value):,.0f}")
    for i, group in enumerate(case.get("mentions") or []):
        good = any(phrase.lower() in answer.lower() for phrase in group)
        checks[f"mentions:{i + 1}"] = (good, "" if good else f"none of {group}")
    if case.get("reply"):
        wanted = getattr(assistant, case["reply"])
        good = " ".join(answer.split()) == " ".join(wanted.split())
        checks["reply"] = (good, "" if good else f"not the {case['reply']} sentence")

    # every dollar figure from a tool result or the conversation
    given = [n for c in calls for n in numbers_in(c["output"])]
    said = " ".join([case["question"], *(t["content"] for t in case.get("history") or [])])
    given += [v for v, _, _ in money_amounts(said)]
    # a difference of two of those (one area's median above another's) is worked out, not
    # made up: noted, not failed
    ungrounded, derived = [], []
    for amount in amounts:
        if any(matches(amount, v) for v in given):
            continue
        pair = next(((x, y) for x in given for y in given if x > y and matches(amount, x - y)), None)
        if pair:
            derived.append(f"{amount[2]} = ${pair[0]:,.0f} - ${pair[1]:,.0f}")
        else:
            ungrounded.append(amount[2])
    detail = f"not from a tool: {', '.join(ungrounded)}" if ungrounded else ""
    checks["grounded"] = (not ungrounded, detail or (f"derived: {'; '.join(derived)}" if derived else ""))
    return checks


def load_cases(only=None):
    with open(CASES_PATH) as f:
        cases = yaml.safe_load(f)
    if only:
        unknown = set(only) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"no case {', '.join(sorted(unknown))} in {CASES_PATH.name}")
        cases = [c for c in cases if c["id"] in only]
    return cases


def run_case(case, run=1):
    parcel = None
    if case.get("address"):
        parcel = search.lookup_property(case["address"])[0]["parcel_id"]
    started = time.monotonic()
    with recorded_tool_calls() as calls:
        try:
            answer, error = assistant.ask(case["question"], case.get("history") or []), None
        except Exception as e:  # a failed request fails the case, not the run (SystemExit still stops it)
            answer, error = "", f"{type(e).__name__}: {e}"
    seconds = time.monotonic() - started
    checks = grade(case, answer, calls, parcel)
    if error:
        checks = {"error": (False, error), **checks}
    return {
        "case": case["id"],
        "run": run,
        "passed": all(ok for ok, _ in checks.values()),
        "failed": [f"{name} ({detail})" for name, (ok, detail) in checks.items() if not ok],
        "notes": [f"{name}: {detail}" for name, (ok, detail) in checks.items() if ok and detail],
        "checks": {name: ok for name, (ok, _) in checks.items()},
        "tools": [c["tool"] for c in calls],
        "seconds": round(seconds, 1),
        "question": case["question"],
        "answer": answer,
        "tool_calls": [{**c, "output": str(c["output"])[:OUTPUT_CHARS]} for c in calls],
    }


def summarize(results):
    """Pass counts per case and per check."""
    df = pd.DataFrame(results)
    by_case = df.groupby("case", sort=False)["passed"].agg(["sum", "count"])
    checks = pd.DataFrame(list(df["checks"])).melt(var_name="check", value_name="passed").dropna()
    checks["check"], checks["passed"] = checks["check"].str.split(":").str[0], checks["passed"].astype(int)
    by_check = checks.groupby("check")["passed"].agg(["sum", "count"])
    return by_case, by_check


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evals.assistant", description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these cases")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (default 1)")
    parser.add_argument("--paid", action="store_true", help="allow a Claude ASSISTANT_MODEL, billed per question")
    args = parser.parse_args(argv)
    if assistant.MODEL.startswith("claude") and not args.paid:
        raise SystemExit(f"ASSISTANT_MODEL is {assistant.MODEL}, billed per question: pass --paid to run on it")

    cases = load_cases(args.only)
    print(f"{len(cases)} cases x {args.repeat} on {assistant.MODEL}; loading the model and data first")
    assistant.warm_up()
    assistant._wait_for_warm_up()  # so the first case's time is its own

    results = []
    for run in range(1, args.repeat + 1):
        for case in cases:
            result = run_case(case, run)
            results.append(result)
            status = "pass" if result["passed"] else "FAIL"
            print(f"{status} {case['id']} ({result['seconds']:.0f}s; {', '.join(result['tools']) or 'no tools'})")
            for line in result["failed"] + result["notes"]:
                print(f"       {line}")

    by_case, by_check = summarize(results)
    passed = int(sum(r["passed"] for r in results))
    print(f"\n{passed} of {len(results)} answers passed every check ({passed / len(results):.0%})")
    print("by check: " + ", ".join(f"{name} {row['sum']}/{row['count']}" for name, row in by_check.iterrows()))
    print(f"median time per question {pd.Series([r['seconds'] for r in results]).median():.0f}s")

    reports = resolve_path(load_config(), "reports")
    Path(reports).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results)[["case", "run", "passed", "failed", "notes", "tools", "seconds"]].assign(
        failed=lambda d: d["failed"].str.join("; "),
        notes=lambda d: d["notes"].str.join("; "),
        tools=lambda d: d["tools"].str.join(", "),
    ).to_csv(reports + "assistant_eval.csv", index=False)
    with open(reports + "assistant_eval.json", "w") as f:
        meta = {"model": assistant.MODEL, "ran_at": datetime.now().isoformat(timespec="seconds"), "passed": passed}
        json.dump({**meta, "results": results}, f, indent=2, default=str)
    print(f"wrote {reports}assistant_eval.csv and assistant_eval.json")
    return results


if __name__ == "__main__":
    main()
