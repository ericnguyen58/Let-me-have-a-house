import json

import pytest

from src.components import assistant


@pytest.mark.parametrize(
    "message, reply",
    [
        ("hi", assistant.GREETING_REPLY),
        ("Hello there, how are you?", assistant.GREETING_REPLY),
        ("thanks so much!", assistant.THANKS_REPLY),
        ("bye", assistant.GOODBYE_REPLY),
        ("what can you do?", assistant.HELP_REPLY),
        ("???", assistant.EMPTY_REPLY),
        ("x" * (assistant.MAX_QUESTION_CHARS + 1), assistant.TOO_LONG_REPLY),
        ("Matthews", assistant.MORE_INFO_REPLY),  # too vague as a first message
        ("3612 Abbey", None),  # a number: an address, parcel or budget
        ("hi, what is 3612 Abbey Hill Ln worth?", None),
    ],
)
def test_quick_reply(message, reply):
    assert assistant.quick_reply(message) == reply


def test_a_short_follow_up_goes_to_the_model():
    assert assistant.quick_reply("Matthews", follow_up=True) is None


def test_ask_answers_small_talk_without_a_model(monkeypatch):
    monkeypatch.setattr(assistant, "_run_ollama", lambda messages: pytest.fail("called the model"))
    monkeypatch.setattr(assistant, "_run_claude", lambda messages: pytest.fail("called the model"))
    assert assistant.ask("thank you") == assistant.THANKS_REPLY


def _turns(n, chars=500):
    return [
        message
        for i in range(n)
        for message in [
            {"role": "user", "content": f"question {i} " + "q" * chars},
            {"role": "assistant", "content": "a" * chars},
        ]
    ]


def test_fit_history_keeps_a_short_conversation():
    messages = [*_turns(3), {"role": "user", "content": "new"}]
    assert assistant._fit_history(messages) == messages


def test_fit_history_drops_old_turns_and_lists_their_questions():
    # 20 turns of about 970 characters: one drop of HISTORY_DROP_TURNS brings them under MAX_HISTORY_CHARS
    messages = [*_turns(20, chars=480), {"role": "user", "content": "new"}]
    fitted = assistant._fit_history(messages)
    dropped = assistant.HISTORY_DROP_TURNS
    assert len(fitted) == 2 * (20 - dropped) + 1
    assert fitted[-1] == {"role": "user", "content": "new"}
    note, first_kept = fitted[0]["content"].split("\n\n", 1)
    assert note.startswith("(Earlier in this chat I asked:")
    assert "question 0 " in note and f"question {dropped - 1} " in note and f"question {dropped} " not in note
    assert first_kept.startswith(f"question {dropped} ")


def test_unknown_tool_goes_back_to_the_model_as_an_error():
    tools = {t.name: t for t in assistant.TOOLS}
    result = assistant._call_tool(tools, {"function": {"name": "sell_my_house", "arguments": {}}})
    assert json.loads(result) == {"error": "unknown tool 'sell_my_house'"}


def test_tool_schemas_carry_the_shared_feature_docs():
    schemas = {t["function"]["name"]: t["function"] for t in assistant.OLLAMA_TOOLS}
    assert set(schemas) == {t.name for t in assistant.TOOLS}
    assert "heated_area" in schemas["predict_total_value"]["parameters"]["properties"]
    assert "{feature_args}" not in schemas["find_similar_houses"]["description"]
