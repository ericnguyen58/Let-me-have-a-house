"""Tool-use layer over the retrieval/prediction functions.

Wires predict.py's predict_total_value, search.py's lookup / comps / deals and
market_estimate.py's market_value as tools the model can call, so a natural-language
question decides which one(s) to use rather than a hand-coded query parser.

ASSISTANT_MODEL (environment or .env, see .env.example) picks the model: a local
Ollama model, free to run (default qwen3.5:9b; needs Ollama running and the model
pulled), or a Claude model id such as claude-opus-5 (needs ANTHROPIC_API_KEY).
"""

import inspect
import json
import os
import textwrap
from typing import Optional

import anthropic
import requests
from anthropic import beta_tool
from dotenv import load_dotenv

from src.components import market_estimate, predict, search
from src.components.training import FEATURE_COLUMNS

load_dotenv()

MODEL = os.getenv("ASSISTANT_MODEL", "qwen3.5:9b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Ollama's default context (4,096 tokens) can't hold the tool schemas plus a few results
OLLAMA_CONTEXT = 32768
# A small model can get stuck calling tools; stop it after this many rounds
MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = (
    "You are a house-value assistant for Mecklenburg County, North Carolina. For what a "
    "real property is worth or would sell for today, call lookup_property, then "
    "estimate_market_value with its parcel_id. Give the market estimate with its 80% "
    "range and the number of sales behind the ratio, and say so when confidence is not "
    "normal. For a what-if (an addition, an extra bathroom), pass the new values to "
    "estimate_market_value and give change_vs_unchanged as what the change is worth. "
    "predict_total_value estimates the 2023 county assessed value (total_value), a "
    "reference point, not a market price - say so when you give it. Its "
    "adjusted_to_today only scales that by Zillow's county-wide index, so use it only "
    "for a described house with no address, and call it rough. Houses are grouped into areas: CHARLOTTE_1 "
    "to CHARLOTTE_15 (parts of Charlotte, numbered north to south), DAVIDSON, "
    "CORNELIUS, HUNTERSVILLE, MINT HILL, MATTHEWS and PINEVILLE. Any feature left "
    "out of a tool call is filled with a county-wide typical value, so estimates "
    "and comps are much weaker without the house's area and location. When the user "
    "gives an address or parcel id, call lookup_property first and pass its feature "
    "values into predict_total_value / find_similar_houses; if it returns several "
    "matches, ask which one. For a described house with no address, use area_stats "
    "for typical values of its area, or ask for the address. Ground every dollar figure in an actual "
    "tool result - never state one you didn't get from a tool. find_deals results "
    "are past sales, not houses for sale now. Some are renovation-driven gaps (a "
    "distressed property sold cheap as-is, then renovated, so its current record "
    "reflects the renovated state), and because assessed values usually sit below "
    "market prices, the gap is not purely a bargain - mention both caveats "
    "whenever you present deals."
)

# Shared by the predict/similar tools, which take the same feature arguments.
FEATURE_ARGS_DOC = """
        land_use_class: single_family, condo, or townhouse.
        area: CHARLOTTE_1 to CHARLOTTE_15, DAVIDSON, CORNELIUS, HUNTERSVILLE, MINT HILL, MATTHEWS, or PINEVILLE.
        acreage: Lot size in acres.
        year_built: Year the structure was built.
        heated_area: Heated square footage.
        bedrooms: Number of bedrooms.
        bathrooms: Number of bathrooms (half baths count as 0.5).
        latitude: Latitude of the property.
        longitude: Longitude of the property.
        property_age: Current year minus year_built.
        neighborhood_median_price_per_sqft: Median $/sqft of recent sales in the property's neighborhood, if known.
        neighborhood_sale_count: Number of recent arms-length sales in the neighborhood, if known.
        price_momentum: Yearly growth rate of the neighborhood's price per sqft over 2020-2022 (e.g. 0.19 for 19%), if known.
        education_nearest_miles: Miles to the nearest school, if known.
        places_count: Number of places of any kind within 5 miles, if known.
        health_care_share: Share (0-1) of places within 5 miles that are health care, if known.
        activity_share: Share (0-1) of places within 5 miles that are activities (parks, theaters, gyms, ...), if known.
        emergency_fire_miles: Miles to the nearest fire station, if known.
        emergency_police_miles: Miles to the nearest police station, if known.
        emergency_medical_miles: Miles to the nearest emergency department, if known.
        elementary_school_score: NC performance score (0-100) of the assigned elementary school, if known.
        middle_school_score: NC performance score (0-100) of the assigned middle school, if known.
        high_school_score: NC performance score (0-100) of the assigned high school, if known.
        violent_crime_rate: Violent crimes per 1,000 residents in the neighborhood profile area, if known.
        property_crime_rate: Property crimes per 1,000 residents in the neighborhood profile area, if known.
        in_floodplain: 1 if the property is in the FEMA floodplain, else 0, if known.
        light_rail_miles: Miles to the nearest LYNX Blue Line station, if known.
        feature_set: "with_neighborhood_price" (default, meant for estimates) or "without_neighborhood_price" (value from area and accessibility alone)."""


def _features(args):
    return {k: v for k, v in args.items() if k in FEATURE_COLUMNS}


def _with_feature_docs(fn):
    """Fill FEATURE_ARGS_DOC into fn's docstring. Must sit under @beta_tool, which builds
    the tool schema from the docstring when it's applied. Both are dedented first, since
    Python 3.13+ already strips docstring indentation at compile time and older versions don't."""
    args_doc = textwrap.indent(textwrap.dedent(FEATURE_ARGS_DOC), "    ")
    fn.__doc__ = inspect.cleandoc(fn.__doc__).format(feature_args=args_doc)
    return fn


def _tool_error(e):
    # tool_result content must be a string; errors go back to Claude to explain
    if isinstance(e, FileNotFoundError):
        return json.dumps({"error": "trained models not found - run src.components.training first"})
    return json.dumps({"error": str(e)})


@beta_tool
@_with_feature_docs
def predict_total_value(
    land_use_class: Optional[str] = None,
    area: Optional[str] = None,
    acreage: Optional[float] = None,
    year_built: Optional[float] = None,
    heated_area: Optional[float] = None,
    bedrooms: Optional[float] = None,
    bathrooms: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    property_age: Optional[float] = None,
    neighborhood_median_price_per_sqft: Optional[float] = None,
    neighborhood_sale_count: Optional[float] = None,
    price_momentum: Optional[float] = None,
    education_nearest_miles: Optional[float] = None,
    places_count: Optional[float] = None,
    health_care_share: Optional[float] = None,
    activity_share: Optional[float] = None,
    emergency_fire_miles: Optional[float] = None,
    emergency_police_miles: Optional[float] = None,
    emergency_medical_miles: Optional[float] = None,
    elementary_school_score: Optional[float] = None,
    middle_school_score: Optional[float] = None,
    high_school_score: Optional[float] = None,
    violent_crime_rate: Optional[float] = None,
    property_crime_rate: Optional[float] = None,
    in_floodplain: Optional[float] = None,
    light_rail_miles: Optional[float] = None,
    feature_set: str = predict.DEFAULT_FEATURE_SET,
) -> str:
    """Estimate a Mecklenburg property's 2023 assessed value (total_value) from its features.

    Missing fields are imputed the same way the model was trained (median/most-frequent) -
    omit anything the caller doesn't know.

    Args:{feature_args}
    """
    features = _features(locals())
    try:
        value = predict.predict_total_value(features, feature_set=feature_set)
    except (ValueError, FileNotFoundError) as e:
        return _tool_error(e)
    result = {"predicted_total_value": round(value, 2)}
    try:
        factor, month = predict.market_adjustment()
        result.update(adjusted_to_today=round(value * factor, 2), index_change=round(factor - 1, 4), index_month=month)
    except FileNotFoundError:
        pass  # Zillow index not downloaded yet - estimate without the adjustment
    return json.dumps(result)


@beta_tool
@_with_feature_docs
def find_similar_houses(
    land_use_class: Optional[str] = None,
    area: Optional[str] = None,
    acreage: Optional[float] = None,
    year_built: Optional[float] = None,
    heated_area: Optional[float] = None,
    bedrooms: Optional[float] = None,
    bathrooms: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    property_age: Optional[float] = None,
    neighborhood_median_price_per_sqft: Optional[float] = None,
    neighborhood_sale_count: Optional[float] = None,
    price_momentum: Optional[float] = None,
    education_nearest_miles: Optional[float] = None,
    places_count: Optional[float] = None,
    health_care_share: Optional[float] = None,
    activity_share: Optional[float] = None,
    emergency_fire_miles: Optional[float] = None,
    emergency_police_miles: Optional[float] = None,
    emergency_medical_miles: Optional[float] = None,
    elementary_school_score: Optional[float] = None,
    middle_school_score: Optional[float] = None,
    high_school_score: Optional[float] = None,
    violent_crime_rate: Optional[float] = None,
    property_crime_rate: Optional[float] = None,
    in_floodplain: Optional[float] = None,
    light_rail_miles: Optional[float] = None,
    feature_set: str = predict.DEFAULT_FEATURE_SET,
    k: int = 5,
) -> str:
    """Find comparable Mecklenburg houses, nearest by distance in the model's feature space.

    Args:{feature_args}
        k: Number of comps to return.
    """
    features = _features(locals())
    try:
        return json.dumps(search.find_similar_houses(features, k=k, feature_set=feature_set), default=str)
    except (ValueError, FileNotFoundError) as e:
        return _tool_error(e)


@beta_tool
def find_deals(
    feature_set: str = predict.DEFAULT_FEATURE_SET,
    top_n: int = 10,
    area: Optional[str] = None,
) -> str:
    """Find past arms-length sales in Mecklenburg priced well below the trained model's
    estimate for that property - retrospective "deals", not houses for sale now. Some
    results are likely flips (cheap as-is sale, later renovated), and assessed values
    usually sit below market prices - mention both when presenting results.

    Args:
        feature_set: "with_neighborhood_price" (default) or "without_neighborhood_price".
        top_n: Number of deals to return, ranked by biggest gap first.
        area: Limit results to one area, e.g. MATTHEWS or CHARLOTTE_3. Omit for the whole county.
    """
    try:
        return json.dumps(
            search.find_deals(feature_set=feature_set, top_n=top_n, area=area),
            default=str,
        )
    except (ValueError, FileNotFoundError) as e:
        return _tool_error(e)


@beta_tool
def lookup_property(address: Optional[str] = None, parcel_id: Optional[str] = None) -> str:
    """Look up a Mecklenburg property's county record by street address or parcel id: its
    assessed value, last sale and every model feature. Pass the returned feature values
    into predict_total_value / find_similar_houses. May return several matches, or none.

    Args:
        address: Street address starting with the house number, e.g. "3612 Abbey Hill Ln, Charlotte".
        parcel_id: County parcel id, e.g. "20947309".
    """
    try:
        return json.dumps(search.lookup_property(address=address, parcel_id=parcel_id), default=str)
    except ValueError as e:
        return _tool_error(e)


@beta_tool
def area_stats(area: Optional[str] = None) -> str:
    """Typical values per area: house count, median assessed value, size, year built, recent
    sale $/sqft, price_momentum and accessibility. Use it to compare areas, or for typical
    feature values when the user describes a house without an address.

    Args:
        area: One area, e.g. MATTHEWS or CHARLOTTE_3. Omit for every area.
    """
    try:
        return json.dumps(search.area_stats(area=area), default=str)
    except ValueError as e:
        return _tool_error(e)


@beta_tool
def estimate_market_value(
    parcel_id: str,
    heated_area: Optional[float] = None,
    bedrooms: Optional[float] = None,
    bathrooms: Optional[float] = None,
) -> str:
    """Estimate what a Mecklenburg property would sell for today, with an 80% range: its
    2023 assessed value times the median sale / assessed ratio of recent sales in its
    neighborhood. Use this, not predict_total_value, for today's price of a real property.

    Pass heated_area, bedrooms or bathrooms only for a what-if, such as an addition. The
    estimate then starts from the model's value of the changed house instead of the
    county's, and change_vs_unchanged is what the change alone is worth.

    Args:
        parcel_id: County parcel id from lookup_property, e.g. "20947309".
        heated_area: New heated square footage, for a what-if.
        bedrooms: New number of bedrooms, for a what-if.
        bathrooms: New number of bathrooms (half baths count as 0.5), for a what-if.
    """
    changes = {k: v for k, v in [("heated_area", heated_area), ("bedrooms", bedrooms), ("bathrooms", bathrooms)]
               if v is not None}
    try:
        matches = search.lookup_property(parcel_id=parcel_id)
        if not matches:
            raise ValueError(f"no property with parcel id {parcel_id}")
        house = matches[0]
        result = market_estimate.what_if(house, changes) if changes else market_estimate.market_value(house)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})  # a missing file's message says which command makes it
    return json.dumps(result, default=str)


TOOLS = [lookup_property, area_stats, estimate_market_value, predict_total_value, find_similar_houses, find_deals]


def _run_claude(messages: list) -> anthropic.types.beta.BetaMessage:
    """Run the tool-use loop over a conversation; return Claude's final message."""
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
        # if Claude declines, the API re-runs the request on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    last = None
    for message in runner:
        last = message
    return last


def _call_tool(tools, call) -> str:
    """Run one Ollama tool call; a bad name or bad arguments go back to the model as an error."""
    name, args = call["function"]["name"], call["function"]["arguments"]
    if name not in tools:
        return json.dumps({"error": f"unknown tool {name!r}"})
    try:
        return tools[name].call(args)
    except Exception as e:  # argument validation; the tools catch their own errors
        return json.dumps({"error": f"bad arguments for {name}: {e}"})


def _run_ollama(messages: list) -> dict:
    """The same tool-use loop on a local Ollama model, with TOOLS' schemas from @beta_tool;
    return the model's final message."""
    tools = {t.name: t for t in TOOLS}
    schemas = [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                                 "parameters": d["input_schema"]}}
               for d in (t.to_dict() for t in TOOLS)]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = requests.post(f"{OLLAMA_URL}/api/chat", timeout=600, json={
                "model": MODEL, "messages": messages, "tools": schemas, "stream": False,
                "options": {"num_ctx": OLLAMA_CONTEXT},
            })
        except requests.ConnectionError:
            raise SystemExit(f"Ollama is not running at {OLLAMA_URL}: start it with `ollama serve`, "
                             f"or set ASSISTANT_MODEL to a Claude model") from None
        if response.status_code == 404:
            raise SystemExit(f"{response.json().get('error')} - run `ollama pull {MODEL}`")
        response.raise_for_status()
        message = response.json()["message"]
        messages.append(message)
        if not message.get("tool_calls"):
            return message
        for call in message["tool_calls"]:
            messages.append({"role": "tool", "tool_name": call["function"]["name"], "content": _call_tool(tools, call)})
    return {"role": "assistant", "content": f"Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls without an answer."}


def _run(messages: list) -> tuple[str, dict]:
    """Run the tool-use loop over a conversation on MODEL; return the answer text and the
    assistant message to keep in the history."""
    if MODEL.startswith("claude"):
        message = _run_claude(messages)
        text = ("The request was declined." if message.stop_reason == "refusal"
                else next((b.text for b in message.content if b.type == "text"), ""))
        return text, {"role": "assistant", "content": message.content}
    message = _run_ollama(messages)
    return message["content"], {"role": "assistant", "content": message["content"]}


def ask(question: str) -> str:
    """Run one natural-language question through the tool-use loop; return the final answer text."""
    return _run([{"role": "user", "content": question}])[0]


def chat():
    """Multi-turn chat in the terminal, so the model can ask follow-up questions (e.g. which
    of several address matches). Only the final answer of each turn is kept in the history;
    the model re-calls a tool if it needs an earlier result again. Empty line or Ctrl-D quits."""
    messages = []
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            break
        if not question:
            break
        messages.append({"role": "user", "content": question})
        text, reply = _run(messages)
        messages.append(reply)
        print(f"\nassistant> {text}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:]:
        print(ask(" ".join(sys.argv[1:])))
    else:
        chat()
