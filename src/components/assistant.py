"""Claude tool-use layer over the retrieval/prediction functions.

Wires predict.py's predict_total_value and search.py's find_similar_houses /
find_deals as tools Claude can call, so a natural-language question decides
which one(s) to use rather than a hand-coded query parser. Requires
ANTHROPIC_API_KEY (in the environment or a .env file - see .env.example).
"""

import inspect
import json
import textwrap
from typing import Optional

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

from src.components import predict, search
from src.components.training import FEATURE_COLUMNS

load_dotenv()

MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You are a house-value assistant for Mecklenburg County, North Carolina. The "
    "models estimate a property's 2023 county assessed value (total_value), which "
    "is a reference point for judging a market price, not a market price itself - "
    "say so when you give an estimate. Houses are grouped into areas: CHARLOTTE_1 "
    "to CHARLOTTE_12 (parts of Charlotte, numbered north to south), DAVIDSON, "
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
        area: CHARLOTTE_1 to CHARLOTTE_12, DAVIDSON, CORNELIUS, HUNTERSVILLE, MINT HILL, MATTHEWS, or PINEVILLE.
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
        return json.dumps({"predicted_total_value": round(value, 2)})
    except (ValueError, FileNotFoundError) as e:
        return _tool_error(e)


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
        address: Street address starting with the house number, e.g. "221 Altondale Ave, Charlotte".
        parcel_id: County parcel id, e.g. "15506217".
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


TOOLS = [lookup_property, area_stats, predict_total_value, find_similar_houses, find_deals]


def _run(messages: list) -> anthropic.types.beta.BetaMessage:
    """Run the tool-use loop over a conversation; return Claude's final message."""
    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
        # if Claude Opus 5 declines, the API re-runs the request on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    last = None
    for message in runner:
        last = message
    return last


def _answer_text(message) -> str:
    if message.stop_reason == "refusal":
        return "The request was declined."
    return next((b.text for b in message.content if b.type == "text"), "")


def ask(question: str) -> str:
    """Run one natural-language question through the tool-use loop; return Claude's final answer text."""
    return _answer_text(_run([{"role": "user", "content": question}]))


def chat():
    """Multi-turn chat in the terminal, so Claude can ask follow-up questions (e.g. which of
    several address matches). Only the final answer of each turn is kept in the history;
    Claude re-calls a tool if it needs an earlier result again. Empty line or Ctrl-D quits."""
    messages = []
    while True:
        try:
            question = input("\nyou> ").strip()
        except EOFError:
            break
        if not question:
            break
        messages.append({"role": "user", "content": question})
        reply = _run(messages)
        messages.append({"role": "assistant", "content": reply.content})
        print(f"\nassistant> {_answer_text(reply)}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:]:
        print(ask(" ".join(sys.argv[1:])))
    else:
        chat()
