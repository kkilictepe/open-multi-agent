"""Example 04 -- Multi-Model Team with Custom Tools

Demonstrates:
- Mixing Anthropic and OpenAI models in the same team
- Defining custom tools with define_tool() and Pydantic models
- Building agents with a custom ToolRegistry so they can use custom tools
- Running a team goal that uses the custom tools

Run:
    python examples/example_04_multi_model_team.py

Prerequisites:
    ANTHROPIC_API_KEY env var must be set.
    OPENAI_API_KEY env var is optional (falls back to Anthropic).
"""

from __future__ import annotations

import asyncio
import json
import locale
import os
import random
import sys
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from open_multi_agent import (
    Agent,
    AgentConfig,
    AgentPool,
    ToolExecutor,
    ToolRegistry,
    define_tool,
    register_built_in_tools,
)

# ---------------------------------------------------------------------------
# Custom tools -- defined with define_tool() + Pydantic schemas
# ---------------------------------------------------------------------------


class ExchangeRateInput(BaseModel):
    from_currency: str = Field(alias="from", description='ISO 4217 currency code, e.g. "USD"')
    to: str = Field(description='ISO 4217 currency code, e.g. "EUR"')

    model_config = {"populate_by_name": True}


async def get_exchange_rate_handler(
    params: ExchangeRateInput, context: object
) -> dict[str, object]:
    """Fetch exchange rate (with graceful stub fallback)."""
    from_c = params.from_currency
    to_c = params.to

    try:
        import urllib.request

        url = f"https://api.exchangerate.host/convert?from={from_c}&to={to_c}&amount=1"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        rate = data.get("result") or (data.get("info") or {}).get("rate")
        if not isinstance(rate, (int, float)):
            raise ValueError("Unexpected API response shape")
        return {
            "data": json.dumps(
                {"from": from_c, "to": to_c, "rate": rate, "timestamp": datetime.now(timezone.utc).isoformat()}
            ),
            "isError": False,
        }
    except Exception as err:
        stub = round(random.uniform(0.8, 1.3), 4)
        return {
            "data": json.dumps(
                {
                    "from": from_c,
                    "to": to_c,
                    "rate": stub,
                    "note": f"Live fetch failed ({err}). Using stub rate.",
                }
            ),
            "isError": False,
        }


exchange_rate_tool = define_tool(
    name="get_exchange_rate",
    description=(
        "Get the current exchange rate between two currencies. "
        "Returns the rate as a decimal: 1 unit of `from` = N units of `to`."
    ),
    input_model=ExchangeRateInput,
    handler=get_exchange_rate_handler,
)


class FormatCurrencyInput(BaseModel):
    amount: float = Field(description="The numeric amount to format.")
    currency: str = Field(description='ISO 4217 currency code, e.g. "USD".')
    locale_str: str = Field(
        default="en_US",
        alias="locale",
        description='Locale string, e.g. "en_US". Defaults to "en_US".',
    )

    model_config = {"populate_by_name": True}


async def format_currency_handler(
    params: FormatCurrencyInput, context: object
) -> dict[str, object]:
    """Format a number as a currency string."""
    try:
        formatted = f"{params.amount:,.2f} {params.currency}"
        return {"data": formatted, "isError": False}
    except Exception:
        return {"data": f"{params.amount} {params.currency}", "isError": True}


format_currency_tool = define_tool(
    name="format_currency",
    description="Format a number as a localised currency string.",
    input_model=FormatCurrencyInput,
    handler=format_currency_handler,
)


# ---------------------------------------------------------------------------
# Helper: build an Agent with both built-in and custom tools registered.
# ---------------------------------------------------------------------------


def build_custom_agent(
    config: AgentConfig,
    extra_tools: list[object],
) -> Agent:
    registry = ToolRegistry()
    register_built_in_tools(registry)
    for tool in extra_tools:
        registry.register(tool)  # type: ignore[arg-type]
    executor = ToolExecutor(registry)
    return Agent(config, registry, executor)


# ---------------------------------------------------------------------------
# Agent definitions -- mixed providers
# ---------------------------------------------------------------------------

use_openai = bool(os.environ.get("OPENAI_API_KEY"))

researcher_config = AgentConfig(
    name="researcher",
    model="claude-sonnet-4-6",
    provider="anthropic",
    system_prompt=(
        "You are a financial data researcher.\n"
        "Use the get_exchange_rate tool to fetch current rates between the "
        "currency pairs you are given.\n"
        'Return the raw rates as a JSON object keyed by pair, e.g. '
        '{ "USD/EUR": 0.91, "USD/GBP": 0.79 }.'
    ),
    tools=["get_exchange_rate"],
    max_turns=6,
    temperature=0.0,
)

analyst_config = AgentConfig(
    name="analyst",
    model="gpt-4o" if use_openai else "claude-sonnet-4-6",
    provider="openai" if use_openai else "anthropic",
    system_prompt=(
        "You are a foreign exchange analyst.\n"
        "You receive exchange rate data and produce a short briefing.\n"
        "Use format_currency to show example conversions.\n"
        "Keep the briefing under 200 words."
    ),
    tools=["format_currency"],
    max_turns=4,
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# Build agents with custom tools
# ---------------------------------------------------------------------------

researcher = build_custom_agent(researcher_config, [exchange_rate_tool])
analyst = build_custom_agent(analyst_config, [format_currency_tool])

# ---------------------------------------------------------------------------
# Run with AgentPool for concurrency control
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Multi-model team with custom tools")
    print(
        f"Providers: researcher=anthropic, "
        f"analyst={'openai (gpt-4o)' if use_openai else 'anthropic (fallback)'}"
    )
    print(f"Custom tools: {exchange_rate_tool.name}, {format_currency_tool.name}")
    print()

    pool = AgentPool(max_concurrency=1)  # sequential for readability
    pool.add(researcher)
    pool.add(analyst)

    # Step 1: researcher fetches the rates
    print("[1/2] Researcher fetching FX rates...")
    research_result = await pool.run(
        "researcher",
        "Fetch exchange rates for these pairs using the get_exchange_rate tool:\n"
        "- USD to EUR\n"
        "- USD to GBP\n"
        "- USD to JPY\n"
        "- EUR to GBP\n\n"
        'Return the results as a JSON object: { "USD/EUR": <rate>, "USD/GBP": <rate>, ... }',
    )

    if not research_result.success:
        print("Researcher failed:", research_result.output)
        sys.exit(1)

    tool_names = ", ".join(c.tool_name for c in research_result.tool_calls)
    print(f"Researcher done. Tool calls made: {tool_names}")

    # Step 2: analyst writes the briefing
    print("\n[2/2] Analyst writing FX briefing...")
    analyst_result = await pool.run(
        "analyst",
        f"Here are the current FX rates gathered by the research team:\n\n"
        f"{research_result.output}\n\n"
        "Using format_currency, show what $1,000 USD and 1,000 EUR convert to "
        "in each of the other currencies.\n"
        "Then write a short FX market briefing (under 200 words) covering:\n"
        "- Each rate with a brief observation\n"
        "- The strongest and weakest currency in the set\n"
        "- One-sentence market comment",
    )

    if not analyst_result.success:
        print("Analyst failed:", analyst_result.output)
        sys.exit(1)

    tool_names = ", ".join(c.tool_name for c in analyst_result.tool_calls)
    print(f"Analyst done. Tool calls made: {tool_names}")

    # -- Results ---------------------------------------------------------------

    print("\n" + "=" * 60)

    print("\nResearcher output:")
    print(research_result.output[:400])

    print("\nAnalyst briefing:")
    print("\u2500" * 60)
    print(analyst_result.output)
    print("\u2500" * 60)

    total_input = (
        research_result.token_usage.input_tokens + analyst_result.token_usage.input_tokens
    )
    total_output = (
        research_result.token_usage.output_tokens + analyst_result.token_usage.output_tokens
    )
    print(f"\nTotal tokens -- input: {total_input}, output: {total_output}")

    # -- Bonus: test custom tools in isolation ---------------------------------

    print("\n--- Bonus: testing custom tools in isolation ---\n")

    fmt_result = await format_currency_tool.handler(
        FormatCurrencyInput(amount=1234.56, currency="EUR", locale_str="de_DE"),
        None,
    )
    print(f"format_currency(1234.56, EUR, de_DE) = {fmt_result['data']}")

    rate_result = await get_exchange_rate_handler(
        ExchangeRateInput(**{"from": "USD", "to": "EUR"}),
        None,
    )
    print(f"get_exchange_rate(USD->EUR) = {rate_result['data']}")


if __name__ == "__main__":
    asyncio.run(main())
