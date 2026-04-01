"""Example 01 -- Single Agent

The simplest possible usage: one agent with bash and file tools, running
a coding task. Then shows streaming output using the Agent class directly.

Run:
    python examples/example_01_single_agent.py

Prerequisites:
    ANTHROPIC_API_KEY env var must be set.
"""

from __future__ import annotations

import asyncio
import sys

from open_multi_agent import (
    Agent,
    AgentConfig,
    OpenMultiAgent,
    OrchestratorConfig,
    OrchestratorEvent,
    ToolExecutor,
    ToolRegistry,
    register_built_in_tools,
)


async def main() -> None:
    # -- Part 1: Single agent via OpenMultiAgent (simplest path) ---------------

    def on_progress(event: OrchestratorEvent) -> None:
        if event.type == "agent_start":
            print(f"[start]    agent={event.agent}")
        elif event.type == "agent_complete":
            print(f"[complete] agent={event.agent}")

    orchestrator = OpenMultiAgent(
        OrchestratorConfig(
            default_model="claude-sonnet-4-6",
            on_progress=on_progress,
        )
    )

    print("Part 1: run_agent() -- single one-shot task\n")

    result = await orchestrator.run_agent(
        AgentConfig(
            name="coder",
            model="claude-sonnet-4-6",
            system_prompt=(
                "You are a focused Python developer.\n"
                "When asked to implement something, write clean, minimal code "
                "with no extra commentary.\n"
                "Use the bash tool to run commands and the file tools to read/write files."
            ),
            tools=["bash", "file_read", "file_write"],
            max_turns=8,
        ),
        "Create a small Python utility function in /tmp/greet.py that:\n"
        "1. Defines a function named greet(name: str) -> str\n"
        '2. Returns "Hello, <name>!"\n'
        "3. Adds a brief usage comment at the top of the file.\n"
        'Then add a default call greet("World") at the bottom and run the file with: python /tmp/greet.py',
    )

    if result.success:
        print("\nAgent output:")
        print("\u2500" * 60)
        print(result.output)
        print("\u2500" * 60)
    else:
        print("Agent failed:", result.output)
        sys.exit(1)

    print("\nToken usage:")
    print(f"  input:  {result.token_usage.input_tokens}")
    print(f"  output: {result.token_usage.output_tokens}")
    print(f"  tool calls made: {len(result.tool_calls)}")

    # -- Part 2: Streaming via Agent directly ----------------------------------

    print("\n\nPart 2: Agent.stream() -- incremental text output\n")

    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)

    streaming_agent = Agent(
        AgentConfig(
            name="explainer",
            model="claude-sonnet-4-6",
            system_prompt="You are a concise technical writer. Keep explanations brief.",
            max_turns=3,
        ),
        registry,
        executor,
    )

    sys.stdout.write("Streaming: ")
    async for event in streaming_agent.stream(
        "In two sentences, explain what a Python generic type constraint is."
    ):
        if event.type == "text" and isinstance(event.data, str):
            sys.stdout.write(event.data)
        elif event.type == "done":
            sys.stdout.write("\n")
        elif event.type == "error":
            print(f"\nStream error: {event.data}")

    # -- Part 3: Multi-turn conversation via Agent.prompt() --------------------

    print("\nPart 3: Agent.prompt() -- multi-turn conversation\n")

    conversation_agent = Agent(
        AgentConfig(
            name="tutor",
            model="claude-sonnet-4-6",
            system_prompt="You are a Python tutor. Give short, direct answers.",
            max_turns=2,
        ),
        ToolRegistry(),
        ToolExecutor(ToolRegistry()),
    )

    turn1 = await conversation_agent.prompt("What is a type guard in Python?")
    print("Turn 1:", turn1.output[:200])

    turn2 = await conversation_agent.prompt(
        "Give me one concrete code example of what you just described."
    )
    print("\nTurn 2:", turn2.output[:300])

    print(f"\nConversation history length: {len(conversation_agent.get_history())} messages")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
