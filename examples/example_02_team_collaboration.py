"""Example 02 -- Multi-Agent Team Collaboration

Three specialised agents (architect, developer, reviewer) collaborate on a
shared goal. The OpenMultiAgent orchestrator breaks the goal into tasks, assigns
them to the right agents, and collects the results.

Run:
    python examples/example_02_team_collaboration.py

Prerequisites:
    ANTHROPIC_API_KEY env var must be set.
"""

from __future__ import annotations

import asyncio
import time

from open_multi_agent import (
    AgentConfig,
    OpenMultiAgent,
    OrchestratorConfig,
    OrchestratorEvent,
    TeamConfig,
)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

architect = AgentConfig(
    name="architect",
    model="claude-sonnet-4-6",
    provider="anthropic",
    system_prompt=(
        "You are a software architect with deep experience in Python and REST API design.\n"
        "Your job is to design clear, production-quality API contracts and file/directory structures.\n"
        "Output concise plans in markdown -- no unnecessary prose."
    ),
    tools=["bash", "file_write"],
    max_turns=5,
    temperature=0.2,
)

developer = AgentConfig(
    name="developer",
    model="claude-sonnet-4-6",
    provider="anthropic",
    system_prompt=(
        "You are a Python developer. You implement what the architect specifies.\n"
        "Write clean, runnable code with proper error handling. "
        "Use the tools to write files and run tests."
    ),
    tools=["bash", "file_read", "file_write", "file_edit"],
    max_turns=12,
    temperature=0.1,
)

reviewer = AgentConfig(
    name="reviewer",
    model="claude-sonnet-4-6",
    provider="anthropic",
    system_prompt=(
        "You are a senior code reviewer. Review code for correctness, security, and clarity.\n"
        "Provide a structured review with: LGTM items, suggestions, and any blocking issues.\n"
        "Read files using the tools before reviewing."
    ),
    tools=["bash", "file_read", "grep"],
    max_turns=5,
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

start_times: dict[str, float] = {}


def handle_progress(event: OrchestratorEvent) -> None:
    ts = time.strftime("%H:%M:%S")

    match event.type:
        case "agent_start":
            start_times[event.agent or ""] = time.time()
            print(f"[{ts}] AGENT START  -> {event.agent}")

        case "agent_complete":
            elapsed = time.time() - start_times.get(event.agent or "", time.time())
            print(f"[{ts}] AGENT DONE   <- {event.agent} ({elapsed:.0f}s)")

        case "task_start":
            print(f"[{ts}] TASK START   | {event.task}")

        case "task_complete":
            print(f"[{ts}] TASK DONE    ^ {event.task}")

        case "message":
            print(f"[{ts}] MESSAGE      * {event.agent} -> (team)")

        case "error":
            print(f"[{ts}] ERROR        x agent={event.agent} task={event.task}")
            if isinstance(event.data, Exception):
                print(f"               {event.data}")


# ---------------------------------------------------------------------------
# Orchestrate
# ---------------------------------------------------------------------------


async def main() -> None:
    orchestrator = OpenMultiAgent(
        OrchestratorConfig(
            default_model="claude-sonnet-4-6",
            max_concurrency=1,  # run agents sequentially so output is readable
            on_progress=handle_progress,
        )
    )

    team = orchestrator.create_team(
        "api-team",
        TeamConfig(
            name="api-team",
            agents=[architect, developer, reviewer],
            shared_memory=True,
            max_concurrency=1,
        ),
    )

    agent_names = ", ".join(a.name for a in team.get_agents())
    print(f'Team "{team.name}" created with agents: {agent_names}')
    print("\nStarting team run...\n")
    print("=" * 60)

    goal = (
        "Create a minimal Flask REST API in /tmp/flask-api/ with:\n"
        '- GET  /health       -> { "status": "ok" }\n'
        "- GET  /users        -> returns a hardcoded list of 2 user dicts\n"
        '- POST /users        -> accepts { "name", "email" } body, logs it, returns 201\n'
        "- Proper error handling middleware\n"
        "- The server should listen on port 3001\n"
        "- Include a requirements.txt with the required dependencies"
    )

    result = await orchestrator.run_team(team, goal)

    print("\n" + "=" * 60)

    # -- Results ---------------------------------------------------------------

    print("\nTeam run complete.")
    print(f"Success: {result.success}")
    print(
        f"Total tokens -- input: {result.total_token_usage.input_tokens}, "
        f"output: {result.total_token_usage.output_tokens}"
    )

    print("\nPer-agent results:")
    for agent_name, agent_result in result.agent_results.items():
        status = "OK" if agent_result.success else "FAILED"
        tools = len(agent_result.tool_calls)
        print(f"  {agent_name:<12} [{status}]  tool_calls={tools}")
        if not agent_result.success:
            print(f"    Error: {agent_result.output[:120]}")

    developer_result = result.agent_results.get("developer")
    if developer_result and developer_result.success:
        print("\nDeveloper output (last 600 chars):")
        print("\u2500" * 60)
        out = developer_result.output
        print(("..." + out[-600:]) if len(out) > 600 else out)
        print("\u2500" * 60)

    reviewer_result = result.agent_results.get("reviewer")
    if reviewer_result and reviewer_result.success:
        print("\nReviewer output:")
        print("\u2500" * 60)
        print(reviewer_result.output)
        print("\u2500" * 60)


if __name__ == "__main__":
    asyncio.run(main())
