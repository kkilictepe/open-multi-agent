"""Example 03 -- Explicit Task Pipeline with Dependencies

Demonstrates how to define tasks with explicit dependency chains
(design -> implement -> test -> review) using run_tasks(). The TaskQueue
automatically blocks downstream tasks until their dependencies complete.

Run:
    python examples/example_03_task_pipeline.py

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
    Task,
    TeamConfig,
)

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

designer = AgentConfig(
    name="designer",
    model="claude-sonnet-4-6",
    system_prompt=(
        "You are a software designer. Your output is always a concise technical spec\n"
        "in markdown. Focus on interfaces, data shapes, and file structure. Be brief."
    ),
    tools=["file_write"],
    max_turns=4,
)

implementer = AgentConfig(
    name="implementer",
    model="claude-sonnet-4-6",
    system_prompt=(
        "You are a Python developer. Read the design spec written by the designer,\n"
        "then implement it. Write all files to /tmp/pipeline-output/. Use the tools."
    ),
    tools=["bash", "file_read", "file_write"],
    max_turns=10,
)

tester = AgentConfig(
    name="tester",
    model="claude-sonnet-4-6",
    system_prompt=(
        "You are a QA engineer. Read the implemented files and run them to verify correctness.\n"
        "Report: what passed, what failed, and any bugs found."
    ),
    tools=["bash", "file_read", "grep"],
    max_turns=6,
)

reviewer = AgentConfig(
    name="reviewer",
    model="claude-sonnet-4-6",
    system_prompt=(
        "You are a code reviewer. Read all files and produce a brief structured review.\n"
        "Sections: Summary, Strengths, Issues (if any), Verdict (SHIP / NEEDS WORK)."
    ),
    tools=["file_read", "grep"],
    max_turns=4,
)

# ---------------------------------------------------------------------------
# Progress handler -- shows dependency blocking/unblocking
# ---------------------------------------------------------------------------

task_times: dict[str, float] = {}


def handle_progress(event: OrchestratorEvent) -> None:
    ts = time.strftime("%H:%M:%S")

    match event.type:
        case "task_start":
            task_times[event.task or ""] = time.time()
            task = event.data if isinstance(event.data, Task) else None
            title = task.title if task else event.task
            assignee = task.assignee if task else "any"
            print(f'[{ts}] TASK READY    "{title}" (assignee: {assignee})')

        case "task_complete":
            elapsed = time.time() - task_times.get(event.task or "", time.time())
            task = event.data if isinstance(event.data, Task) else None
            title = task.title if task else event.task
            print(f'[{ts}] TASK DONE     "{title}" in {elapsed:.0f}s')

        case "agent_start":
            print(f"[{ts}] AGENT START   {event.agent}")

        case "agent_complete":
            print(f"[{ts}] AGENT DONE    {event.agent}")

        case "error":
            print(f"[{ts}] ERROR         {event.agent or ''}  task={event.task}")


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------


async def main() -> None:
    orchestrator = OpenMultiAgent(
        OrchestratorConfig(
            default_model="claude-sonnet-4-6",
            max_concurrency=2,  # allow test + review to potentially run in parallel
            on_progress=handle_progress,
        )
    )

    team = orchestrator.create_team(
        "pipeline-team",
        TeamConfig(
            name="pipeline-team",
            agents=[designer, implementer, tester, reviewer],
            shared_memory=True,
        ),
    )

    SPEC_FILE = "/tmp/pipeline-output/design-spec.md"

    tasks = [
        {
            "title": "Design: URL shortener data model",
            "description": (
                f"Design a minimal in-memory URL shortener service.\n"
                f"Write a markdown spec to {SPEC_FILE} covering:\n"
                "- Python dataclasses for Url and ShortenRequest\n"
                "- The shortening algorithm (hash approach is fine)\n"
                "- API contract: POST /shorten, GET /:code\n"
                "Keep the spec under 30 lines."
            ),
            "assignee": "designer",
            # no dependencies -- this is the root task
        },
        {
            "title": "Implement: URL shortener",
            "description": (
                f"Read the design spec at {SPEC_FILE}.\n"
                "Implement the URL shortener in /tmp/pipeline-output/src/:\n"
                "- shortener.py: core logic (shorten, resolve functions)\n"
                "- server.py: tiny HTTP server using Python's built-in http.server module\n"
                "  - POST /shorten  body: { url: string } -> { code, short }\n"
                "  - GET  /:code    -> redirect (301) or 404\n"
                "- main.py: entry point that starts the server on port 3002\n"
                "No external dependencies beyond the standard library."
            ),
            "assignee": "implementer",
            "dependsOn": ["Design: URL shortener data model"],
        },
        {
            "title": "Test: URL shortener",
            "description": (
                "Run the URL shortener implementation:\n"
                "1. Start the server: python /tmp/pipeline-output/src/main.py &\n"
                "2. POST a URL to shorten it using curl\n"
                "3. Verify the GET redirect works\n"
                "4. Report what passed and what (if anything) failed.\n"
                "Kill the server after testing."
            ),
            "assignee": "tester",
            "dependsOn": ["Implement: URL shortener"],
        },
        {
            "title": "Review: URL shortener",
            "description": (
                "Read all .py files in /tmp/pipeline-output/src/ and the design spec.\n"
                "Produce a structured code review with sections:\n"
                "- Summary (2 sentences)\n"
                "- Strengths (bullet list)\n"
                '- Issues (bullet list, or "None" if clean)\n'
                "- Verdict: SHIP or NEEDS WORK"
            ),
            "assignee": "reviewer",
            "dependsOn": ["Implement: URL shortener"],  # runs in parallel with Test
        },
    ]

    # -- Run -------------------------------------------------------------------

    print("Starting 4-stage task pipeline...\n")
    print("Pipeline: design -> implement -> test + review (parallel)")
    print("=" * 60)

    result = await orchestrator.run_tasks(team, tasks)

    # -- Summary ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Pipeline complete.\n")
    print(f"Overall success: {result.success}")
    print(
        f"Tokens -- input: {result.total_token_usage.input_tokens}, "
        f"output: {result.total_token_usage.output_tokens}"
    )

    print("\nPer-agent summary:")
    for name, r in result.agent_results.items():
        icon = "OK  " if r.success else "FAIL"
        tool_names = ", ".join(c.tool_name for c in r.tool_calls)
        print(f"  [{icon}] {name:<14}  tools used: {tool_names or '(none)'}")

    review = result.agent_results.get("reviewer")
    if review and review.success:
        print("\nCode review:")
        print("\u2500" * 60)
        print(review.output)
        print("\u2500" * 60)


if __name__ == "__main__":
    asyncio.run(main())
