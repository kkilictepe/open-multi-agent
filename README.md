
# Open Multi-Agent (Python)

Build AI agent teams that work together. One agent plans, another implements, a third reviews — the framework handles task scheduling, dependencies, and communication automatically.

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-green)](https://docs.pydantic.dev/)
[![license](https://img.shields.io/badge/license-MIT-brightgreen)](./LICENSE)

> **Note:** This package is a faithful Python conversion of the TypeScript [open-multi-agent](https://github.com/JackChen-me/open-multi-agent) framework by JackChen-me. The architecture, features, and API surface have been preserved 1:1, adapted to Python idioms (Pydantic models, asyncio, snake_case).

## Why Open Multi-Agent?

- **Multi-Agent Teams** — Define agents with different roles, tools, and even different models. They collaborate through a message bus and shared memory.
- **Task DAG Scheduling** — Tasks have dependencies. The framework resolves them topologically — dependent tasks wait, independent tasks run in parallel.
- **Model Agnostic** — Claude and GPT in the same team. Swap models per agent. Bring your own adapter for any LLM.
- **In-Process Execution** — No subprocess overhead. Everything runs in one Python process with `asyncio`. Deploy to serverless, Docker, CI/CD.

## Quick Start

```bash
pip install open-multi-agent
```

Set `ANTHROPIC_API_KEY` (and optionally `OPENAI_API_KEY`) in your environment.

```python
import asyncio
from open_multi_agent import OpenMultiAgent, AgentConfig, OrchestratorConfig

async def main():
    orchestrator = OpenMultiAgent(OrchestratorConfig(default_model="claude-sonnet-4-6"))

    # One agent, one task
    result = await orchestrator.run_agent(
        AgentConfig(
            name="coder",
            model="claude-sonnet-4-6",
            tools=["bash", "file_write"],
        ),
        "Write a Python function that reverses a string, save it to /tmp/reverse.py, and run it.",
    )

    print(result.output)

asyncio.run(main())
```

## Multi-Agent Team

This is where it gets interesting. Three agents, one goal:

```python
import asyncio
from open_multi_agent import OpenMultiAgent, AgentConfig, OrchestratorConfig, TeamConfig

architect = AgentConfig(
    name="architect",
    model="claude-sonnet-4-6",
    system_prompt="You design clean API contracts and file structures.",
    tools=["file_write"],
)

developer = AgentConfig(
    name="developer",
    model="claude-sonnet-4-6",
    system_prompt="You implement what the architect designs.",
    tools=["bash", "file_read", "file_write", "file_edit"],
)

reviewer = AgentConfig(
    name="reviewer",
    model="claude-sonnet-4-6",
    system_prompt="You review code for correctness and clarity.",
    tools=["file_read", "grep"],
)

async def main():
    orchestrator = OpenMultiAgent(
        OrchestratorConfig(
            default_model="claude-sonnet-4-6",
            on_progress=lambda event: print(event.type, event.agent or event.task or ""),
        )
    )

    team = orchestrator.create_team("api-team", TeamConfig(
        name="api-team",
        agents=[architect, developer, reviewer],
        shared_memory=True,
    ))

    # Describe a goal — the framework breaks it into tasks and orchestrates execution
    result = await orchestrator.run_team(team, "Create a REST API for a todo list in /tmp/todo-api/")

    print(f"Success: {result.success}")
    print(f"Tokens: {result.total_token_usage.output_tokens} output tokens")

asyncio.run(main())
```

## More Examples

<details>
<summary><b>Task Pipeline</b> — explicit control over task graph and assignments</summary>

```python
result = await orchestrator.run_tasks(team, [
    {
        "title": "Design the data model",
        "description": "Write a spec to /tmp/spec.md",
        "assignee": "architect",
    },
    {
        "title": "Implement the module",
        "description": "Read /tmp/spec.md and implement in /tmp/src/",
        "assignee": "developer",
        "dependsOn": ["Design the data model"],  # blocked until design completes
    },
    {
        "title": "Write tests",
        "description": "Read the implementation and write pytest tests.",
        "assignee": "developer",
        "dependsOn": ["Implement the module"],
    },
    {
        "title": "Review code",
        "description": "Review /tmp/src/ and produce a structured code review.",
        "assignee": "reviewer",
        "dependsOn": ["Implement the module"],  # can run in parallel with tests
    },
])
```

</details>

<details>
<summary><b>Custom Tools</b> — define tools with Pydantic models</summary>

```python
from pydantic import BaseModel, Field
from open_multi_agent import (
    Agent, AgentConfig, ToolRegistry, ToolExecutor,
    define_tool, register_built_in_tools,
)

class SearchInput(BaseModel):
    query: str = Field(description="The search query.")
    max_results: int = Field(default=5, description="Number of results.")

async def search_handler(params: SearchInput, context):
    results = await my_search_provider(params.query, params.max_results)
    return {"data": json.dumps(results), "isError": False}

search_tool = define_tool(
    name="web_search",
    description="Search the web and return the top results.",
    input_model=SearchInput,
    handler=search_handler,
)

registry = ToolRegistry()
register_built_in_tools(registry)
registry.register(search_tool)

executor = ToolExecutor(registry)
agent = Agent(
    AgentConfig(name="researcher", model="claude-sonnet-4-6", tools=["web_search"]),
    registry,
    executor,
)

result = await agent.run("Find the three most recent Python releases.")
```

</details>

<details>
<summary><b>Multi-Model Teams</b> — mix Claude and GPT in one workflow</summary>

```python
claude_agent = AgentConfig(
    name="strategist",
    model="claude-opus-4-6",
    provider="anthropic",
    system_prompt="You plan high-level approaches.",
    tools=["file_write"],
)

gpt_agent = AgentConfig(
    name="implementer",
    model="gpt-4o",
    provider="openai",
    system_prompt="You implement plans as working code.",
    tools=["bash", "file_read", "file_write"],
)

team = orchestrator.create_team("mixed-team", TeamConfig(
    name="mixed-team",
    agents=[claude_agent, gpt_agent],
    shared_memory=True,
))

result = await orchestrator.run_team(team, "Build a CLI tool that converts JSON to CSV.")
```

</details>

<details>
<summary><b>Streaming Output</b></summary>

```python
import sys
from open_multi_agent import Agent, AgentConfig, ToolRegistry, ToolExecutor, register_built_in_tools

registry = ToolRegistry()
register_built_in_tools(registry)
executor = ToolExecutor(registry)

agent = Agent(
    AgentConfig(name="writer", model="claude-sonnet-4-6", max_turns=3),
    registry,
    executor,
)

async for event in agent.stream("Explain monads in two sentences."):
    if event.type == "text" and isinstance(event.data, str):
        sys.stdout.write(event.data)
```

</details>

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OpenMultiAgent (Orchestrator)                                  │
│                                                                 │
│  create_team()  run_team()  run_tasks()  run_agent()            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Team               │
            │  - AgentConfig[]    │
            │  - MessageBus       │
            │  - TaskQueue        │
            │  - SharedMemory     │
            └──────────┬──────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
┌────────▼──────────┐    ┌───────────▼───────────┐
│  AgentPool        │    │  TaskQueue             │
│  - Semaphore      │    │  - dependency graph    │
│  - run_parallel() │    │  - auto unblock        │
└────────┬──────────┘    │  - cascade failure     │
         │               └───────────────────────┘
┌────────▼──────────┐
│  Agent            │
│  - run()          │    ┌──────────────────────┐
│  - prompt()       │───►│  LLMAdapter          │
│  - stream()       │    │  - AnthropicAdapter  │
└────────┬──────────┘    │  - OpenAIAdapter     │
         │               └──────────────────────┘
┌────────▼──────────┐
│  AgentRunner      │    ┌──────────────────────┐
│  - conversation   │───►│  ToolRegistry        │
│    loop           │    │  - define_tool()     │
│  - tool dispatch  │    │  - 5 built-in tools  │
└───────────────────┘    └──────────────────────┘
```

## Built-in Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands. Returns stdout + stderr. Supports timeout and cwd. |
| `file_read` | Read file contents at an absolute path. Supports offset/limit for large files. |
| `file_write` | Write or create a file. Auto-creates parent directories. |
| `file_edit` | Edit a file by replacing an exact string match. |
| `grep` | Search file contents with regex. Uses ripgrep when available, falls back to Python. |

## Key Differences from TypeScript Version

| TypeScript | Python |
|---|---|
| Zod schemas for tool inputs | Pydantic `BaseModel` + `model_json_schema()` |
| `Promise.all` / `Promise.allSettled` | `asyncio.gather` / `asyncio.gather(return_exceptions=True)` |
| `AbortSignal` | `asyncio.Event` (cooperative cancellation) |
| EventEmitter | Dict-based event subscriptions with unsubscribe closures |
| `camelCase` API | `snake_case` API |
| `child_process.spawn` | `asyncio.create_subprocess_exec` |
| `fs/promises` | `pathlib.Path` + `asyncio.to_thread()` |


## Credits

This project is a Python port of [open-multi-agent](https://github.com/JackChen-me/open-multi-agent) by [JackChen-me](https://github.com/JackChen-me). All credit for the original architecture, design, and framework goes to the original author.

## License

MIT
