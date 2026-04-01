"""OpenMultiAgent — the top-level multi-agent orchestration class."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..agent.agent import Agent
from ..agent.pool import AgentPool
from ..task.queue import TaskQueue
from ..task.task import create_task
from ..team.team import Team
from ..tool.built_in import register_built_in_tools
from ..tool.executor import ToolExecutor
from ..tool.framework import ToolRegistry
from ..types import (
    AgentConfig,
    AgentRunResult,
    OrchestratorConfig,
    OrchestratorEvent,
    Task,
    TeamConfig,
    TeamRunResult,
    TokenUsage,
)
from .scheduler import Scheduler

ZERO_USAGE = TokenUsage(input_tokens=0, output_tokens=0)
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_MODEL = "claude-opus-4-6"


def _build_agent(config: AgentConfig) -> Agent:
    registry = ToolRegistry()
    register_built_in_tools(registry)
    executor = ToolExecutor(registry)
    return Agent(config, registry, executor)


def _parse_task_specs(raw: str) -> list[dict[str, Any]] | None:
    fence_match = re.search(r"```json\s*([\s\S]*?)```", raw)
    candidate = fence_match.group(1) if fence_match else raw

    array_start = candidate.find("[")
    array_end = candidate.rfind("]")
    if array_start == -1 or array_end == -1 or array_end <= array_start:
        return None

    json_slice = candidate[array_start : array_end + 1]
    try:
        parsed = json.loads(json_slice)
        if not isinstance(parsed, list):
            return None

        specs: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get("title"), str):
                continue
            if not isinstance(item.get("description"), str):
                continue

            spec: dict[str, Any] = {
                "title": item["title"],
                "description": item["description"],
            }
            if isinstance(item.get("assignee"), str):
                spec["assignee"] = item["assignee"]
            if isinstance(item.get("dependsOn"), list):
                spec["depends_on"] = [x for x in item["dependsOn"] if isinstance(x, str)]

            specs.append(spec)

        return specs if specs else None
    except (json.JSONDecodeError, ValueError):
        return None


async def _build_task_prompt(task: Task, team: Team) -> str:
    lines = [f"# Task: {task.title}", "", task.description]

    shared_mem = team.get_shared_memory_instance()
    if shared_mem:
        summary = await shared_mem.get_summary()
        if summary:
            lines.extend(["", summary])

    if task.assignee:
        messages = team.get_messages(task.assignee)
        if messages:
            lines.extend(["", "## Messages from team members"])
            for msg in messages:
                lines.append(f"- **{msg.from_agent}**: {msg.content}")

    return "\n".join(lines)


async def _execute_queue(
    queue: TaskQueue,
    team: Team,
    pool: AgentPool,
    scheduler: Scheduler,
    agent_results: dict[str, AgentRunResult],
    config: OrchestratorConfig,
) -> None:
    while True:
        scheduler.auto_assign(queue, team.get_agents())

        pending = queue.get_by_status("pending")
        if not pending:
            break

        async def _dispatch_one(task: Task) -> None:
            queue.update(task.id, status="in_progress")

            assignee = task.assignee
            if not assignee:
                msg = f'Task "{task.title}" has no assignee.'
                queue.fail(task.id, msg)
                if config.on_progress:
                    config.on_progress(OrchestratorEvent(type="error", task=task.id, data=msg))
                return

            agent = pool.get(assignee)
            if not agent:
                msg = f'Agent "{assignee}" not found in pool for task "{task.title}".'
                queue.fail(task.id, msg)
                if config.on_progress:
                    config.on_progress(
                        OrchestratorEvent(type="error", task=task.id, agent=assignee, data=msg)
                    )
                return

            if config.on_progress:
                config.on_progress(
                    OrchestratorEvent(type="task_start", task=task.id, agent=assignee, data=task)
                )
                config.on_progress(
                    OrchestratorEvent(type="agent_start", agent=assignee, task=task.id, data=task)
                )

            prompt = await _build_task_prompt(task, team)

            try:
                result = await pool.run(assignee, prompt)
                agent_results[f"{assignee}:{task.id}"] = result

                if result.success:
                    shared_mem = team.get_shared_memory_instance()
                    if shared_mem:
                        await shared_mem.write(assignee, f"task:{task.id}:result", result.output)

                    queue.complete(task.id, result.output)

                    if config.on_progress:
                        config.on_progress(
                            OrchestratorEvent(
                                type="task_complete", task=task.id, agent=assignee, data=result
                            )
                        )
                        config.on_progress(
                            OrchestratorEvent(
                                type="agent_complete", agent=assignee, task=task.id, data=result
                            )
                        )
                else:
                    queue.fail(task.id, result.output)
                    if config.on_progress:
                        config.on_progress(
                            OrchestratorEvent(
                                type="error", task=task.id, agent=assignee, data=result
                            )
                        )
            except Exception as err:
                queue.fail(task.id, str(err))
                if config.on_progress:
                    config.on_progress(
                        OrchestratorEvent(type="error", task=task.id, agent=assignee, data=err)
                    )

        await asyncio.gather(*[_dispatch_one(task) for task in pending])


class OpenMultiAgent:
    """Top-level orchestrator for the open-multi-agent framework."""

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        cfg = config or OrchestratorConfig()
        self._max_concurrency = cfg.max_concurrency or DEFAULT_MAX_CONCURRENCY
        self._default_model = cfg.default_model or DEFAULT_MODEL
        self._default_provider = cfg.default_provider or "anthropic"
        self._on_progress = cfg.on_progress
        self._config = cfg

        self._teams: dict[str, Team] = {}
        self._completed_task_count = 0

    # -- Team management -------------------------------------------------------

    def create_team(self, name: str, config: TeamConfig) -> Team:
        if name in self._teams:
            raise ValueError(
                f'OpenMultiAgent: a team named "{name}" already exists. '
                "Use a unique name or call shutdown() to clear all teams."
            )
        team = Team(config)
        self._teams[name] = team
        return team

    # -- Single-agent convenience ----------------------------------------------

    async def run_agent(self, config: AgentConfig, prompt: str) -> AgentRunResult:
        agent = _build_agent(config)
        if self._on_progress:
            self._on_progress(
                OrchestratorEvent(type="agent_start", agent=config.name, data={"prompt": prompt})
            )

        result = await agent.run(prompt)

        if self._on_progress:
            self._on_progress(
                OrchestratorEvent(type="agent_complete", agent=config.name, data=result)
            )

        if result.success:
            self._completed_task_count += 1

        return result

    # -- Auto-orchestrated team run --------------------------------------------

    async def run_team(self, team: Team, goal: str) -> TeamRunResult:
        agent_configs = team.get_agents()

        # Step 1: Coordinator decomposes goal
        coordinator_config = AgentConfig(
            name="coordinator",
            model=self._default_model,
            provider=self._default_provider,
            system_prompt=self._build_coordinator_system_prompt(agent_configs),
            max_turns=3,
        )

        decomposition_prompt = self._build_decomposition_prompt(goal, agent_configs)
        coordinator_agent = _build_agent(coordinator_config)

        if self._on_progress:
            self._on_progress(
                OrchestratorEvent(
                    type="agent_start",
                    agent="coordinator",
                    data={"phase": "decomposition", "goal": goal},
                )
            )

        decomposition_result = await coordinator_agent.run(decomposition_prompt)
        agent_results: dict[str, AgentRunResult] = {"coordinator:decompose": decomposition_result}

        # Step 2: Parse tasks
        task_specs = _parse_task_specs(decomposition_result.output)

        queue = TaskQueue()
        scheduler = Scheduler("dependency-first")

        if task_specs:
            self._load_specs_into_queue(task_specs, agent_configs, queue)
        else:
            for agent_config in agent_configs:
                task = create_task(
                    title=f"{agent_config.name}: {goal[:80]}",
                    description=goal,
                    assignee=agent_config.name,
                )
                queue.add(task)

        # Step 3: Auto-assign
        scheduler.auto_assign(queue, agent_configs)

        # Step 4: Build pool and execute
        pool = self._build_pool(agent_configs)
        await _execute_queue(queue, team, pool, scheduler, agent_results, self._config)

        # Step 5: Coordinator synthesises final result
        synthesis_prompt = await self._build_synthesis_prompt(goal, queue.list(), team)
        synthesis_result = await coordinator_agent.run(synthesis_prompt)
        agent_results["coordinator"] = synthesis_result

        if self._on_progress:
            self._on_progress(
                OrchestratorEvent(type="agent_complete", agent="coordinator", data=synthesis_result)
            )

        return self._build_team_run_result(agent_results)

    # -- Explicit-task team run ------------------------------------------------

    async def run_tasks(
        self,
        team: Team,
        tasks: list[dict[str, Any]],
    ) -> TeamRunResult:
        agent_configs = team.get_agents()
        queue = TaskQueue()
        scheduler = Scheduler("dependency-first")

        specs = [
            {
                "title": t["title"],
                "description": t["description"],
                "assignee": t.get("assignee"),
                "depends_on": t.get("dependsOn") or t.get("depends_on"),
            }
            for t in tasks
        ]
        self._load_specs_into_queue(specs, agent_configs, queue)
        scheduler.auto_assign(queue, agent_configs)

        pool = self._build_pool(agent_configs)
        agent_results: dict[str, AgentRunResult] = {}

        await _execute_queue(queue, team, pool, scheduler, agent_results, self._config)

        return self._build_team_run_result(agent_results)

    # -- Observability ---------------------------------------------------------

    def get_status(self) -> dict[str, int]:
        return {
            "teams": len(self._teams),
            "active_agents": 0,
            "completed_tasks": self._completed_task_count,
        }

    # -- Lifecycle -------------------------------------------------------------

    async def shutdown(self) -> None:
        self._teams.clear()
        self._completed_task_count = 0

    # -- Private helpers -------------------------------------------------------

    def _build_coordinator_system_prompt(self, agents: list[AgentConfig]) -> str:
        roster = "\n".join(
            f"- **{a.name}** ({a.model}): {(a.system_prompt or 'general purpose agent')[:120]}"
            for a in agents
        )

        return "\n".join([
            "You are a task coordinator responsible for decomposing high-level goals",
            "into concrete, actionable tasks and assigning them to the right team members.",
            "",
            "## Team Roster",
            roster,
            "",
            "## Output Format",
            "When asked to decompose a goal, respond ONLY with a JSON array of task objects.",
            "Each task must have:",
            '  - "title":       Short descriptive title (string)',
            '  - "description": Full task description with context and expected output (string)',
            '  - "assignee":    One of the agent names listed in the roster (string)',
            '  - "dependsOn":   Array of titles of tasks this task depends on (string[], may be empty)',
            "",
            "Wrap the JSON in a ```json code fence.",
            "Do not include any text outside the code fence.",
            "",
            "## When synthesising results",
            "You will be given completed task outputs and asked to synthesise a final answer.",
            "Write a clear, comprehensive response that addresses the original goal.",
        ])

    def _build_decomposition_prompt(self, goal: str, agents: list[AgentConfig]) -> str:
        names = ", ".join(a.name for a in agents)
        return "\n".join([
            f"Decompose the following goal into tasks for your team ({names}).",
            "",
            "## Goal",
            goal,
            "",
            "Return ONLY the JSON task array in a ```json code fence.",
        ])

    async def _build_synthesis_prompt(
        self, goal: str, tasks: list[Task], team: Team
    ) -> str:
        completed_tasks = [t for t in tasks if t.status == "completed"]
        failed_tasks = [t for t in tasks if t.status == "failed"]

        result_sections = [
            f"### {t.title} (completed by {t.assignee or 'unknown'})\n{t.result or '(no output)'}"
            for t in completed_tasks
        ]

        failure_sections = [
            f"### {t.title} (FAILED)\nError: {t.result or 'unknown error'}" for t in failed_tasks
        ]

        memory_summary = ""
        shared_mem = team.get_shared_memory_instance()
        if shared_mem:
            memory_summary = await shared_mem.get_summary()

        lines = [
            "## Original Goal",
            goal,
            "",
            "## Task Results",
            *result_sections,
        ]
        if failure_sections:
            lines.extend(["", "## Failed Tasks", *failure_sections])
        if memory_summary:
            lines.extend(["", memory_summary])
        lines.extend([
            "",
            "## Your Task",
            "Synthesise the above results into a comprehensive final answer that addresses the original goal.",
            "If some tasks failed, note any gaps in the result.",
        ])

        return "\n".join(lines)

    def _load_specs_into_queue(
        self,
        specs: list[dict[str, Any]],
        agent_configs: list[AgentConfig],
        queue: TaskQueue,
    ) -> None:
        agent_names = {a.name for a in agent_configs}

        # First pass: create tasks to get stable IDs
        title_to_id: dict[str, str] = {}
        created_tasks: list[Task] = []

        for spec in specs:
            task = create_task(
                title=spec["title"],
                description=spec["description"],
                assignee=spec.get("assignee") if spec.get("assignee") in agent_names else None,
            )
            title_to_id[spec["title"].lower().strip()] = task.id
            created_tasks.append(task)

        # Second pass: resolve title-based dependsOn to IDs
        for i, task in enumerate(created_tasks):
            spec = specs[i]
            dep_refs = spec.get("depends_on") or []

            if not dep_refs:
                queue.add(task)
                continue

            resolved_deps: list[str] = []
            for dep_ref in dep_refs:
                by_id = next((t for t in created_tasks if t.id == dep_ref), None)
                by_title = title_to_id.get(dep_ref.lower().strip())
                resolved_id = (by_id.id if by_id else None) or by_title
                if resolved_id:
                    resolved_deps.append(resolved_id)

            if resolved_deps:
                task = task.model_copy(update={"depends_on": resolved_deps})

            queue.add(task)

    def _build_pool(self, agent_configs: list[AgentConfig]) -> AgentPool:
        pool = AgentPool(self._max_concurrency)
        for config in agent_configs:
            effective = config.model_copy(
                update={"provider": config.provider or self._default_provider}
            )
            pool.add(_build_agent(effective))
        return pool

    def _build_team_run_result(self, agent_results: dict[str, AgentRunResult]) -> TeamRunResult:
        total_usage = ZERO_USAGE
        overall_success = True
        collapsed: dict[str, AgentRunResult] = {}

        for key, result in agent_results.items():
            agent_name = key.split(":")[0] if ":" in key else key

            total_usage = total_usage + result.token_usage
            if not result.success:
                overall_success = False

            existing = collapsed.get(agent_name)
            if not existing:
                collapsed[agent_name] = result
            else:
                collapsed[agent_name] = AgentRunResult(
                    success=existing.success and result.success,
                    output="\n\n---\n\n".join(
                        filter(None, [existing.output, result.output])
                    ),
                    messages=[*existing.messages, *result.messages],
                    token_usage=existing.token_usage + result.token_usage,
                    tool_calls=[*existing.tool_calls, *result.tool_calls],
                )

            if result.success and not key.startswith("coordinator"):
                self._completed_task_count += 1

        return TeamRunResult(
            success=overall_success,
            agent_results=collapsed,
            total_token_usage=total_usage,
        )
