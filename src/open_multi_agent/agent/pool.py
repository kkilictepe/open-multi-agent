"""Agent pool for managing and scheduling multiple agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import AgentRunResult, TokenUsage
from ..utils.semaphore import Semaphore

if TYPE_CHECKING:
    from .agent import Agent


@dataclass(frozen=True)
class PoolStatus:
    total: int
    idle: int
    running: int
    completed: int
    error: int


class AgentPool:
    """Registry and scheduler for a collection of Agent instances."""

    def __init__(self, max_concurrency: int = 5) -> None:
        self._agents: dict[str, Agent] = {}
        self._semaphore = Semaphore(max_concurrency)
        self._round_robin_index = 0

    # -- Registry operations ---------------------------------------------------

    def add(self, agent: Agent) -> None:
        if agent.name in self._agents:
            raise ValueError(
                f"AgentPool: agent '{agent.name}' is already registered. "
                f"Call remove('{agent.name}') before re-adding."
            )
        self._agents[agent.name] = agent

    def remove(self, name: str) -> None:
        if name not in self._agents:
            raise KeyError(f"AgentPool: agent '{name}' is not registered.")
        del self._agents[name]

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def list(self) -> list[Agent]:
        return list(self._agents.values())

    # -- Execution API ---------------------------------------------------------

    async def run(self, agent_name: str, prompt: str) -> AgentRunResult:
        agent = self._require_agent(agent_name)

        await self._semaphore.acquire()
        try:
            return await agent.run(prompt)
        finally:
            self._semaphore.release()

    async def run_parallel(
        self,
        tasks: list[dict[str, str]],
    ) -> dict[str, AgentRunResult]:
        result_map: dict[str, AgentRunResult] = {}

        async def _run_one(idx: int, task: dict[str, str]) -> None:
            try:
                result = await self.run(task["agent"], task["prompt"])
                result_map[task["agent"]] = result
            except Exception as err:
                result_map[task.get("agent", "unknown")] = self._error_result(err)

        results = await asyncio.gather(
            *[_run_one(i, t) for i, t in enumerate(tasks)],
            return_exceptions=True,
        )

        return result_map

    async def run_any(self, prompt: str) -> AgentRunResult:
        all_agents = self.list()
        if not all_agents:
            raise RuntimeError("AgentPool: cannot call run_any on an empty pool.")

        self._round_robin_index = self._round_robin_index % len(all_agents)
        agent = all_agents[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(all_agents)

        await self._semaphore.acquire()
        try:
            return await agent.run(prompt)
        finally:
            self._semaphore.release()

    # -- Observability ---------------------------------------------------------

    def get_status(self) -> PoolStatus:
        idle = running = completed = error = 0

        for agent in self._agents.values():
            status = agent.get_state().status
            if status == "idle":
                idle += 1
            elif status == "running":
                running += 1
            elif status == "completed":
                completed += 1
            elif status == "error":
                error += 1

        return PoolStatus(
            total=len(self._agents), idle=idle, running=running, completed=completed, error=error
        )

    # -- Lifecycle -------------------------------------------------------------

    async def shutdown(self) -> None:
        for agent in self._agents.values():
            agent.reset()

    # -- Private helpers -------------------------------------------------------

    def _require_agent(self, name: str) -> Agent:
        agent = self._agents.get(name)
        if agent is None:
            registered = ", ".join(self._agents.keys())
            raise KeyError(
                f"AgentPool: agent '{name}' is not registered. Registered agents: [{registered}]"
            )
        return agent

    @staticmethod
    def _error_result(reason: BaseException | str) -> AgentRunResult:
        message = str(reason)
        return AgentRunResult(
            success=False,
            output=message,
            messages=[],
            token_usage=TokenUsage(input_tokens=0, output_tokens=0),
            tool_calls=[],
        )
