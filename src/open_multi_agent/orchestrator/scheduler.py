"""Task scheduling strategies for the open-multi-agent orchestrator."""

from __future__ import annotations

import re
from collections import deque
from typing import Literal

from ..task.queue import TaskQueue
from ..types import AgentConfig, Task

SchedulingStrategy = Literal["round-robin", "least-busy", "capability-match", "dependency-first"]

STOP_WORDS = frozenset([
    "the", "and", "for", "that", "this", "with", "are", "from", "have",
    "will", "your", "you", "can", "all", "each", "when", "then", "they",
    "them", "their", "about", "into", "more", "also", "should", "must",
])


def _count_blocked_dependents(task_id: str, all_tasks: list[Task]) -> int:
    dependents: dict[str, list[str]] = {}
    id_set = {t.id for t in all_tasks}
    for t in all_tasks:
        for dep_id in t.depends_on or []:
            dependents.setdefault(dep_id, []).append(t.id)

    visited: set[str] = set()
    queue: deque[str] = deque([task_id])
    while queue:
        current = queue.popleft()
        for dep_id in dependents.get(current, []):
            if dep_id not in visited and dep_id in id_set:
                visited.add(dep_id)
                queue.append(dep_id)

    return len(visited)


def _keyword_score(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def _extract_keywords(text: str) -> list[str]:
    words = re.split(r"\W+", text.lower())
    return list({w for w in words if len(w) > 3 and w not in STOP_WORDS})


class Scheduler:
    """Maps pending tasks to available agents using configurable strategies."""

    def __init__(self, strategy: SchedulingStrategy = "dependency-first") -> None:
        self._strategy = strategy
        self._round_robin_cursor = 0

    def schedule(self, tasks: list[Task], agents: list[AgentConfig]) -> dict[str, str]:
        if not agents:
            return {}

        unassigned = [t for t in tasks if t.status == "pending" and not t.assignee]

        match self._strategy:
            case "round-robin":
                return self._schedule_round_robin(unassigned, agents)
            case "least-busy":
                return self._schedule_least_busy(unassigned, agents, tasks)
            case "capability-match":
                return self._schedule_capability_match(unassigned, agents)
            case "dependency-first":
                return self._schedule_dependency_first(unassigned, agents, tasks)

    def auto_assign(self, queue: TaskQueue, agents: list[AgentConfig]) -> None:
        all_tasks = queue.list()
        assignments = self.schedule(all_tasks, agents)

        for task_id, agent_name in assignments.items():
            try:
                queue.update(task_id, assignee=agent_name)
            except (KeyError, Exception):
                pass

    def _schedule_round_robin(
        self, unassigned: list[Task], agents: list[AgentConfig]
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for task in unassigned:
            agent = agents[self._round_robin_cursor % len(agents)]
            result[task.id] = agent.name
            self._round_robin_cursor = (self._round_robin_cursor + 1) % len(agents)
        return result

    def _schedule_least_busy(
        self, unassigned: list[Task], agents: list[AgentConfig], all_tasks: list[Task]
    ) -> dict[str, str]:
        load: dict[str, int] = {a.name: 0 for a in agents}
        for task in all_tasks:
            if task.status == "in_progress" and task.assignee:
                load[task.assignee] = load.get(task.assignee, 0) + 1

        result: dict[str, str] = {}
        for task in unassigned:
            best_agent = agents[0]
            best_load = load.get(best_agent.name, 0)

            for agent in agents[1:]:
                agent_load = load.get(agent.name, 0)
                if agent_load < best_load:
                    best_load = agent_load
                    best_agent = agent

            result[task.id] = best_agent.name
            load[best_agent.name] = load.get(best_agent.name, 0) + 1

        return result

    def _schedule_capability_match(
        self, unassigned: list[Task], agents: list[AgentConfig]
    ) -> dict[str, str]:
        agent_keywords = {
            a.name: _extract_keywords(f"{a.name} {a.system_prompt or ''} {a.model}")
            for a in agents
        }

        result: dict[str, str] = {}
        for task in unassigned:
            task_text = f"{task.title} {task.description}"
            task_keywords = _extract_keywords(task_text)

            best_agent = agents[0]
            best_score = -1

            for agent in agents:
                agent_text = f"{agent.name} {agent.system_prompt or ''}"
                score_a = _keyword_score(agent_text, task_keywords)
                score_b = _keyword_score(task_text, agent_keywords.get(agent.name, []))
                score = score_a + score_b

                if score > best_score:
                    best_score = score
                    best_agent = agent

            result[task.id] = best_agent.name

        return result

    def _schedule_dependency_first(
        self, unassigned: list[Task], agents: list[AgentConfig], all_tasks: list[Task]
    ) -> dict[str, str]:
        ranked = sorted(
            unassigned,
            key=lambda t: _count_blocked_dependents(t.id, all_tasks),
            reverse=True,
        )

        result: dict[str, str] = {}
        cursor = self._round_robin_cursor

        for task in ranked:
            agent = agents[cursor % len(agents)]
            result[task.id] = agent.name
            cursor = (cursor + 1) % len(agents)

        self._round_robin_cursor = cursor
        return result
