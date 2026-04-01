"""Team — central coordination object for a named group of agents."""

from __future__ import annotations

from typing import Any, Callable

from ..memory.shared import SharedMemory
from ..task.queue import TaskQueue
from ..task.task import create_task
from ..types import AgentConfig, MemoryStore, OrchestratorEvent, Task, TaskStatus, TeamConfig
from .messaging import Message, MessageBus


class _EventBus:
    """Minimal synchronous event emitter."""

    def __init__(self) -> None:
        self._listeners: dict[str, dict[int, Callable[[Any], None]]] = {}
        self._next_id = 0

    def on(self, event: str, handler: Callable[[Any], None]) -> Callable[[], None]:
        if event not in self._listeners:
            self._listeners[event] = {}
        sub_id = self._next_id
        self._next_id += 1
        self._listeners[event][sub_id] = handler

        def unsubscribe() -> None:
            self._listeners.get(event, {}).pop(sub_id, None)

        return unsubscribe

    def emit(self, event: str, data: Any) -> None:
        handlers = self._listeners.get(event, {})
        for handler in handlers.values():
            handler(data)


class Team:
    """Coordinates a named group of agents with shared messaging, task queuing,
    and optional shared memory."""

    def __init__(self, config: TeamConfig) -> None:
        self.config = config
        self.name = config.name

        self._agent_map: dict[str, AgentConfig] = {a.name: a for a in config.agents}
        self._bus = MessageBus()
        self._queue = TaskQueue()
        self._memory = SharedMemory() if config.shared_memory else None
        self._events = _EventBus()

        # Bridge queue events onto the team's event bus
        self._queue.on("task:ready", lambda task: self._events.emit(
            "task:ready",
            OrchestratorEvent(type="task_start", task=task.id, data=task),
        ))
        self._queue.on("task:complete", lambda task: self._events.emit(
            "task:complete",
            OrchestratorEvent(type="task_complete", task=task.id, data=task),
        ))
        self._queue.on("task:failed", lambda task: self._events.emit(
            "task:failed",
            OrchestratorEvent(type="error", task=task.id, data=task),
        ))
        self._queue.on("all:complete", lambda: self._events.emit("all:complete", None))

    # -- Agent roster ----------------------------------------------------------

    def get_agents(self) -> list[AgentConfig]:
        return list(self._agent_map.values())

    def get_agent(self, name: str) -> AgentConfig | None:
        return self._agent_map.get(name)

    # -- Messaging -------------------------------------------------------------

    def send_message(self, from_agent: str, to: str, content: str) -> None:
        message = self._bus.send(from_agent, to, content)
        event = OrchestratorEvent(type="message", agent=from_agent, data=message)
        self._events.emit("message", event)

    def get_messages(self, agent_name: str) -> list[Message]:
        return self._bus.get_all(agent_name)

    def broadcast(self, from_agent: str, content: str) -> None:
        message = self._bus.broadcast(from_agent, content)
        event = OrchestratorEvent(type="message", agent=from_agent, data=message)
        self._events.emit("broadcast", event)

    # -- Task management -------------------------------------------------------

    def add_task(
        self,
        *,
        title: str,
        description: str,
        status: TaskStatus = "pending",
        assignee: str | None = None,
        depends_on: list[str] | None = None,
        result: str | None = None,
    ) -> Task:
        created = create_task(
            title=title,
            description=description,
            assignee=assignee,
            depends_on=depends_on,
        )

        if status != "pending":
            created = created.model_copy(update={"status": status, "result": result})

        self._queue.add(created)
        return created

    def get_tasks(self) -> list[Task]:
        return self._queue.list()

    def get_tasks_by_assignee(self, agent_name: str) -> list[Task]:
        return [t for t in self._queue.list() if t.assignee == agent_name]

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        result: str | None = None,
        assignee: str | None = None,
    ) -> Task:
        return self._queue.update(task_id, status=status, result=result, assignee=assignee)

    def get_next_task(self, agent_name: str) -> Task | None:
        assigned = self._queue.next(agent_name)
        if assigned:
            return assigned
        return self._queue.next_available()

    # -- Memory ----------------------------------------------------------------

    def get_shared_memory(self) -> MemoryStore | None:
        return self._memory.get_store() if self._memory else None

    def get_shared_memory_instance(self) -> SharedMemory | None:
        return self._memory

    # -- Events ----------------------------------------------------------------

    def on(self, event: str, handler: Callable[[Any], None]) -> Callable[[], None]:
        return self._events.on(event, handler)

    def emit(self, event: str, data: Any) -> None:
        self._events.emit(event, data)

    # -- Expose internal queue for orchestrator --------------------------------

    @property
    def queue(self) -> TaskQueue:
        return self._queue
