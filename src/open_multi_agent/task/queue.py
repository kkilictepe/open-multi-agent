"""Dependency-aware task queue with event-driven dependency resolution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Literal

from ..types import Task, TaskStatus
from .task import is_task_ready

TaskQueueEvent = Literal["task:ready", "task:complete", "task:failed", "all:complete"]


class TaskQueue:
    """Mutable, event-driven queue with topological dependency resolution."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._listeners: dict[str, dict[int, Callable[..., Any]]] = {}
        self._next_id = 0

    # -- Mutation: add ----------------------------------------------------------

    def add(self, task: Task) -> None:
        resolved = self._resolve_initial_status(task)
        self._tasks[resolved.id] = resolved
        if resolved.status == "pending":
            self._emit("task:ready", resolved)

    def add_batch(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.add(task)

    # -- Mutation: update / complete / fail ------------------------------------

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        result: str | None = None,
        assignee: str | None = None,
    ) -> Task:
        task = self._require_task(task_id)
        updated = task.model_copy(
            update={
                **({"status": status} if status is not None else {}),
                **({"result": result} if result is not None else {}),
                **({"assignee": assignee} if assignee is not None else {}),
                "updated_at": datetime.now(),
            }
        )
        self._tasks[task_id] = updated
        return updated

    def complete(self, task_id: str, result: str | None = None) -> Task:
        completed = self.update(task_id, status="completed", result=result)
        self._emit("task:complete", completed)
        self._unblock_dependents(task_id)
        if self.is_complete():
            self._emit_all_complete()
        return completed

    def fail(self, task_id: str, error: str) -> Task:
        failed = self.update(task_id, status="failed", result=error)
        self._emit("task:failed", failed)
        self._cascade_failure(task_id)
        if self.is_complete():
            self._emit_all_complete()
        return failed

    def _cascade_failure(self, failed_task_id: str) -> None:
        for task in list(self._tasks.values()):
            if task.status not in ("blocked", "pending"):
                continue
            if not task.depends_on or failed_task_id not in task.depends_on:
                continue

            cascaded = self.update(
                task.id,
                status="failed",
                result=f'Cancelled: dependency "{failed_task_id}" failed.',
            )
            self._emit("task:failed", cascaded)
            self._cascade_failure(task.id)

    # -- Queries ---------------------------------------------------------------

    def next(self, assignee: str | None = None) -> Task | None:
        if assignee is None:
            return self.next_available()

        for task in self._tasks.values():
            if task.status == "pending" and task.assignee == assignee:
                return task
        return None

    def next_available(self) -> Task | None:
        fallback: Task | None = None

        for task in self._tasks.values():
            if task.status != "pending":
                continue
            if not task.assignee:
                return task
            if fallback is None:
                fallback = task

        return fallback

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_status(self, status: TaskStatus) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == status]

    def is_complete(self) -> bool:
        return all(t.status in ("completed", "failed") for t in self._tasks.values())

    def get_progress(self) -> dict[str, int]:
        counts = {"total": 0, "completed": 0, "failed": 0, "in_progress": 0, "pending": 0, "blocked": 0}
        for task in self._tasks.values():
            counts["total"] += 1
            if task.status == "completed":
                counts["completed"] += 1
            elif task.status == "failed":
                counts["failed"] += 1
            elif task.status == "in_progress":
                counts["in_progress"] += 1
            elif task.status == "pending":
                counts["pending"] += 1
            elif task.status == "blocked":
                counts["blocked"] += 1
        return counts

    # -- Events ----------------------------------------------------------------

    def on(self, event: TaskQueueEvent, handler: Callable[..., Any]) -> Callable[[], None]:
        if event not in self._listeners:
            self._listeners[event] = {}
        sub_id = self._next_id
        self._next_id += 1
        self._listeners[event][sub_id] = handler

        def unsubscribe() -> None:
            self._listeners.get(event, {}).pop(sub_id, None)

        return unsubscribe

    # -- Private helpers -------------------------------------------------------

    def _resolve_initial_status(self, task: Task) -> Task:
        if not task.depends_on:
            return task

        all_current = list(self._tasks.values())
        if is_task_ready(task, all_current):
            return task

        return task.model_copy(update={"status": "blocked", "updated_at": datetime.now()})

    def _unblock_dependents(self, completed_id: str) -> None:
        all_tasks = list(self._tasks.values())
        task_by_id = {t.id: t for t in all_tasks}

        for task in all_tasks:
            if task.status != "blocked":
                continue
            if not task.depends_on or completed_id not in task.depends_on:
                continue

            if is_task_ready(task, all_tasks, task_by_id):
                unblocked = task.model_copy(
                    update={"status": "pending", "updated_at": datetime.now()}
                )
                self._tasks[task.id] = unblocked
                task_by_id[task.id] = unblocked
                self._emit("task:ready", unblocked)

    def _emit(self, event: str, task: Task) -> None:
        handlers = self._listeners.get(event, {})
        for handler in handlers.values():
            handler(task)

    def _emit_all_complete(self) -> None:
        handlers = self._listeners.get("all:complete", {})
        for handler in handlers.values():
            handler()

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f'TaskQueue: task "{task_id}" not found.')
        return task
