"""Pure task utility functions."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from uuid import uuid4

from ..types import Task, TaskStatus


def create_task(
    *,
    title: str,
    description: str,
    assignee: str | None = None,
    depends_on: list[str] | None = None,
) -> Task:
    now = datetime.now()
    return Task(
        id=str(uuid4()),
        title=title,
        description=description,
        status="pending",
        assignee=assignee,
        depends_on=list(depends_on) if depends_on else None,
        result=None,
        created_at=now,
        updated_at=now,
    )


def is_task_ready(
    task: Task,
    all_tasks: list[Task],
    task_by_id: dict[str, Task] | None = None,
) -> bool:
    if task.status != "pending":
        return False
    if not task.depends_on:
        return True

    lookup = task_by_id if task_by_id is not None else {t.id: t for t in all_tasks}

    for dep_id in task.depends_on:
        dep = lookup.get(dep_id)
        if not dep or dep.status != "completed":
            return False

    return True


def get_task_dependency_order(tasks: list[Task]) -> list[Task]:
    """Topological sort via Kahn's algorithm."""
    if not tasks:
        return []

    task_by_id = {t.id: t for t in tasks}

    in_degree: dict[str, int] = {}
    successors: dict[str, list[str]] = {}

    for task in tasks:
        in_degree.setdefault(task.id, 0)
        successors.setdefault(task.id, [])

        for dep_id in task.depends_on or []:
            if dep_id in task_by_id:
                in_degree[task.id] = in_degree.get(task.id, 0) + 1
                successors.setdefault(dep_id, []).append(task.id)

    queue: deque[str] = deque()
    for task_id, degree in in_degree.items():
        if degree == 0:
            queue.append(task_id)

    ordered: list[Task] = []
    while queue:
        task_id = queue.popleft()
        task = task_by_id.get(task_id)
        if task:
            ordered.append(task)

        for successor_id in successors.get(task_id, []):
            in_degree[successor_id] -= 1
            if in_degree[successor_id] == 0:
                queue.append(successor_id)

    return ordered


def validate_task_dependencies(tasks: list[Task]) -> tuple[bool, list[str]]:
    """Validate the dependency graph. Returns (valid, errors)."""
    errors: list[str] = []
    task_by_id = {t.id: t for t in tasks}

    # Pass 1: unknown refs and self-deps
    for task in tasks:
        for dep_id in task.depends_on or []:
            if dep_id == task.id:
                errors.append(f'Task "{task.title}" ({task.id}) depends on itself.')
                continue
            if dep_id not in task_by_id:
                errors.append(
                    f'Task "{task.title}" ({task.id}) references unknown dependency "{dep_id}".'
                )

    # Pass 2: cycle detection via DFS colouring (0=white, 1=grey, 2=black)
    colour: dict[str, int] = {t.id: 0 for t in tasks}

    def visit(task_id: str, path: list[str]) -> None:
        if colour[task_id] == 2:
            return
        if colour[task_id] == 1:
            cycle_start = path.index(task_id)
            cycle = path[cycle_start:] + [task_id]
            errors.append(f"Cyclic dependency detected: {' -> '.join(cycle)}")
            return

        colour[task_id] = 1
        task = task_by_id.get(task_id)
        for dep_id in (task.depends_on or []) if task else []:
            if dep_id in task_by_id:
                visit(dep_id, [*path, task_id])
        colour[task_id] = 2

    for task in tasks:
        if colour[task.id] == 0:
            visit(task.id, [])

    return (len(errors) == 0, errors)
