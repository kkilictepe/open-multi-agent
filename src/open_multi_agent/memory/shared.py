"""Shared memory layer for teams of cooperating agents."""

from __future__ import annotations

from typing import Any

from ..types import MemoryEntry, MemoryStore
from .store import InMemoryStore


class SharedMemory:
    """Namespaced shared memory for a team of agents.

    Writes are namespaced as ``<agent_name>/<key>`` so that entries from different
    agents never collide and are always attributable.
    """

    def __init__(self) -> None:
        self._store = InMemoryStore()

    # -- Write -----------------------------------------------------------------

    async def write(
        self,
        agent_name: str,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        namespaced_key = self._namespace_key(agent_name, key)
        await self._store.set(namespaced_key, value, {**(metadata or {}), "agent": agent_name})

    # -- Read ------------------------------------------------------------------

    async def read(self, key: str) -> MemoryEntry | None:
        return await self._store.get(key)

    # -- List ------------------------------------------------------------------

    async def list_all(self) -> list[MemoryEntry]:
        return await self._store.list()

    async def list_by_agent(self, agent_name: str) -> list[MemoryEntry]:
        prefix = self._namespace_key(agent_name, "")
        all_entries = await self._store.list()
        return [e for e in all_entries if e.key.startswith(prefix)]

    # -- Summary ---------------------------------------------------------------

    async def get_summary(self) -> str:
        all_entries = await self._store.list()
        if not all_entries:
            return ""

        by_agent: dict[str, list[tuple[str, str]]] = {}
        for entry in all_entries:
            slash_idx = entry.key.find("/")
            if slash_idx == -1:
                agent = "_unknown"
                local_key = entry.key
            else:
                agent = entry.key[:slash_idx]
                local_key = entry.key[slash_idx + 1 :]

            by_agent.setdefault(agent, []).append((local_key, entry.value))

        lines: list[str] = ["## Shared Team Memory", ""]
        for agent, entries in by_agent.items():
            lines.append(f"### {agent}")
            for local_key, value in entries:
                display_value = f"{value[:197]}..." if len(value) > 200 else value
                lines.append(f"- {local_key}: {display_value}")
            lines.append("")

        return "\n".join(lines).rstrip()

    # -- Store access ----------------------------------------------------------

    def get_store(self) -> MemoryStore:
        return self._store

    # -- Private ---------------------------------------------------------------

    @staticmethod
    def _namespace_key(agent_name: str, key: str) -> str:
        return f"{agent_name}/{key}"
