"""In-memory implementation of MemoryStore."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..types import MemoryEntry, MemoryStore


class InMemoryStore(MemoryStore):
    """Synchronous-under-the-hood key/value store that exposes an async surface."""

    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}

    async def get(self, key: str) -> MemoryEntry | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        existing = self._data.get(key)
        entry = MemoryEntry(
            key=key,
            value=value,
            metadata=dict(metadata) if metadata is not None else None,
            created_at=existing.created_at if existing else datetime.now(),
        )
        self._data[key] = entry

    async def list(self) -> list[MemoryEntry]:
        return list(self._data.values())

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def clear(self) -> None:
        self._data.clear()

    # Extensions beyond the base MemoryStore interface

    async def search(self, query: str) -> list[MemoryEntry]:
        if not query:
            return await self.list()
        lower = query.lower()
        return [
            entry
            for entry in self._data.values()
            if lower in entry.key.lower() or lower in entry.value.lower()
        ]

    @property
    def size(self) -> int:
        return len(self._data)

    def has(self, key: str) -> bool:
        return key in self._data
