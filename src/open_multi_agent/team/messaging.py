"""Inter-agent message bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4


@dataclass(frozen=True)
class Message:
    id: str
    from_agent: str  # 'from' is a reserved keyword in Python
    to: str
    content: str
    timestamp: datetime


class MessageBus:
    """In-memory message bus for inter-agent communication."""

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._read_state: dict[str, set[str]] = {}
        self._subscribers: dict[str, dict[int, Callable[[Message], None]]] = {}
        self._next_id = 0

    # -- Write -----------------------------------------------------------------

    def send(self, from_agent: str, to: str, content: str) -> Message:
        message = Message(
            id=str(uuid4()),
            from_agent=from_agent,
            to=to,
            content=content,
            timestamp=datetime.now(),
        )
        self._persist(message)
        return message

    def broadcast(self, from_agent: str, content: str) -> Message:
        return self.send(from_agent, "*", content)

    # -- Read ------------------------------------------------------------------

    def get_unread(self, agent_name: str) -> list[Message]:
        read = self._read_state.get(agent_name, set())
        return [m for m in self._messages if self._is_addressed_to(m, agent_name) and m.id not in read]

    def get_all(self, agent_name: str) -> list[Message]:
        return [m for m in self._messages if self._is_addressed_to(m, agent_name)]

    def mark_read(self, agent_name: str, message_ids: list[str]) -> None:
        if not message_ids:
            return
        read = self._read_state.setdefault(agent_name, set())
        read.update(message_ids)

    def get_conversation(self, agent1: str, agent2: str) -> list[Message]:
        return [
            m
            for m in self._messages
            if (m.from_agent == agent1 and m.to == agent2)
            or (m.from_agent == agent2 and m.to == agent1)
        ]

    # -- Subscriptions ---------------------------------------------------------

    def subscribe(self, agent_name: str, callback: Callable[[Message], None]) -> Callable[[], None]:
        if agent_name not in self._subscribers:
            self._subscribers[agent_name] = {}
        sub_id = self._next_id
        self._next_id += 1
        self._subscribers[agent_name][sub_id] = callback

        def unsubscribe() -> None:
            self._subscribers.get(agent_name, {}).pop(sub_id, None)

        return unsubscribe

    # -- Private ---------------------------------------------------------------

    @staticmethod
    def _is_addressed_to(message: Message, agent_name: str) -> bool:
        if message.to == "*":
            return message.from_agent != agent_name
        return message.to == agent_name

    def _persist(self, message: Message) -> None:
        self._messages.append(message)
        self._notify_subscribers(message)

    def _notify_subscribers(self, message: Message) -> None:
        if message.to != "*":
            self._fire_callbacks(message.to, message)
            return

        for agent_name, subs in self._subscribers.items():
            if agent_name != message.from_agent and subs:
                self._fire_callbacks(agent_name, message)

    def _fire_callbacks(self, agent_name: str, message: Message) -> None:
        subs = self._subscribers.get(agent_name, {})
        for callback in subs.values():
            callback(message)
