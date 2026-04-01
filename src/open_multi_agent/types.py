"""Core type definitions for the open-multi-agent orchestration framework.

All public types are exported from this single module. Downstream modules
import only what they need, keeping the dependency graph acyclic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterable, Awaitable, Callable, Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool | None = None


class ImageBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["image"] = "image"
    source: ImageSource


class ImageSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock]

# ---------------------------------------------------------------------------
# LLM messages & responses
# ---------------------------------------------------------------------------


class LLMMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    content: list[ContentBlock]
    model: str
    stop_reason: str
    usage: TokenUsage


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["text", "tool_use", "tool_result", "done", "error"]
    data: Any = None


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class LLMToolDef(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    input_schema: dict[str, Any] = Field(alias="inputSchema", default_factory=dict)

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: str
    is_error: bool | None = None


class AgentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    role: str
    model: str


class TeamInfo(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    agents: list[str]
    shared_memory: Any  # MemoryStore instance


class ToolUseContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    agent: AgentInfo
    team: TeamInfo | None = None
    cancel_event: asyncio.Event | None = None
    cwd: str | None = None
    metadata: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    description: str
    input_schema: type[BaseModel]
    execute: Callable[..., Awaitable[ToolResult]]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SupportedProvider = Literal["anthropic", "openai"]


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    model: str
    provider: SupportedProvider | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class AgentState(BaseModel):
    status: Literal["idle", "running", "completed", "error"] = "idle"
    messages: list[LLMMessage] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_name: str
    input: dict[str, Any]
    output: str
    duration: float  # milliseconds


class AgentRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    success: bool
    output: str
    messages: list[LLMMessage]
    token_usage: TokenUsage
    tool_calls: list[ToolCallRecord]


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------


class TeamConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    agents: list[AgentConfig]
    shared_memory: bool | None = None
    max_concurrency: int | None = None


class TeamRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    success: bool
    agent_results: dict[str, AgentRunResult]
    total_token_usage: TokenUsage


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

TaskStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]


class Task(BaseModel):
    id: str
    title: str
    description: str
    status: TaskStatus = "pending"
    assignee: str | None = None
    depends_on: list[str] | None = None
    result: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

OrchestratorEventType = Literal[
    "agent_start", "agent_complete", "task_start", "task_complete", "message", "error"
]


class OrchestratorEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: OrchestratorEventType
    agent: str | None = None
    task: str | None = None
    data: Any = None


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_concurrency: int | None = None
    default_model: str | None = None
    default_provider: SupportedProvider | None = None
    on_progress: Callable[[OrchestratorEvent], None] | None = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str
    value: str
    metadata: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryStore(ABC):
    @abstractmethod
    async def get(self, key: str) -> MemoryEntry | None: ...

    @abstractmethod
    async def set(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    async def list(self) -> list[MemoryEntry]: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------


class LLMChatOptions(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    model: str
    tools: list[LLMToolDef] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    cancel_event: asyncio.Event | None = None


class LLMStreamOptions(LLMChatOptions):
    pass


class LLMAdapter(Protocol):
    @property
    def name(self) -> str: ...

    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse: ...

    def stream(
        self, messages: list[LLMMessage], options: LLMStreamOptions
    ) -> AsyncIterable[StreamEvent]: ...
