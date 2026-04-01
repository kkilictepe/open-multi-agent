"""open-multi-agent -- public API surface.

Import from ``open_multi_agent`` to access everything you need::

    from open_multi_agent import OpenMultiAgent, Agent, Team, define_tool

Quickstart -- single agent::

    import asyncio
    from open_multi_agent import OpenMultiAgent

    orchestrator = OpenMultiAgent()
    result = asyncio.run(orchestrator.run_agent(
        {"name": "assistant", "model": "claude-opus-4-6"},
        "Explain monads in one paragraph.",
    ))
    print(result.output)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Orchestrator (primary entry point)
# ---------------------------------------------------------------------------

from .orchestrator.orchestrator import OpenMultiAgent
from .orchestrator.scheduler import Scheduler, SchedulingStrategy

# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------

from .agent.agent import Agent
from .agent.pool import AgentPool, PoolStatus

# ---------------------------------------------------------------------------
# Team layer
# ---------------------------------------------------------------------------

from .team.team import Team
from .team.messaging import Message, MessageBus

# ---------------------------------------------------------------------------
# Task layer
# ---------------------------------------------------------------------------

from .task.queue import TaskQueue, TaskQueueEvent
from .task.task import (
    create_task,
    is_task_ready,
    get_task_dependency_order,
    validate_task_dependencies,
)

# ---------------------------------------------------------------------------
# Tool system
# ---------------------------------------------------------------------------

from .tool.framework import define_tool, ToolRegistry
from .tool.executor import BatchToolCall, ToolExecutor
from .tool.built_in import (
    register_built_in_tools,
    BUILT_IN_TOOLS,
    bash_tool,
    file_read_tool,
    file_write_tool,
    file_edit_tool,
    grep_tool,
)

# ---------------------------------------------------------------------------
# LLM adapters
# ---------------------------------------------------------------------------

from .llm.adapter import create_adapter

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

from .memory.store import InMemoryStore
from .memory.shared import SharedMemory

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

from .utils.semaphore import Semaphore

# ---------------------------------------------------------------------------
# Types -- all public models re-exported for consumer type-checking
# ---------------------------------------------------------------------------

from .types import (
    # Content blocks
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    ContentBlock,
    # LLM
    LLMMessage,
    LLMResponse,
    LLMAdapter,
    LLMChatOptions,
    LLMStreamOptions,
    LLMToolDef,
    TokenUsage,
    StreamEvent,
    # Tools
    ToolDefinition,
    ToolResult,
    ToolUseContext,
    AgentInfo,
    TeamInfo,
    # Agent
    AgentConfig,
    AgentState,
    AgentRunResult,
    ToolCallRecord,
    # Team
    TeamConfig,
    TeamRunResult,
    # Task
    Task,
    TaskStatus,
    # Orchestrator
    OrchestratorConfig,
    OrchestratorEvent,
    # Memory
    MemoryEntry,
    MemoryStore,
    # Provider
    SupportedProvider,
)

__all__ = [
    # Orchestrator
    "OpenMultiAgent",
    "Scheduler",
    "SchedulingStrategy",
    # Agent
    "Agent",
    "AgentPool",
    "PoolStatus",
    # Team
    "Team",
    "Message",
    "MessageBus",
    # Task
    "TaskQueue",
    "TaskQueueEvent",
    "create_task",
    "is_task_ready",
    "get_task_dependency_order",
    "validate_task_dependencies",
    # Tool system
    "define_tool",
    "ToolRegistry",
    "BatchToolCall",
    "ToolExecutor",
    "register_built_in_tools",
    "BUILT_IN_TOOLS",
    "bash_tool",
    "file_read_tool",
    "file_write_tool",
    "file_edit_tool",
    "grep_tool",
    # LLM
    "create_adapter",
    # Memory
    "InMemoryStore",
    "SharedMemory",
    # Utils
    "Semaphore",
    # Types
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "ContentBlock",
    "LLMMessage",
    "LLMResponse",
    "LLMAdapter",
    "LLMChatOptions",
    "LLMStreamOptions",
    "LLMToolDef",
    "TokenUsage",
    "StreamEvent",
    "ToolDefinition",
    "ToolResult",
    "ToolUseContext",
    "AgentInfo",
    "TeamInfo",
    "AgentConfig",
    "AgentState",
    "AgentRunResult",
    "ToolCallRecord",
    "TeamConfig",
    "TeamRunResult",
    "Task",
    "TaskStatus",
    "OrchestratorConfig",
    "OrchestratorEvent",
    "MemoryEntry",
    "MemoryStore",
    "SupportedProvider",
]
