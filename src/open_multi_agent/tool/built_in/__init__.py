"""Built-in tool collection."""

from __future__ import annotations

from ...types import ToolDefinition
from ..framework import ToolRegistry
from .bash import bash_tool
from .file_edit import file_edit_tool
from .file_read import file_read_tool
from .file_write import file_write_tool
from .grep_tool import grep_tool

__all__ = [
    "bash_tool",
    "file_read_tool",
    "file_write_tool",
    "file_edit_tool",
    "grep_tool",
    "BUILT_IN_TOOLS",
    "register_built_in_tools",
]

BUILT_IN_TOOLS: list[ToolDefinition] = [
    bash_tool,
    file_read_tool,
    file_write_tool,
    file_edit_tool,
    grep_tool,
]


def register_built_in_tools(registry: ToolRegistry) -> None:
    for tool in BUILT_IN_TOOLS:
        registry.register(tool)
