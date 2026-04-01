"""Tool definition framework for open-multi-agent.

Provides the core primitives for declaring, registering, and converting
tools to the JSON Schema format that LLM APIs expect.

Pydantic's ``model_json_schema()`` replaces the entire zodToJsonSchema converter.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from ..types import LLMToolDef, ToolDefinition, ToolResult, ToolUseContext


def define_tool(
    *,
    name: str,
    description: str,
    input_schema: type[BaseModel],
    execute: Callable[[Any, ToolUseContext], Awaitable[ToolResult]],
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        execute=execute,
    )


class ToolRegistry:
    """Registry that holds a set of named tools and can produce the JSON Schema
    representation expected by LLM APIs."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f'ToolRegistry: a tool named "{tool.name}" is already registered. '
                "Use a unique name or deregister the existing one first."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def to_tool_defs(self) -> list[LLMToolDef]:
        result: list[LLMToolDef] = []
        for tool in self._tools.values():
            schema = tool.input_schema.model_json_schema()
            result.append(
                LLMToolDef(name=tool.name, description=tool.description, inputSchema=schema)
            )
        return result

    def to_llm_tools(
        self,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in self._tools.values():
            schema = tool.input_schema.model_json_schema()
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        **({"required": schema["required"]} if "required" in schema else {}),
                    },
                }
            )
        return result
