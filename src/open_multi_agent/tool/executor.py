"""Parallel tool executor with concurrency control and error isolation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..types import ToolDefinition, ToolResult, ToolUseContext
from ..utils.semaphore import Semaphore
from .framework import ToolRegistry


@dataclass
class BatchToolCall:
    id: str
    name: str
    input: dict[str, Any]


class ToolExecutor:
    """Executes tools from a ToolRegistry, validating input against each tool's
    Pydantic schema and enforcing a concurrency limit for batch execution."""

    def __init__(self, registry: ToolRegistry, *, max_concurrency: int = 4) -> None:
        self._registry = registry
        self._semaphore = Semaphore(max_concurrency)

    async def execute(
        self,
        tool_name: str,
        input: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return self._error_result(f'Tool "{tool_name}" is not registered in the ToolRegistry.')

        if context.cancel_event and context.cancel_event.is_set():
            return self._error_result(f'Tool "{tool_name}" was aborted before execution began.')

        return await self._run_tool(tool, input, context)

    async def execute_batch(
        self,
        calls: list[BatchToolCall],
        context: ToolUseContext,
    ) -> dict[str, ToolResult]:
        results: dict[str, ToolResult] = {}

        async def _run_one(call: BatchToolCall) -> None:
            result = await self._semaphore.run(
                lambda: self.execute(call.name, call.input, context)
            )
            results[call.id] = result

        await asyncio.gather(*[_run_one(call) for call in calls])
        return results

    async def _run_tool(
        self,
        tool: ToolDefinition,
        raw_input: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        # Pydantic validation
        try:
            parsed = tool.input_schema.model_validate(raw_input)
        except ValidationError as e:
            issues = "\n".join(f"  - {err['loc']}: {err['msg']}" for err in e.errors())
            return self._error_result(f'Invalid input for tool "{tool.name}":\n{issues}')

        if context.cancel_event and context.cancel_event.is_set():
            return self._error_result(f'Tool "{tool.name}" was aborted before execution began.')

        try:
            return await tool.execute(parsed, context)
        except Exception as err:
            return self._error_result(f'Tool "{tool.name}" threw an error: {err}')

    @staticmethod
    def _error_result(message: str) -> ToolResult:
        return ToolResult(data=message, is_error=True)
