"""Core conversation loop engine for open-multi-agent."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from ..tool.executor import ToolExecutor
from ..tool.framework import ToolRegistry
from ..types import (
    AgentInfo,
    ContentBlock,
    LLMAdapter,
    LLMChatOptions,
    LLMMessage,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolCallRecord,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseContext,
)


@dataclass
class RunnerOptions:
    model: str
    system_prompt: str | None = None
    max_turns: int = 10
    max_tokens: int | None = None
    temperature: float | None = None
    cancel_event: Any | None = None
    allowed_tools: list[str] | None = None
    agent_name: str = "runner"
    agent_role: str = "assistant"


@dataclass
class RunOptions:
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None
    on_tool_result: Callable[[str, ToolResult], None] | None = None
    on_message: Callable[[LLMMessage], None] | None = None


@dataclass
class RunResult:
    messages: list[LLMMessage]
    output: str
    tool_calls: list[ToolCallRecord]
    token_usage: TokenUsage
    turns: int


def _extract_text(content: list[ContentBlock]) -> str:
    return "".join(b.text for b in content if isinstance(b, TextBlock))


def _extract_tool_use_blocks(content: list[ContentBlock]) -> list[ToolUseBlock]:
    return [b for b in content if isinstance(b, ToolUseBlock)]


def _add_token_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
    )


ZERO_USAGE = TokenUsage(input_tokens=0, output_tokens=0)


class AgentRunner:
    """Drives a full agentic conversation: LLM calls, tool execution, and looping."""

    def __init__(
        self,
        adapter: LLMAdapter,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        options: RunnerOptions,
    ) -> None:
        self._adapter = adapter
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._options = options
        self._max_turns = options.max_turns

    async def run(
        self,
        messages: list[LLMMessage],
        options: RunOptions | None = None,
    ) -> RunResult:
        opts = options or RunOptions()
        result = RunResult(messages=[], output="", tool_calls=[], token_usage=ZERO_USAGE, turns=0)

        async for event in self.stream(messages, opts):
            if event.type == "done":
                data = event.data
                result.messages = data.messages
                result.output = data.output
                result.tool_calls = data.tool_calls
                result.token_usage = data.token_usage
                result.turns = data.turns

        return result

    async def stream(
        self,
        initial_messages: list[LLMMessage],
        options: RunOptions | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        opts = options or RunOptions()
        conversation_messages = list(initial_messages)

        total_usage = ZERO_USAGE
        all_tool_calls: list[ToolCallRecord] = []
        final_output = ""
        turns = 0

        # Build stable LLM options
        all_defs = self._tool_registry.to_tool_defs()
        tool_defs = (
            [d for d in all_defs if d.name in self._options.allowed_tools]
            if self._options.allowed_tools
            else all_defs
        )

        base_chat_options = LLMChatOptions(
            model=self._options.model,
            tools=tool_defs if tool_defs else None,
            max_tokens=self._options.max_tokens,
            temperature=self._options.temperature,
            system_prompt=self._options.system_prompt,
            cancel_event=self._options.cancel_event,
        )

        try:
            while True:
                if self._options.cancel_event and self._options.cancel_event.is_set():
                    break
                if turns >= self._max_turns:
                    break

                turns += 1

                # Step 1: Call the LLM
                response = await self._adapter.chat(conversation_messages, base_chat_options)
                total_usage = _add_token_usage(total_usage, response.usage)

                # Step 2: Build assistant message
                assistant_message = LLMMessage(role="assistant", content=response.content)
                conversation_messages.append(assistant_message)
                if opts.on_message:
                    opts.on_message(assistant_message)

                turn_text = _extract_text(response.content)
                if turn_text:
                    yield StreamEvent(type="text", data=turn_text)

                tool_use_blocks = _extract_tool_use_blocks(response.content)
                for block in tool_use_blocks:
                    yield StreamEvent(type="tool_use", data=block)

                # Step 3: Continue?
                if not tool_use_blocks:
                    final_output = turn_text
                    break

                # Step 4: Execute all tool calls in parallel
                tool_context = self._build_tool_context()

                async def _exec_one(block: ToolUseBlock) -> tuple[ToolResultBlock, ToolCallRecord]:
                    if opts.on_tool_call:
                        opts.on_tool_call(block.name, block.input)

                    start_time = time.monotonic()
                    try:
                        result = await self._tool_executor.execute(block.name, block.input, tool_context)
                    except Exception as err:
                        result = ToolResult(data=str(err), is_error=True)

                    duration = (time.monotonic() - start_time) * 1000

                    if opts.on_tool_result:
                        opts.on_tool_result(block.name, result)

                    record = ToolCallRecord(
                        tool_name=block.name,
                        input=block.input,
                        output=result.data,
                        duration=duration,
                    )

                    result_block = ToolResultBlock(
                        tool_use_id=block.id,
                        content=result.data,
                        is_error=result.is_error,
                    )

                    return result_block, record

                executions = await asyncio.gather(*[_exec_one(b) for b in tool_use_blocks])

                # Step 5: Accumulate results
                tool_result_blocks: list[ContentBlock] = []
                for result_block, record in executions:
                    tool_result_blocks.append(result_block)
                    all_tool_calls.append(record)
                    yield StreamEvent(type="tool_result", data=result_block)

                tool_result_message = LLMMessage(role="user", content=tool_result_blocks)
                conversation_messages.append(tool_result_message)
                if opts.on_message:
                    opts.on_message(tool_result_message)

        except Exception as err:
            yield StreamEvent(type="error", data=err)
            return

        # If loop exited due to maxTurns, extract last text
        if not final_output and conversation_messages:
            for msg in reversed(conversation_messages):
                if msg.role == "assistant":
                    final_output = _extract_text(msg.content)
                    break

        run_result = RunResult(
            messages=conversation_messages[len(initial_messages):],
            output=final_output,
            tool_calls=all_tool_calls,
            token_usage=total_usage,
            turns=turns,
        )

        yield StreamEvent(type="done", data=run_result)

    def _build_tool_context(self) -> ToolUseContext:
        return ToolUseContext(
            agent=AgentInfo(
                name=self._options.agent_name,
                role=self._options.agent_role,
                model=self._options.model,
            ),
            cancel_event=self._options.cancel_event,
        )
