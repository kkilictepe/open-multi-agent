"""High-level Agent class for open-multi-agent."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from ..llm.adapter import create_adapter
from ..tool.executor import ToolExecutor
from ..tool.framework import ToolRegistry
from ..types import (
    AgentConfig,
    AgentInfo,
    AgentRunResult,
    AgentState,
    LLMMessage,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolDefinition,
    ToolUseContext,
)
from .runner import AgentRunner, RunOptions, RunResult, RunnerOptions

ZERO_USAGE = TokenUsage(input_tokens=0, output_tokens=0)


class Agent:
    """High-level wrapper around AgentRunner that manages conversation
    history, state transitions, and tool lifecycle."""

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
    ) -> None:
        self.name = config.name
        self.config = config
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._runner: AgentRunner | None = None
        self._state = AgentState()
        self._message_history: list[LLMMessage] = []

    def _get_runner(self) -> AgentRunner:
        if self._runner is not None:
            return self._runner

        provider = self.config.provider or "anthropic"
        adapter = create_adapter(provider)

        runner_options = RunnerOptions(
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            max_turns=self.config.max_turns or 10,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            allowed_tools=list(self.config.tools) if self.config.tools else None,
            agent_name=self.name,
            agent_role=(self.config.system_prompt or "assistant")[:50],
        )

        self._runner = AgentRunner(
            adapter, self._tool_registry, self._tool_executor, runner_options
        )
        return self._runner

    # -- Primary execution methods ---------------------------------------------

    async def run(self, prompt: str) -> AgentRunResult:
        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        return await self._execute_run(messages)

    async def prompt(self, message: str) -> AgentRunResult:
        user_message = LLMMessage(role="user", content=[TextBlock(text=message)])
        self._message_history.append(user_message)

        result = await self._execute_run(list(self._message_history))

        for msg in result.messages:
            self._message_history.append(msg)

        return result

    async def stream(self, prompt: str) -> AsyncGenerator[StreamEvent, None]:
        messages = [LLMMessage(role="user", content=[TextBlock(text=prompt)])]
        async for event in self._execute_stream(messages):
            yield event

    # -- State management ------------------------------------------------------

    def get_state(self) -> AgentState:
        return self._state.model_copy(deep=True)

    def get_history(self) -> list[LLMMessage]:
        return list(self._message_history)

    def reset(self) -> None:
        self._message_history = []
        self._state = AgentState()

    # -- Dynamic tool management -----------------------------------------------

    def add_tool(self, tool: ToolDefinition) -> None:
        self._tool_registry.register(tool)

    def remove_tool(self, name: str) -> None:
        self._tool_registry.deregister(name)

    def get_tools(self) -> list[str]:
        return [t.name for t in self._tool_registry.list()]

    # -- Private execution core ------------------------------------------------

    async def _execute_run(self, messages: list[LLMMessage]) -> AgentRunResult:
        self._transition_to("running")

        try:
            runner = self._get_runner()
            run_options = RunOptions(
                on_message=lambda msg: self._state.messages.append(msg),
            )

            result = await runner.run(messages, run_options)

            self._state.token_usage = self._state.token_usage + result.token_usage
            self._transition_to("completed")

            return self._to_agent_run_result(result, True)
        except Exception as err:
            self._transition_to_error(str(err))
            return AgentRunResult(
                success=False,
                output=str(err),
                messages=[],
                token_usage=ZERO_USAGE,
                tool_calls=[],
            )

    async def _execute_stream(self, messages: list[LLMMessage]) -> AsyncGenerator[StreamEvent, None]:
        self._transition_to("running")

        try:
            runner = self._get_runner()

            async for event in runner.stream(messages):
                if event.type == "done":
                    data = event.data
                    self._state.token_usage = self._state.token_usage + data.token_usage
                    self._transition_to("completed")
                elif event.type == "error":
                    self._transition_to_error(str(event.data))

                yield event
        except Exception as err:
            self._transition_to_error(str(err))
            yield StreamEvent(type="error", data=err)

    # -- State transitions -----------------------------------------------------

    def _transition_to(self, status: str) -> None:
        self._state.status = status  # type: ignore[assignment]

    def _transition_to_error(self, error: str) -> None:
        self._state.status = "error"
        self._state.error = error

    # -- Result mapping --------------------------------------------------------

    @staticmethod
    def _to_agent_run_result(result: RunResult, success: bool) -> AgentRunResult:
        return AgentRunResult(
            success=success,
            output=result.output,
            messages=result.messages,
            token_usage=result.token_usage,
            tool_calls=result.tool_calls,
        )

    def build_tool_context(self, cancel_event: asyncio.Event | None = None) -> ToolUseContext:
        return ToolUseContext(
            agent=AgentInfo(
                name=self.name,
                role=(self.config.system_prompt or "assistant")[:60],
                model=self.config.model,
            ),
            cancel_event=cancel_event,
        )
