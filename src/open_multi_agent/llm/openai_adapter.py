"""OpenAI adapter implementing LLMAdapter."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterable

from ..types import (
    ContentBlock,
    LLMChatOptions,
    LLMMessage,
    LLMResponse,
    LLMStreamOptions,
    LLMToolDef,
    StreamEvent,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)


def _to_openai_tool(tool: LLMToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _to_openai_messages(
    messages: list[LLMMessage], system_prompt: str | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "assistant":
            result.append(_to_openai_assistant_message(msg))
        else:
            # user role
            has_tool_results = any(b.type == "tool_result" for b in msg.content)
            if not has_tool_results:
                result.append(_to_openai_user_message(msg))
            else:
                non_tool_blocks = [b for b in msg.content if b.type != "tool_result"]
                if non_tool_blocks:
                    fake_msg = LLMMessage(role="user", content=non_tool_blocks)
                    result.append(_to_openai_user_message(fake_msg))
                for block in msg.content:
                    if block.type == "tool_result":
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_use_id,
                                "content": block.content,
                            }
                        )

    return result


def _to_openai_user_message(msg: LLMMessage) -> dict[str, Any]:
    if len(msg.content) == 1 and msg.content[0].type == "text":
        return {"role": "user", "content": msg.content[0].text}

    parts: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.source.media_type};base64,{block.source.data}"
                    },
                }
            )

    return {"role": "user", "content": parts}


def _to_openai_assistant_message(msg: LLMMessage) -> dict[str, Any]:
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in msg.content:
        if block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
        elif block.type == "text":
            text_parts.append(block.text)

    result: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }

    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def _normalize_finish_reason(reason: str) -> str:
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "content_filter",
    }
    return mapping.get(reason, reason)


class OpenAIAdapter:
    """LLM adapter backed by the OpenAI Chat Completions API."""

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def name(self) -> str:
        return "openai"

    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        openai_messages = _to_openai_messages(messages, options.system_prompt)

        kwargs: dict[str, Any] = {
            "model": options.model,
            "messages": openai_messages,
            "stream": False,
        }
        if options.max_tokens is not None:
            kwargs["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in options.tools]

        completion = await self._client.chat.completions.create(**kwargs)

        choice = completion.choices[0]
        content: list[ContentBlock] = []
        message = choice.message

        if message.content:
            content.append(TextBlock(text=message.content))

        for tool_call in message.tool_calls or []:
            parsed_input: dict[str, Any] = {}
            try:
                parsed = json.loads(tool_call.function.arguments)
                if isinstance(parsed, dict):
                    parsed_input = parsed
            except (json.JSONDecodeError, ValueError):
                pass

            content.append(
                ToolUseBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=parsed_input,
                )
            )

        return LLMResponse(
            id=completion.id,
            content=content,
            model=completion.model,
            stop_reason=_normalize_finish_reason(choice.finish_reason or "stop"),
            usage=TokenUsage(
                input_tokens=completion.usage.prompt_tokens if completion.usage else 0,
                output_tokens=completion.usage.completion_tokens if completion.usage else 0,
            ),
        )

    async def stream(
        self, messages: list[LLMMessage], options: LLMStreamOptions
    ) -> AsyncIterable[StreamEvent]:
        return self._stream_impl(messages, options)

    async def _stream_impl(
        self, messages: list[LLMMessage], options: LLMStreamOptions
    ) -> AsyncIterable[StreamEvent]:
        openai_messages = _to_openai_messages(messages, options.system_prompt)

        kwargs: dict[str, Any] = {
            "model": options.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if options.max_tokens is not None:
            kwargs["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in options.tools]

        completion_id = ""
        completion_model = ""
        final_finish_reason = "stop"
        input_tokens = 0
        output_tokens = 0
        tool_call_buffers: dict[int, dict[str, str]] = {}
        full_text = ""

        try:
            stream_response = await self._client.chat.completions.create(**kwargs)

            async for chunk in stream_response:
                completion_id = chunk.id
                completion_model = chunk.model

                if chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # text delta
                if delta.content is not None:
                    full_text += delta.content
                    yield StreamEvent(type="text", data=delta.content)

                # tool call delta
                for tool_call_delta in delta.tool_calls or []:
                    idx = tool_call_delta.index

                    if idx not in tool_call_buffers:
                        tool_call_buffers[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": (tool_call_delta.function.name or "") if tool_call_delta.function else "",
                            "args_json": "",
                        }

                    buf = tool_call_buffers[idx]
                    if tool_call_delta.id:
                        buf["id"] = tool_call_delta.id
                    if tool_call_delta.function and tool_call_delta.function.name:
                        buf["name"] = tool_call_delta.function.name
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        buf["args_json"] += tool_call_delta.function.arguments

                if choice.finish_reason is not None:
                    final_finish_reason = choice.finish_reason

            # Emit accumulated tool_use events
            final_tool_use_blocks: list[ToolUseBlock] = []
            for buf in tool_call_buffers.values():
                parsed_input: dict[str, Any] = {}
                try:
                    parsed = json.loads(buf["args_json"])
                    if isinstance(parsed, dict):
                        parsed_input = parsed
                except (json.JSONDecodeError, ValueError):
                    pass

                tool_use_block = ToolUseBlock(
                    id=buf["id"],
                    name=buf["name"],
                    input=parsed_input,
                )
                final_tool_use_blocks.append(tool_use_block)
                yield StreamEvent(type="tool_use", data=tool_use_block)

            # Build final response
            done_content: list[ContentBlock] = []
            if full_text:
                done_content.append(TextBlock(text=full_text))
            done_content.extend(final_tool_use_blocks)

            final_response = LLMResponse(
                id=completion_id,
                content=done_content,
                model=completion_model,
                stop_reason=_normalize_finish_reason(final_finish_reason),
                usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            )
            yield StreamEvent(type="done", data=final_response)

        except Exception as err:
            yield StreamEvent(type="error", data=err)
