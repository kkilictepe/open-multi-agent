"""Anthropic Claude adapter implementing LLMAdapter."""

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


def _to_anthropic_content_block(block: ContentBlock) -> dict[str, Any]:
    match block.type:
        case "text":
            return {"type": "text", "text": block.text}
        case "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        case "tool_result":
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }
            if block.is_error is not None:
                result["is_error"] = block.is_error
            return result
        case "image":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.source.media_type,
                    "data": block.source.data,
                },
            }
        case _:
            raise ValueError(f"Unhandled content block type: {block.type}")


def _to_anthropic_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    return [
        {"role": msg.role, "content": [_to_anthropic_content_block(b) for b in msg.content]}
        for msg in messages
    ]


def _to_anthropic_tools(tools: list[LLMToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": {"type": "object", **t.input_schema},
        }
        for t in tools
    ]


def _from_anthropic_content_block(block: Any) -> ContentBlock:
    block_type = getattr(block, "type", None) or block.get("type") if isinstance(block, dict) else block.type
    if block_type == "text":
        text = block.text if hasattr(block, "text") else block["text"]
        return TextBlock(text=text)
    elif block_type == "tool_use":
        return ToolUseBlock(
            id=block.id if hasattr(block, "id") else block["id"],
            name=block.name if hasattr(block, "name") else block["name"],
            input=block.input if hasattr(block, "input") else block["input"],
        )
    else:
        return TextBlock(text=f"[unsupported block type: {block_type}]")


class AnthropicAdapter:
    """LLM adapter backed by the Anthropic Claude API."""

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return "anthropic"

    async def chat(self, messages: list[LLMMessage], options: LLMChatOptions) -> LLMResponse:
        anthropic_messages = _to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or 4096,
            "messages": anthropic_messages,
        }
        if options.system_prompt:
            kwargs["system"] = options.system_prompt
        if options.tools:
            kwargs["tools"] = _to_anthropic_tools(options.tools)
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature

        response = await self._client.messages.create(**kwargs)

        content = [_from_anthropic_content_block(b) for b in response.content]

        return LLMResponse(
            id=response.id,
            content=content,
            model=response.model,
            stop_reason=response.stop_reason or "end_turn",
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def stream(
        self, messages: list[LLMMessage], options: LLMStreamOptions
    ) -> AsyncIterable[StreamEvent]:
        return self._stream_impl(messages, options)

    async def _stream_impl(
        self, messages: list[LLMMessage], options: LLMStreamOptions
    ) -> AsyncIterable[StreamEvent]:
        anthropic_messages = _to_anthropic_messages(messages)

        kwargs: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or 4096,
            "messages": anthropic_messages,
        }
        if options.system_prompt:
            kwargs["system"] = options.system_prompt
        if options.tools:
            kwargs["tools"] = _to_anthropic_tools(options.tools)
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature

        # tool input buffers: index -> {id, name, json}
        tool_input_buffers: dict[int, dict[str, str]] = {}

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type") and block.type == "tool_use":
                            tool_input_buffers[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "json": "",
                            }

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type"):
                            if delta.type == "text_delta":
                                yield StreamEvent(type="text", data=delta.text)
                            elif delta.type == "input_json_delta":
                                buf = tool_input_buffers.get(event.index)
                                if buf is not None:
                                    buf["json"] += delta.partial_json

                    elif event.type == "content_block_stop":
                        buf = tool_input_buffers.get(event.index)
                        if buf is not None:
                            parsed_input: dict[str, Any] = {}
                            try:
                                parsed = json.loads(buf["json"])
                                if isinstance(parsed, dict):
                                    parsed_input = parsed
                            except (json.JSONDecodeError, ValueError):
                                pass

                            tool_use_block = ToolUseBlock(
                                id=buf["id"],
                                name=buf["name"],
                                input=parsed_input,
                            )
                            yield StreamEvent(type="tool_use", data=tool_use_block)
                            del tool_input_buffers[event.index]

                final_message = await stream.get_final_message()
                content = [_from_anthropic_content_block(b) for b in final_message.content]

                final_response = LLMResponse(
                    id=final_message.id,
                    content=content,
                    model=final_message.model,
                    stop_reason=final_message.stop_reason or "end_turn",
                    usage=TokenUsage(
                        input_tokens=final_message.usage.input_tokens,
                        output_tokens=final_message.usage.output_tokens,
                    ),
                )
                yield StreamEvent(type="done", data=final_response)

        except Exception as err:
            yield StreamEvent(type="error", data=err)
