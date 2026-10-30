# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar, overload

import anthropic
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.streaming import (
    Action,
    JsonStringFieldStream,
    StreamDone,
    StreamEvent,
    TextDelta,
)
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.base.usage import clean_usage_value, track_usage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_TOKENS = 8192

# Keeps one utterance from becoming a research session.
MAX_WEB_FETCHES = 5

# A fetched page lands in context whole.
MAX_FETCH_CONTENT_TOKENS = 10_000

WEB_FETCH_TOOL: dict[str, Any] = {
    "type": "web_fetch_20250910",
    "name": "web_fetch",
    "max_uses": MAX_WEB_FETCHES,
    "max_content_tokens": MAX_FETCH_CONTENT_TOKENS,
}

# Bounds pause_turn resumes so a turn that never settles cannot spin.
MAX_PAUSE_CONTINUATIONS = 3


def _resume_params(params: dict[str, Any], response: object) -> dict[str, Any]:
    """Echo a paused turn back as blocks, not text; the server needs the tool
    blocks to know what it already ran."""
    return {
        **params,
        "messages": [
            *params["messages"],
            {"role": "assistant", "content": response.content},
        ],
    }


def _extract_text(response: object) -> str:
    """Join the text blocks. Not ``content[0].text``: a tool-running turn
    interleaves tool blocks, and block 0 is often one of them."""
    content = getattr(response, "content", None) or []
    return "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )


class AnthropicLLM(BaseLLM):
    supports_token_streaming = True
    supports_web_search = True

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()

    def _prepare_messages(
        self,
        messages: str | list[LLMMessage],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Separate system message from conversation messages.

        Anthropic's API takes system as a top-level parameter
        rather than a message role.
        """
        if isinstance(messages, str):
            return None, [{"role": "user", "content": messages}]

        system = None
        conversation = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conversation.append({"role": m["role"], "content": m["content"]})

        if not conversation and system:
            conversation = [{"role": "user", "content": system}]
            system = None

        return system, conversation

    def _prepare_params(
        self,
        messages: str | list[LLMMessage],
        schema: type[BaseModel] | None = None,
        web_search: bool = False,
    ) -> dict[str, Any]:
        system, conversation = self._prepare_messages(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if system:
            params["system"] = system
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if schema:
            params["output_format"] = schema
        if web_search:
            params["tools"] = [WEB_FETCH_TOOL]
        return params

    def _track_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        base_input = clean_usage_value(getattr(usage, "input_tokens", 0))
        cache_read = clean_usage_value(getattr(usage, "cache_read_input_tokens", 0))
        cache_create = clean_usage_value(
            getattr(usage, "cache_creation_input_tokens", 0)
        )
        output_tokens = clean_usage_value(getattr(usage, "output_tokens", 0))
        input_tokens = base_input + cache_read + cache_create
        track_usage(
            self.model,
            "anthropic",
            input_tokens,
            output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
            total_tokens=input_tokens + output_tokens,
        )

    @overload
    def call(
        self, messages: str | list[LLMMessage], schema: type[T], **kwargs: object
    ) -> T: ...

    @overload
    def call(
        self, messages: str | list[LLMMessage], schema: None = None, **kwargs: object
    ) -> str: ...

    @retry()
    def call(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        web_search: bool = False,
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema, web_search=web_search)
        # For structured output, return the parsed output
        if schema:
            response = self.client.messages.parse(**params)
            self._track_response_usage(response)
            return response.parsed_output
        # For text output, return the text (default)
        response = self.client.messages.create(**params)
        self._track_response_usage(response)
        texts = [_extract_text(response)]
        for _ in range(MAX_PAUSE_CONTINUATIONS):
            if getattr(response, "stop_reason", None) != "pause_turn":
                break
            params = _resume_params(params, response)
            response = self.client.messages.create(**params)
            self._track_response_usage(response)
            texts.append(_extract_text(response))
        return "".join(texts)

    @overload
    async def call_async(
        self, messages: str | list[LLMMessage], schema: type[T], **kwargs: object
    ) -> T: ...

    @overload
    async def call_async(
        self, messages: str | list[LLMMessage], schema: None = None, **kwargs: object
    ) -> str: ...

    @retry()
    async def call_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        web_search: bool = False,
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema, web_search=web_search)
        # For structured output, return the parsed output
        if schema:
            response = await self.async_client.messages.parse(**params)
            self._track_response_usage(response)
            return response.parsed_output
        # For text output, return the text (default)
        response = await self.async_client.messages.create(**params)
        self._track_response_usage(response)
        texts = [_extract_text(response)]
        for _ in range(MAX_PAUSE_CONTINUATIONS):
            if getattr(response, "stop_reason", None) != "pause_turn":
                break
            params = _resume_params(params, response)
            response = await self.async_client.messages.create(**params)
            self._track_response_usage(response)
            texts.append(_extract_text(response))
        return "".join(texts)

    async def stream_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        cancel: Callable[[], bool] | None = None,
        text_field: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Stream text deltas, then the parsed action if a schema was given.

        With a schema the raw deltas carry the structured body rather than prose.
        ``text_field`` names the field to recover from it as it streams; without
        one, nothing is yielded until the parsed action at the end.
        """
        if cancel is not None and cancel():
            yield StreamDone(cancelled=True)
            return

        params = self._prepare_params(messages, schema=schema)
        parts: list[str] = []
        visible: list[str] = []
        action: Action | None = None
        reader = (
            JsonStringFieldStream(text_field)
            if schema is not None and text_field
            else None
        )

        async with self.async_client.messages.stream(**params) as stream:
            async for event in stream:
                if cancel is not None and cancel():
                    await stream.close()
                    yield StreamDone(text="".join(visible), cancelled=True)
                    return

                if getattr(event, "type", "") == "content_block_delta":
                    delta = getattr(getattr(event, "delta", None), "text", "") or ""
                    if not delta:
                        continue
                    parts.append(delta)
                    if schema is None:
                        visible.append(delta)
                        yield TextDelta(text=delta)
                    elif reader is not None:
                        shown = reader.feed(delta)
                        if shown:
                            visible.append(shown)
                            yield TextDelta(text=shown)

            final = await stream.get_final_message()

        self._track_response_usage(final)
        text = "".join(visible) if schema is not None else "".join(parts)
        if schema is not None:
            parsed = getattr(final, "parsed_output", None)
            if parsed is not None:
                action = Action(payload=parsed, raw="".join(parts))
                yield action
        yield StreamDone(text=text, action=action)
