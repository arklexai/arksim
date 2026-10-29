# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar, overload

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.streaming import (
    Action,
    JsonStringFieldStream,
    StreamDone,
    StreamEvent,
    TextDelta,
    TruncatedActionError,
)
from arksim.llms.chat.base.tools import ToolCall, ToolDeclaration, ToolExchange
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.base.usage import clean_usage_value, track_usage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)


def _incomplete_reason(response: object) -> str:
    """Why the Responses API says it stopped early, or "" if it does not say."""
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", "") or ""


WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search"}


def _tool_history_items(exchange: ToolExchange) -> list[dict[str, Any]]:
    """Replay one completed round as Responses API input items."""
    if exchange.call is None:
        # Not an answer to any call, so it must not be dressed as one: an
        # output whose call_id matches no call is rejected.
        return [{"role": "user", "content": exchange.observation}]
    # The output has to name the call it answers, and the id is what the model
    # issued, so it travels on the call. A call rebuilt without one gets a
    # fresh id: the pair still validates, it just is not the original.
    call_id = (
        exchange.call.provider_state.decode()
        if exchange.call.provider_state
        else f"call_{uuid.uuid4().hex}"
    )
    return [
        {
            "type": "function_call",
            "call_id": call_id,
            "name": exchange.call.name,
            "arguments": json.dumps(exchange.call.arguments or {}),
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": exchange.observation,
        },
    ]


def _parse_tool_calls(response: object) -> list[ToolCall]:
    """Pull function calls out of a response, ignoring any prose beside them.

    A model that answers with text and no call is saying it is done, which
    comes back as an empty list rather than an error.
    """
    calls: list[ToolCall] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", "") != "function_call" or not getattr(
            item, "name", ""
        ):
            continue
        try:
            arguments = json.loads(getattr(item, "arguments", "") or "{}")
        except ValueError:
            arguments = {}
        call_id = getattr(item, "call_id", None)
        calls.append(
            ToolCall(
                name=item.name,
                arguments=arguments if isinstance(arguments, dict) else {},
                provider_state=call_id.encode() if call_id else None,
            )
        )
    return calls


class OpenAILLM(BaseLLM):
    supports_token_streaming = True
    supports_web_search = True
    supports_tool_calling = True

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def _prepare_params(
        self,
        messages: str | list[LLMMessage],
        schema: type[BaseModel] | None = None,
        web_search: bool = False,
    ) -> dict[str, Any]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        else:
            raise ValueError("Invalid messages type")

        params: dict[str, Any] = {
            "model": self.model,
            "input": messages,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if schema:
            params["text_format"] = schema
        if web_search:
            params["tools"] = [WEB_SEARCH_TOOL]

        return params

    def _track_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        input_tokens = clean_usage_value(getattr(usage, "input_tokens", 0))
        output_tokens = clean_usage_value(getattr(usage, "output_tokens", 0))
        total_tokens = clean_usage_value(
            getattr(usage, "total_tokens", input_tokens + output_tokens)
        )
        cache_read = clean_usage_value(getattr(input_details, "cached_tokens", 0))
        reasoning = clean_usage_value(getattr(output_details, "reasoning_tokens", 0))
        track_usage(
            self.model,
            "openai",
            input_tokens,
            output_tokens,
            cache_read_tokens=cache_read,
            reasoning_tokens=reasoning,
            total_tokens=total_tokens,
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
        response = self.client.responses.parse(**params)
        self._track_response_usage(response)
        if schema:
            return response.output_parsed
        return response.output_text

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
        response = await self.async_client.responses.parse(**params)
        self._track_response_usage(response)
        if schema:
            return response.output_parsed
        return response.output_text

    async def call_with_tools_async(
        self,
        messages: str | list[LLMMessage],
        *,
        tools: list[ToolDeclaration],
        history: Sequence[ToolExchange] = (),
        **kwargs: object,
    ) -> list[ToolCall]:
        params = self._prepare_params(messages)
        for exchange in history:
            params["input"].extend(_tool_history_items(exchange))
        params["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
                # The schemas are whatever the tool's author wrote. Strict mode
                # refuses any object that does not forbid extra keys, and
                # rewriting a third party's schema to satisfy it would change
                # what the tool accepts.
                "strict": False,
            }
            for tool in tools
        ]
        # The Responses API only proposes calls; nothing here executes one.
        params["tool_choice"] = "auto"
        # ponytail: reasoning items are not replayed, so a reasoning model
        # re-derives its plan each round. Carry the reasoning item in
        # provider_state alongside the call_id if that proves too lossy.
        response = await self.async_client.responses.create(**params)
        self._track_response_usage(response)
        return _parse_tool_calls(response)

    async def stream_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        cancel: Callable[[], bool] | None = None,
        text_field: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Stream via the Responses API's typed events.

        With a schema, ``output_text`` is the JSON body rather than prose, so the
        raw deltas are never forwarded: doing so would print JSON into the
        conversation. ``text_field`` names the field to recover from that body as
        it streams. The parsed action still arrives at the end either way.

        Usage is tracked from the final response rather than accumulated per event,
        because only the terminal event carries the authoritative totals.
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
        async with self.async_client.responses.stream(**params) as stream:
            async for event in stream:
                if cancel is not None and cancel():
                    # Close the upstream response instead of abandoning it, so a
                    # cancelled turn stops being billed for tokens nobody reads.
                    await stream.close()
                    yield StreamDone(text="".join(visible), cancelled=True)
                    return

                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
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

            final = await stream.get_final_response()

        self._track_response_usage(final)
        raw = "".join(parts) or (getattr(final, "output_text", "") or "")
        text = "".join(visible) if schema is not None else raw
        if schema is not None:
            parsed = getattr(final, "output_parsed", None)
            if parsed is None:
                # The Responses API parses server-side, so a body that ran out of
                # room comes back as no object at all rather than as bad JSON.
                # Said out loud for the same reason the other providers say it:
                # a caller left with an empty action cannot tell "it stopped
                # mid-answer" from "it had nothing to do".
                reason = _incomplete_reason(final)
                raise TruncatedActionError(
                    f"the model stopped mid-answer after {len(raw)} characters"
                    + (f" ({reason})" if reason else ""),
                    raw=raw,
                )
            action = Action(payload=parsed, raw=raw)
            yield action
        yield StreamDone(text=text, action=action)
