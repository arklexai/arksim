# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar, overload

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.streaming import (
    Action,
    JsonStringFieldStream,
    RunawayGuard,
    StreamDone,
    StreamEvent,
    TextDelta,
    TruncatedActionError,
)
from arksim.llms.chat.base.tools import (
    ToolCall,
    ToolDeclaration,
    ToolExchange,
)
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.base.usage import clean_usage_value, track_usage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)


def _finish_reason(response: object) -> str:
    """Why the model stopped, as the API reported it, or "" if it did not say."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", None) or (str(reason) if reason else "")


def _parse_action(schema: type[T], raw: str, response: object) -> T:
    """Parse a structured body, telling a cut-off one from a wrong one.

    Gemini returns the structured body as text, so a response that ran out of
    room arrives as JSON with no closing quote. Pydantic reports that as an
    ordinary validation error and quotes the whole body back - hundreds of
    kilobytes of it - which is the wrong shape for something the caller can only
    respond to by asking again.

    A schema mismatch is a different thing and is left to raise as it is: that is
    a prompt or a model to fix, and the body is small enough to read.
    """
    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        if not any(error.get("type") == "json_invalid" for error in exc.errors()):
            raise
        reason = _finish_reason(response)
        raise TruncatedActionError(
            f"the model stopped mid-answer after {len(raw)} characters"
            + (f" ({reason})" if reason else ""),
            raw=raw,
        ) from None


class GoogleLLM(BaseLLM):
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
        # An already-configured client may be handed in. Constructing one here
        # reads ambient credentials, which a test has no way to satisfy and a
        # deployment with a non-default endpoint cannot redirect.
        client = kwargs.pop("client", None)
        super().__init__(model, provider, temperature, **kwargs)
        self.client = client if client is not None else genai.Client()

    def _prepare_contents(
        self,
        messages: str | list[LLMMessage],
    ) -> tuple[str | None, list[types.Content] | str]:
        """Convert messages to Google format.

        Google uses "model" instead of "assistant"
        and handles system instructions separately.
        """
        if isinstance(messages, str):
            return None, messages

        system = None
        contents: list[types.Content] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                role = "model" if m["role"] == "assistant" else m["role"]
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=m["content"])],
                    )
                )

        if not contents and system:
            contents = [types.Content(role="user", parts=[types.Part(text=system)])]
            system = None

        return system, contents

    def _prepare_params(
        self,
        messages: str | list[LLMMessage],
        schema: type[BaseModel] | None = None,
        web_search: bool = False,
    ) -> dict[str, Any]:
        system, contents = self._prepare_contents(messages)

        config_params: dict[str, Any] = {}
        if system:
            config_params["system_instruction"] = system
        if self.temperature is not None:
            config_params["temperature"] = self.temperature
        if schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = schema
        if web_search:
            if schema:
                raise ValueError(
                    "Gemini does not support structured output and hosted tools "
                    "in the same call; request one or the other."
                )
            config_params["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        params: dict[str, Any] = {
            "model": self.model,
            "contents": contents,
        }
        if config_params:
            params["config"] = types.GenerateContentConfig(**config_params)

        return params

    def _track_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        prompt = clean_usage_value(getattr(usage, "prompt_token_count", 0))
        total = clean_usage_value(getattr(usage, "total_token_count", 0))
        candidates = clean_usage_value(getattr(usage, "candidates_token_count", 0))
        thoughts = clean_usage_value(getattr(usage, "thoughts_token_count", 0))
        cached = clean_usage_value(getattr(usage, "cached_content_token_count", 0))
        output = max(total - prompt, candidates + thoughts, 0)
        track_usage(
            self.model,
            "google",
            prompt,
            output,
            cache_read_tokens=cached,
            reasoning_tokens=thoughts,
            total_tokens=total or (prompt + output),
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
        response = self.client.models.generate_content(**params)
        self._track_response_usage(response)
        if schema:
            return _parse_action(schema, response.text, response)
        return response.text

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
        response = await self.client.aio.models.generate_content(**params)
        self._track_response_usage(response)
        if schema:
            return _parse_action(schema, response.text, response)
        return response.text

    async def stream_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        cancel: Callable[[], bool] | None = None,
        text_field: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Stream generated chunks, parsing the action from the accumulated JSON.

        With a schema, Google streams the JSON body itself, so the raw chunks are
        not prose and must not be shown as assistant text. ``text_field`` names the
        field to recover from that body as it streams; without one, nothing is
        shown until the action is parsed at the end.
        """
        if cancel is not None and cancel():
            yield StreamDone(cancelled=True)
            return

        params = self._prepare_params(messages, schema=schema)
        parts: list[str] = []
        visible: list[str] = []
        action: Action | None = None
        last_chunk: object | None = None
        reader = (
            JsonStringFieldStream(text_field)
            if schema is not None and text_field
            else None
        )
        # Stop Google structured-output loops early so the caller can retry.
        guard = RunawayGuard() if schema is not None else None

        stream = await self.client.aio.models.generate_content_stream(**params)
        async for chunk in stream:
            if cancel is not None and cancel():
                yield StreamDone(
                    text="".join(visible) if schema is not None else "".join(parts),
                    cancelled=True,
                )
                return

            last_chunk = chunk
            piece = getattr(chunk, "text", "") or ""
            if not piece:
                continue
            parts.append(piece)
            if guard is not None and (stopped := guard.feed(piece)) is not None:
                raise TruncatedActionError(f"the model {stopped}", raw="".join(parts))
            if schema is None:
                visible.append(piece)
                yield TextDelta(text=piece)
            elif reader is not None:
                shown = reader.feed(piece)
                if shown:
                    visible.append(shown)
                    yield TextDelta(text=shown)

        if last_chunk is not None:
            self._track_response_usage(last_chunk)

        raw = "".join(parts)
        if schema is not None:
            if raw:
                action = Action(payload=_parse_action(schema, raw, last_chunk), raw=raw)
                yield action
            yield StreamDone(text="".join(visible), action=action)
            return
        yield StreamDone(text=raw)

    @staticmethod
    def _tool_turn_contents(turn: list[ToolExchange]) -> list[types.Content]:
        """Replay one model turn: every call it made, then every answer.

        The turn is the unit, not the call. Gemini mints one thought_signature
        per turn and carries it on the first part, so a turn split into one
        turn per call strands every call after the first.
        """
        return [
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name=exchange.call.name,
                            args=exchange.call.arguments or {},
                        ),
                        thought_signature=exchange.call.provider_state,
                    )
                    for exchange in turn
                ],
            ),
            types.Content(
                role="user",
                # Same order as the calls above: the SDK pairs them positionally.
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=exchange.call.name,
                            response={"result": exchange.observation},
                        )
                    )
                    for exchange in turn
                ],
            ),
        ]

    @staticmethod
    def _group_turns(history: Sequence[ToolExchange]) -> list[list[ToolExchange]]:
        """Group consecutive calls that report the same turn_index.

        A call reporting none is replayed alone, as before.
        """
        turns: list[list[ToolExchange]] = []
        for exchange in history:
            if (
                turns
                and exchange.call is not None
                and exchange.call.turn_index is not None
                and turns[-1][0].call is not None
                and turns[-1][0].call.turn_index == exchange.call.turn_index
            ):
                turns[-1].append(exchange)
                continue
            turns.append([exchange])
        return turns

    @staticmethod
    def _plain_message_contents(exchange: ToolExchange) -> list[types.Content]:
        """Replay an exchange that answers no call, as a message of its own.

        A function_response the model never asked for is rejected.
        """
        return [
            types.Content(role="user", parts=[types.Part(text=exchange.observation)])
        ]

    async def call_with_tools_async(
        self,
        messages: str | list[LLMMessage],
        *,
        tools: list[ToolDeclaration],
        history: Sequence[ToolExchange] = (),
        **kwargs: object,
    ) -> list[ToolCall]:
        system, prepared = self._prepare_contents(messages)
        contents: list[types.Content] = (
            [types.Content(role="user", parts=[types.Part(text=prepared)])]
            if isinstance(prepared, str)
            else list(prepared)
        )
        for turn in self._group_turns(history):
            if turn[0].call is None:
                contents.extend(self._plain_message_contents(turn[0]))
                continue
            contents.extend(self._tool_turn_contents(turn))

        config_params: dict[str, Any] = {
            "tools": [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ],
            # The SDK will happily execute a function for the model. That would
            # run before the caller could authorize it, so it is off and the
            # requests come back instead.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system:
            config_params["system_instruction"] = system
        if self.temperature is not None:
            config_params["temperature"] = self.temperature

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_params),
        )
        self._track_response_usage(response)
        return _parse_tool_calls(response)


def _parse_tool_calls(response: object) -> list[ToolCall]:
    """Pull function calls out of a response, ignoring any prose beside them.

    A model that answers with text and no call is saying it is done, which
    comes back as an empty list rather than an error.
    """
    calls: list[ToolCall] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            function_call = getattr(part, "function_call", None)
            if function_call is None or not getattr(function_call, "name", ""):
                continue
            calls.append(
                ToolCall(
                    name=function_call.name,
                    arguments=dict(getattr(function_call, "args", None) or {}),
                    # Absent on non-thinking models, hence optional rather than
                    # asserted.
                    provider_state=getattr(part, "thought_signature", None),
                )
            )
    return calls
