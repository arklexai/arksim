# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, TypeVar, overload

import anthropic
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.base.usage import track_usage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_TOKENS = 8192


class AnthropicLLM(BaseLLM):
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
        return params

    def _track_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            track_usage(
                self.model,
                "anthropic",
                usage.input_tokens,
                usage.output_tokens,
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
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema)
        # For structured output, return the parsed output
        if schema:
            response = self.client.messages.parse(**params)
            self._track_response_usage(response)
            return response.parsed_output
        # For text output, return the text (default)
        response = self.client.messages.create(**params)
        self._track_response_usage(response)
        return response.content[0].text

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
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema)
        # For structured output, return the parsed output
        if schema:
            response = await self.async_client.messages.parse(**params)
            self._track_response_usage(response)
            return response.parsed_output
        # For text output, return the text (default)
        response = await self.async_client.messages.create(**params)
        self._track_response_usage(response)
        return response.content[0].text
