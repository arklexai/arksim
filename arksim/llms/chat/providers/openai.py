# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, TypeVar, overload

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.base.usage import clean_usage_value, track_usage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)


class OpenAILLM(BaseLLM):
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
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema)
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
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema)
        response = await self.async_client.responses.parse(**params)
        self._track_response_usage(response)
        if schema:
            return response.output_parsed
        return response.output_text
