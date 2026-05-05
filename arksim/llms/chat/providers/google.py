# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, TypeVar, overload

from google import genai
from google.genai import types
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.base.usage import clean_usage_value, track_usage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)


class GoogleLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)
        self.client = genai.Client()

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
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema)
        response = self.client.models.generate_content(**params)
        self._track_response_usage(response)
        if schema:
            return schema.model_validate_json(response.text)
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
        **kwargs: object,
    ) -> T | str:
        params = self._prepare_params(messages, schema=schema)
        response = await self.client.aio.models.generate_content(**params)
        self._track_response_usage(response)
        if schema:
            return schema.model_validate_json(response.text)
        return response.text
