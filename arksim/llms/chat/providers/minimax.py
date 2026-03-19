# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar, overload

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)

        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY environment variable is required")

        base_url = os.getenv("MINIMAX_BASE_URL", DEFAULT_BASE_URL)

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _prepare_messages(
        self,
        messages: str | list[LLMMessage],
    ) -> list[dict[str, str]]:
        """Convert input to chat messages format."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def _prepare_params(
        self,
        messages: str | list[LLMMessage],
        schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        chat_messages = self._prepare_messages(messages)

        if schema:
            schema_json = json.dumps(schema.model_json_schema())
            system_prompt = (
                "You must respond with valid JSON only, no extra text. "
                f"JSON Schema: {schema_json}"
            )
            chat_messages.insert(0, {"role": "system", "content": system_prompt})

        params: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }

        if self.temperature is not None:
            params["temperature"] = max(0.01, min(self.temperature, 1.0))

        if schema:
            params["response_format"] = {"type": "json_object"}

        return params

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract JSON from the response text.

        Handles ``<think>…</think>`` reasoning blocks and markdown code fences.
        """
        # Strip <think>…</think> reasoning blocks the model may emit
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if match:
            return json.loads(match.group(1).strip())
        raise ValueError(f"Failed to parse JSON from response: {cleaned[:200]}")

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
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if schema:
            parsed = self._parse_json(content)
            return schema.model_validate(parsed)
        return content

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
        response = await self.async_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if schema:
            parsed = self._parse_json(content)
            return schema.model_validate(parsed)
        return content
