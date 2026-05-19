# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, TypeVar, overload

from openai import AsyncOpenAI, OpenAI, OpenAIError
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.utils import retry

T = TypeVar("T", bound=BaseModel)


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)
        # Coerce empty or whitespace-only strings to omitted, then fall back
        # to OPENAI_BASE_URL / OPENAI_API_KEY env vars. YAML configs commonly
        # produce "" for unset fields; whitespace-only values would otherwise
        # be URL-encoded by the SDK and cause hard-to-diagnose hangs.
        client_kwargs: dict[str, str] = {}
        if base_url and base_url.strip():
            client_kwargs["base_url"] = base_url.strip()
        if api_key and api_key.strip():
            client_kwargs["api_key"] = api_key.strip()
        try:
            self.client = OpenAI(**client_kwargs)
            self.async_client = AsyncOpenAI(**client_kwargs)
        except OpenAIError as exc:
            raise ValueError(
                "OpenAI SDK requires an api_key. Set 'api_key' in your "
                "config.yaml (alongside 'model' and 'provider') or set the "
                "OPENAI_API_KEY environment variable. "
                "See https://docs.arklex.ai/main/user-simulator-on-open-responses"
            ) from exc

        # Per-instance cumulative usage counters. Read via `cache_stats()`.
        # Surfaces OpenAI's automatic prompt-cache hits so users can verify
        # the cost-reduction story without leaving arksim.
        self._cumulative_input_tokens: int = 0
        self._cumulative_cached_input_tokens: int = 0
        self._cumulative_output_tokens: int = 0
        self._call_count: int = 0

    def _record_usage(self, response: object) -> None:
        """Accumulate token-usage counters from a Responses API response.

        Tolerates missing fields: older SDK versions or non-OpenAI backends
        (Ollama, vLLM) may omit `cached_tokens` or even the full `usage`
        object. We never raise from telemetry.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._call_count += 1
        self._cumulative_input_tokens += getattr(usage, "input_tokens", 0) or 0
        self._cumulative_output_tokens += getattr(usage, "output_tokens", 0) or 0
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            self._cumulative_cached_input_tokens += (
                getattr(details, "cached_tokens", 0) or 0
            )

    def cache_stats(self) -> dict[str, int | float]:
        """Return cumulative cache-hit statistics across all calls.

        Keys:
        - ``call_count``: total responses.parse calls observed
        - ``input_tokens``: total input tokens billed
        - ``cached_input_tokens``: subset of input_tokens that hit cache
        - ``output_tokens``: total output tokens
        - ``cache_hit_rate``: cached_input_tokens / input_tokens (0.0 if no calls)
        """
        rate = (
            self._cumulative_cached_input_tokens / self._cumulative_input_tokens
            if self._cumulative_input_tokens > 0
            else 0.0
        )
        return {
            "call_count": self._call_count,
            "input_tokens": self._cumulative_input_tokens,
            "cached_input_tokens": self._cumulative_cached_input_tokens,
            "output_tokens": self._cumulative_output_tokens,
            "cache_hit_rate": rate,
        }

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
        self._record_usage(response)
        # For structured output, return the parsed output
        if schema:
            return response.output_parsed
        # For text output, return the text (default)
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
        self._record_usage(response)
        # For structured output, return the parsed output
        if schema:
            return response.output_parsed
        # For text output, return the text (default)
        return response.output_text
