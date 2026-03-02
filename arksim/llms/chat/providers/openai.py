from typing import Any, TypeVar, overload

from openai import AsyncOpenAI, OpenAI
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
        # For structured output, return the parsed output
        if schema:
            return response.output_parsed
        # For text output, return the text (default)
        return response.output_text
