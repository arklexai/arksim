import os
from typing import Any, TypeVar, overload

from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.types import LLMMessage
from arksim.llms.chat.utils import retry
from arksim.llms.utils import get_azure_token_provider

T = TypeVar("T", bound=BaseModel)


class AzureOpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, provider, temperature, **kwargs)

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_client_id = os.getenv("AZURE_CLIENT_ID")

        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        if not api_version:
            raise ValueError(
                "AZURE_OPENAI_API_VERSION environment variable is required"
            )
        if not azure_client_id:
            raise ValueError("AZURE_CLIENT_ID environment variable is required")

        token_provider = get_azure_token_provider(azure_client_id)

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
        )
        self.async_client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
        )

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
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature

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
        params = self._prepare_params(messages)
        if schema:
            response = self.client.beta.chat.completions.parse(
                **params,
                response_format=schema,
            )
            if response.choices[0].message.parsed:
                return response.choices[0].message.parsed
            return response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content

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
        params = self._prepare_params(messages)
        if schema:
            response = await self.async_client.beta.chat.completions.parse(
                **params,
                response_format=schema,
            )
            if response.choices[0].message.parsed:
                return response.choices[0].message.parsed
            return response.choices[0].message.content
        else:
            response = await self.async_client.chat.completions.create(**params)
            return response.choices[0].message.content
