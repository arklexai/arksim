# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import cast

from typing_extensions import Self

from arksim.llms.chat.base.base_llm import BaseLLM


class LLM(BaseLLM):
    def __new__(cls, model: str, **kwargs: object) -> BaseLLM:
        if not model or not isinstance(model, str):
            raise ValueError("Model name is required")

        provider = kwargs.pop("provider", None)
        llm_class = cls._get_provider(provider)

        return cast(Self, llm_class(model=model, provider=provider, **kwargs))

    @classmethod
    def _get_provider(cls, provider: str) -> type:
        if provider == "openai":
            from arksim.llms.chat.providers.openai import OpenAILLM

            return OpenAILLM
        elif provider == "azure":
            from arksim.llms.chat.providers.azure_openai import AzureOpenAILLM

            return AzureOpenAILLM
        elif provider == "anthropic":
            from arksim.llms.chat.providers.anthropic import AnthropicLLM

            return AnthropicLLM
        elif provider == "google":
            from arksim.llms.chat.providers.google import GoogleLLM

            return GoogleLLM
        else:
            raise ValueError(f"Provider {provider} is not supported")
