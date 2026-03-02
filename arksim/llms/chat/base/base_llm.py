from abc import ABC, abstractmethod
from typing import TypeVar, overload

from pydantic import BaseModel

from arksim.llms.chat.base.types import LLMMessage

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    def __init__(
        self,
        model: str,
        provider: str | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> None:
        if not model:
            raise ValueError("Model name is required")

        self.model = model
        self.provider = provider
        self.temperature = temperature

    @overload
    def call(
        self,
        messages: str | list[LLMMessage],
        schema: type[T],
        **kwargs: object,
    ) -> T: ...

    @overload
    def call(
        self,
        messages: str | list[LLMMessage],
        schema: None = None,
        **kwargs: object,
    ) -> str: ...

    @abstractmethod
    def call(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        **kwargs: object,
    ) -> T | str:
        """Call the LLM with the given messages.

        Args:
            messages: Can be a string or a list of LLMMessage objects.
            schema: Optional schema for structured output.
            **kwargs: Provider-specific call parameters.

        Returns:
            Either a string or a parsed structured object
            (provider-dependent).
        """

    @overload
    async def call_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T],
        **kwargs: object,
    ) -> T: ...

    @overload
    async def call_async(
        self,
        messages: str | list[LLMMessage],
        schema: None = None,
        **kwargs: object,
    ) -> str: ...

    @abstractmethod
    async def call_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        **kwargs: object,
    ) -> T | str:
        """Async version of call. Call the LLM with the given messages.

        Args:
            messages: Can be a string or a list of LLMMessage objects.
            schema: Optional schema for structured output.
            **kwargs: Provider-specific call parameters.

        Returns:
            Either a string or a parsed structured object
            (provider-dependent).
        """
