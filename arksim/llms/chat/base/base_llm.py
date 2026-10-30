# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from typing import ClassVar, TypeVar, overload

from pydantic import BaseModel

from arksim.llms.chat.base.streaming import (
    Action,
    StreamDone,
    StreamEvent,
    TextDelta,
)
from arksim.llms.chat.base.tools import (
    ToolCall,
    ToolCallingUnsupported,
    ToolDeclaration,
    ToolExchange,
)
from arksim.llms.chat.base.types import LLMMessage

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    #: Whether ``stream_async`` yields text as the model produces it. False means
    #: the contract is still honoured, but the whole reply arrives in one delta.
    #: Not a gate on calling it: it tells a UI whether to expect progressive text.
    supports_token_streaming: ClassVar[bool] = False
    supports_web_search: ClassVar[bool] = False
    #: Whether ``call_with_tools_async`` is implemented. False means calling it
    #: raises ``ToolCallingUnsupported`` rather than answering as though no tool
    #: had been offered, which would look like the model declining to use one.
    supports_tool_calling: ClassVar[bool] = False

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

    async def stream_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[T] | None = None,
        cancel: Callable[[], bool] | None = None,
        text_field: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a reply as ``TextDelta*`` then an optional ``Action``, then ``StreamDone``.

        The default implementation completes the call and yields the whole reply as
        one delta, so every provider satisfies the contract without implementing
        streaming. Providers that can stream tokens override this and set
        ``supports_token_streaming``.

        ``cancel`` is polled between events. A cancelled stream still terminates
        with ``StreamDone(cancelled=True)`` rather than simply stopping, because a
        caller that stopped early must be able to tell that from a turn that
        finished.

        Args:
            messages: Can be a string or a list of LLMMessage objects.
            schema: Optional schema for the turn's structured action.
            cancel: Polled between events; truthy aborts the stream.
            text_field: With a schema, the top-level string field to emit as
                deltas. Without it a structured turn yields no text at all, since
                the raw deltas are JSON. Ignored when ``schema`` is None.
            **kwargs: Provider-specific call parameters.
        """
        if cancel is not None and cancel():
            yield StreamDone(cancelled=True)
            return

        result = await self.call_async(messages, schema=schema, **kwargs)  # type: ignore[call-overload]

        if schema is not None:
            action = Action(payload=result, raw="")
            # No token streaming here, but the field still has to reach the caller
            # or a non-streaming provider would silently drop the turn's prose.
            text = getattr(result, text_field, "") if text_field else ""
            if text:
                yield TextDelta(text=str(text))
            yield action
            yield StreamDone(text=str(text), action=action)
            return

        text = result if isinstance(result, str) else str(result)
        if text:
            yield TextDelta(text=text)
        yield StreamDone(text=text)

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

    async def call_with_tools_async(
        self,
        messages: str | list[LLMMessage],
        *,
        tools: list[ToolDeclaration],
        history: Sequence[ToolExchange] = (),
        **kwargs: object,
    ) -> list[ToolCall]:
        """Ask the model which of ``tools`` it wants called next.

        Returns what it asked for, having executed none of it. A caller that
        must authorize, bound or audit a call has to see the request first, so
        providers disable any automatic function calling their SDK offers.

        An empty list means the model considers itself done. That is a normal
        answer and not a failure: it has either gathered what it needed or
        decided no offered tool can help.

        ``history`` replays earlier rounds. Passing back the same ``ToolCall``
        objects that were returned matters, because a provider may have
        attached state to them that it requires on replay.

        Args:
            messages: A string, or messages including any system instruction.
            tools: What the model may call. Offering none is pointless rather
                than an error, and providers need not special-case it.
            history: Completed rounds, oldest first.
            **kwargs: Provider-specific call parameters.

        Raises:
            ToolCallingUnsupported: If this provider has no tool-calling
                channel. Callers branch on this to pick another model.
        """
        raise ToolCallingUnsupported(
            f"{type(self).__name__} has no tool-calling channel; "
            "check supports_tool_calling before calling this"
        )
