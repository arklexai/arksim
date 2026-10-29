# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral tool calling for ``BaseLLM``.

A tool-calling turn is not a text turn with extra arguments. The model answers
with structured requests rather than prose, the caller usually has to authorize
or bound each request before it runs, and the result has to be fed back in a
shape the provider recognizes as that call's answer. ``call`` and
``call_async`` build a request from messages and a schema and have nowhere to
put any of that, which is why this is a separate channel rather than another
keyword argument.

Nothing here executes a tool. The model proposes and the caller disposes: an
SDK that runs functions on the model's behalf would act before the caller could
check what it was about to do, so every provider implementing this channel must
disable that and hand the requests back instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolCallingUnsupported(NotImplementedError):
    """Raised by a provider with no tool-calling channel.

    A distinct type because callers routinely branch on it: choosing another
    model is a reasonable response to this and not to an arbitrary failure.
    """


@dataclass(frozen=True)
class ToolDeclaration:
    """One tool offered to the model, as the model gets to see it.

    ``parameters`` is a JSON Schema object. None declares a tool that takes no
    arguments, which is different from an empty schema in some providers.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCall:
    """One call the model asked for. Not yet executed.

    ``provider_state`` is opaque bytes some providers require echoed back
    verbatim on any later turn that replays this call. Gemini's
    ``thought_signature`` is the motivating case: a thinking model rejects a
    replayed call that arrives without it, with a 400 rather than a degraded
    answer, so a multi-round loop cannot proceed at all.

    It rides on the call rather than in a table the caller keeps on the side,
    because the two must not be able to drift apart. Keyed separately, a
    signature can be looked up for the wrong call or lost when a call is copied,
    and the failure only shows up on the second round.

    ``turn_index`` says which model turn issued this call, so calls made
    together are replayed together: a signature belongs to the turn, not the
    call. None means the caller does not track turns, and each call is then
    replayed alone, as before.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    provider_state: bytes | None = None
    turn_index: int | None = None


@dataclass(frozen=True)
class ToolExchange:
    """One completed round, replayed to the model on the next one.

    Carried as the call and its result rather than as prose, so the model sees
    exactly what it asked for and exactly what came back. Summarizing here
    would let a paraphrase stand in for the tool's own answer.

    ``call`` is None for a plain message the caller wants read as its own turn
    rather than as some tool's result: naming what is still outstanding, for
    instance, which is not an answer to any call the model made.
    """

    observation: str
    call: ToolCall | None = None
    is_error: bool = False


__all__ = [
    "ToolCall",
    "ToolCallingUnsupported",
    "ToolDeclaration",
    "ToolExchange",
]
