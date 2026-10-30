# SPDX-License-Identifier: Apache-2.0
"""A normalized streaming contract across providers.

Each provider streams differently: OpenAI emits typed response events, Anthropic
emits content-block deltas, Google yields whole chunks. A caller that wants to show
text as it arrives should not have to know which. So every provider yields the same
three things, in the same order:

    TextDelta*  ->  Action?  ->  StreamDone

An ``Action`` carries the structured result the caller asked for with ``schema``.
There is at most one, and ``single_action`` enforces that: a turn that could emit
two would leave the caller to decide which one counted, and whichever rule it chose
would be invisible to the user.

Providers that cannot stream tokens still satisfy the contract. They complete the
call and yield one delta holding the whole text, so a caller never needs a branch
for them. ``supports_token_streaming`` says which behaviour to expect when that
distinction matters (a progress indicator, say), not whether the contract holds.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class TextDelta:
    """A fragment of assistant text, in arrival order."""

    text: str


@dataclass
class Action:
    """The structured result of a turn.

    ``payload`` is the parsed ``schema`` instance. ``raw`` keeps the text it was
    parsed from so a caller can log what the model actually said when validation
    is the thing under investigation.
    """

    payload: BaseModel
    raw: str = ""


@dataclass
class StreamDone:
    """End of stream, with everything the turn produced.

    ``text`` is the concatenation of every delta, so a caller that only wants the
    final message does not have to accumulate deltas itself.
    """

    text: str = ""
    action: Action | None = None
    #: Set when the stream stopped because the caller cancelled it, rather than
    #: because the model finished. Kept distinct so a cancelled turn is never
    #: persisted as a complete one.
    cancelled: bool = False


StreamEvent = TextDelta | Action | StreamDone


class MultipleActionsError(RuntimeError):
    """A turn tried to emit more than one action."""


class TruncatedActionError(RuntimeError):
    """A schema'd response stopped before its JSON was closed.

    Distinct from a schema mismatch, which is a bug in the caller's model or its
    prompt and should stay loud. This is the model running out of room mid-write:
    a long answer that reached the output ceiling, or one that fell into a
    repeating loop and reached it faster. Nothing is recoverable from the body,
    so the caller's choice is to ask again or to report it.

    The body is kept on ``raw`` rather than in the message. It is as long as the
    model could make it, and pydantic quotes the whole thing in a ValidationError
    - the error then travels wherever errors travel, which for a streaming caller
    is down the wire to a browser. Off the message it cannot do that, and a caller
    that wants to know what the model actually wrote can still log a sample of it.
    Diagnosis needs the middle: what distinguishes a genuinely long answer from a
    model stuck repeating six characters is not visible at either end.
    """

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


#: When a body is long enough to be worth watching. Below this there is nothing
#: to find: the median structured body measured over 214 real turns was 1.5KB.
_GUARD_FROM = 4_000

#: An absolute stop, so an endless stream cannot be accumulated without bound.
#: Deliberately far above anything a caller could legitimately produce, because
#: it is a resource limit and not a judgement about the content.
#:
#: Length cannot be the judgement. A body's legitimate size scales with the work
#: it describes - one that revises 100 drafts by name measures 37,000 characters
#: and one revising 300 measures 74,000, both entirely valid - while the runaways
#: measured came in at 64,000, 218,000 and 447,000. The two overlap, so there is
#: no threshold that separates them, and any limit low enough to catch the small
#: runaway would cut off a large real answer. That is what the loop check is for.
MAX_ACTION_CHARS = 1_000_000

#: How much trailing text has to be pure repetition before it counts, and the
#: longest cycle looked for. 512 characters made of a cycle no longer than this
#: does not occur in prose or in JSON structure, however much of either there is:
#: a list of 300 revisions is long but every entry differs. Size-independent,
#: which is the property length does not have.
_LOOP_WINDOW = 512
_LOOP_MAX_CYCLE = 64


def _repeating_cycle(text: str) -> int:
    """Length of the shortest cycle ``text`` is made of, or 0 if it is not.

    Only whole repetitions count, so a passage that merely happens to share a
    prefix with itself is not a loop.
    """
    for size in range(1, _LOOP_MAX_CYCLE + 1):
        unit = text[:size]
        if (
            len(text) >= size * 2
            and text == (unit * (len(text) // size + 1))[: len(text)]
        ):
            return size
    return 0


class RunawayGuard:
    """Watch a streaming structured body for a model that has stopped finishing.

    A response that runs to the output ceiling costs the whole wait and then
    throws, because the JSON never closes. Every one measured got there by
    repeating itself, so this looks for that and not for length: a long answer and
    a stuck one are the same size.

    Used by the Google provider only. Stopping early saves a doomed call but it
    cannot tell a loop from content that legitimately repeats, and a false
    positive costs a response that would have parsed. That trade is worth making
    where the failure has been measured and not everywhere.
    """

    def __init__(self) -> None:
        self._total = 0
        self._tail = ""

    def feed(self, chunk: str) -> str | None:
        """Why this body should be abandoned, or None to keep reading."""
        self._total += len(chunk)
        if self._total < _GUARD_FROM:
            return None
        if self._total > MAX_ACTION_CHARS:
            return f"ran past {MAX_ACTION_CHARS} characters without finishing"
        self._tail = (self._tail + chunk)[-_LOOP_WINDOW:]
        if len(self._tail) < _LOOP_WINDOW:
            return None
        cycle = _repeating_cycle(self._tail)
        if cycle:
            return (
                f"repeated a {cycle} character sequence for {_LOOP_WINDOW} "
                f"characters at {self._total} in"
            )
        return None


_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class JsonStringFieldStream:
    """Recover one string field's text from JSON as it streams.

    With a schema the provider's deltas carry the structured body, so the prose the
    user should see arrives interleaved with syntax. Emitting the deltas raw shows
    them JSON; withholding them entirely means a turn that only ever appears when it
    is over. This yields the decoded characters of a single top-level field, and
    nothing else.

    Deliberately not a JSON parser. It scans for the key at depth 1 and then reads
    its value, which is all that is needed to show text early. The authoritative
    result is still the provider's own parsed object.
    """

    def __init__(self, field_name: str) -> None:
        self._key = f'"{field_name}"'
        self._buffer = ""
        self._in_value = False
        self._done = False
        self._depth = 0

    def feed(self, chunk: str) -> str:
        """Decoded characters of the field that became available in this chunk."""
        if self._done or not chunk:
            return ""
        self._buffer += chunk
        return self._read_value() if self._in_value else self._seek_value()

    def _seek_value(self) -> str:
        """Find the key at depth 1, then the opening quote of its value."""
        index = self._buffer.find(self._key)
        if index == -1:
            # Keep a tail in case the key straddles this chunk and the next.
            keep = len(self._key)
            if len(self._buffer) > keep:
                self._buffer = self._buffer[-keep:]
            return ""

        rest = self._buffer[index + len(self._key) :]
        stripped = rest.lstrip()
        if not stripped:
            return ""
        if stripped[0] != ":":
            # A string elsewhere that merely equals the key name. Skip past it.
            self._buffer = rest
            return self._seek_value()
        after_colon = stripped[1:].lstrip()
        if not after_colon:
            return ""
        if after_colon[0] != '"':
            # Present but not a string, so there is nothing to stream.
            self._done = True
            return ""
        self._buffer = after_colon[1:]
        self._in_value = True
        return self._read_value()

    def _read_value(self) -> str:
        """Decode buffered characters up to the closing quote."""
        out: list[str] = []
        i = 0
        buffer = self._buffer
        while i < len(buffer):
            char = buffer[i]
            if char == '"':
                self._done = True
                self._buffer = ""
                return "".join(out)
            if char == "\\":
                # An escape split across chunks is held whole until it completes,
                # so a half-read \\uXXXX never reaches the caller as literal text.
                if i + 1 >= len(buffer):
                    break
                nxt = buffer[i + 1]
                if nxt == "u":
                    if i + 6 > len(buffer):
                        break
                    try:
                        out.append(chr(int(buffer[i + 2 : i + 6], 16)))
                    except ValueError:
                        out.append(buffer[i : i + 6])
                    i += 6
                    continue
                out.append(_ESCAPES.get(nxt, nxt))
                i += 2
                continue
            out.append(char)
            i += 1

        self._buffer = buffer[i:]
        return "".join(out)


@dataclass
class _Collected:
    text_parts: list[str] = field(default_factory=list)
    action: Action | None = None


async def single_action(
    stream: AsyncIterator[StreamEvent],
) -> AsyncIterator[StreamEvent]:
    """Pass events through, rejecting a second action.

    Wraps a provider stream rather than living inside each one, so the rule is
    stated once and cannot drift between providers.
    """
    seen_action = False
    async for event in stream:
        if isinstance(event, Action):
            if seen_action:
                raise MultipleActionsError(
                    "a turn may emit only one action; a second was produced"
                )
            seen_action = True
        yield event


async def collect(stream: AsyncIterator[StreamEvent]) -> StreamDone:
    """Drain a stream into its final result.

    For callers that want the whole turn rather than the tokens: tests, batch
    paths, and anything not driving a UI.
    """
    collected = _Collected()
    done: StreamDone | None = None
    async for event in single_action(stream):
        if isinstance(event, TextDelta):
            collected.text_parts.append(event.text)
        elif isinstance(event, Action):
            collected.action = event
        else:
            done = event

    text = "".join(collected.text_parts)
    if done is None:
        # A provider that ended without a terminator still has to produce one, or
        # the caller cannot tell a finished turn from a truncated one.
        return StreamDone(text=text, action=collected.action)
    return StreamDone(
        text=done.text or text,
        action=done.action or collected.action,
        cancelled=done.cancelled,
    )
