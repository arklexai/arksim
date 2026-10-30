# SPDX-License-Identifier: Apache-2.0
"""Tests for the normalized streaming contract.

The contract is what callers depend on, so these assert the shape every provider
must produce rather than any one SDK's event names: text deltas in order, at most
one action, and always a terminator that says whether the turn finished or was
cancelled.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.streaming import (
    Action,
    JsonStringFieldStream,
    MultipleActionsError,
    RunawayGuard,
    StreamDone,
    StreamEvent,
    TextDelta,
    TruncatedActionError,
    collect,
    single_action,
)
from arksim.llms.chat.base.types import LLMMessage


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


# The provider SDKs are optional extras, and CI installs only ``.[dev]``.
requires_google = pytest.mark.skipif(
    not _installed("google.genai"), reason="google-genai not installed"
)
requires_anthropic = pytest.mark.skipif(
    not _installed("anthropic"), reason="anthropic not installed"
)


class TurnAction(BaseModel):
    action: str
    reason: str = ""


class _NonStreamingLLM(BaseLLM):
    """A provider with no streaming of its own, exercising the base fallback."""

    def __init__(self, result: object = "hello world") -> None:
        super().__init__(model="fake-model", provider="fake")
        self.result = result
        self.calls = 0

    def call(
        self,
        messages: str | list[LLMMessage],
        schema: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> object:
        raise NotImplementedError

    async def call_async(
        self,
        messages: str | list[LLMMessage],
        schema: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> object:
        self.calls += 1
        return self.result


async def _events(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


def _feed(field: str, chunks: list[str]) -> str:
    reader = JsonStringFieldStream(field)
    return "".join(reader.feed(chunk) for chunk in chunks)


class TestJsonStringFieldStream:
    def test_emits_the_field_as_it_arrives(self) -> None:
        reader = JsonStringFieldStream("message")
        assert reader.feed('{"action": "ask_user", "mess') == ""
        assert reader.feed('age": "Got ') == "Got "
        assert reader.feed('it now"') == "it now"

    def test_stops_at_the_closing_quote(self) -> None:
        # Everything after the field is structure, and leaking it would put JSON
        # into the conversation, which is the bug this whole class exists to fix.
        text = _feed("message", ['{"message": "done", "questions": [{"id": "q1"}]}'])
        assert text == "done"

    def test_escapes_are_decoded(self) -> None:
        assert (
            _feed("message", ['{"message": "a\\nb \\"q\\" c\\\\d"}']) == 'a\nb "q" c\\d'
        )

    def test_escape_split_across_chunks_is_held_whole(self) -> None:
        # A chunk boundary inside \\n must not surface a lone backslash.
        assert _feed("message", ['{"message": "a\\', 'nb"}']) == "a\nb"

    def test_unicode_escape_split_across_chunks(self) -> None:
        assert _feed("message", ['{"message": "x\\u00', 'e9y"}']) == "xéy"

    def test_a_matching_string_elsewhere_is_not_the_field(self) -> None:
        # The word appearing as a value must not be mistaken for the key.
        text = _feed("message", ['{"kind": "message", "message": "real"}'])
        assert text == "real"

    def test_absent_field_yields_nothing(self) -> None:
        assert _feed("message", ['{"action": "generate", "questions": []}']) == ""

    def test_non_string_field_yields_nothing(self) -> None:
        assert _feed("message", ['{"message": null, "action": "generate"}']) == ""

    def test_key_split_across_chunks(self) -> None:
        assert _feed("message", ['{"mess', 'age"', ': "hi"}']) == "hi"


class TestBaseFallback:
    async def test_non_streaming_provider_still_satisfies_the_contract(self) -> None:
        llm = _NonStreamingLLM()
        events = await _events(llm.stream_async("hi"))

        assert [type(e) for e in events] == [TextDelta, StreamDone]
        assert events[0].text == "hello world"
        assert events[-1].text == "hello world"
        assert events[-1].cancelled is False

    async def test_flag_reports_no_token_streaming(self) -> None:
        assert _NonStreamingLLM().supports_token_streaming is False

    async def test_schema_yields_one_action(self) -> None:
        payload = TurnAction(action="ask_user")
        llm = _NonStreamingLLM(result=payload)
        events = await _events(llm.stream_async("hi", schema=TurnAction))

        assert [type(e) for e in events] == [Action, StreamDone]
        assert events[0].payload is payload
        assert events[-1].action is events[0]

    async def test_cancel_before_start_does_not_call_the_model(self) -> None:
        llm = _NonStreamingLLM()
        events = await _events(llm.stream_async("hi", cancel=lambda: True))

        # Terminated rather than simply empty: a caller must be able to tell a
        # cancelled turn from one that produced nothing.
        assert [type(e) for e in events] == [StreamDone]
        assert events[0].cancelled is True
        assert llm.calls == 0

    async def test_empty_reply_yields_no_delta_but_still_terminates(self) -> None:
        llm = _NonStreamingLLM(result="")
        events = await _events(llm.stream_async("hi"))
        assert [type(e) for e in events] == [StreamDone]
        assert events[0].cancelled is False


class TestSingleAction:
    async def test_a_second_action_is_rejected(self) -> None:
        async def two_actions() -> AsyncIterator[StreamEvent]:
            yield Action(payload=TurnAction(action="generate"))
            yield Action(payload=TurnAction(action="review"))

        with pytest.raises(MultipleActionsError, match="only one action"):
            await _events(single_action(two_actions()))

    async def test_one_action_passes_through(self) -> None:
        async def one_action() -> AsyncIterator[StreamEvent]:
            yield TextDelta(text="thinking")
            yield Action(payload=TurnAction(action="generate"))
            yield StreamDone(text="thinking")

        events = await _events(single_action(one_action()))
        assert [type(e) for e in events] == [TextDelta, Action, StreamDone]

    async def test_deltas_are_not_limited(self) -> None:
        async def many() -> AsyncIterator[StreamEvent]:
            for i in range(5):
                yield TextDelta(text=str(i))
            yield StreamDone(text="01234")

        events = await _events(single_action(many()))
        assert len([e for e in events if isinstance(e, TextDelta)]) == 5


class TestCollect:
    async def test_joins_deltas_in_order(self) -> None:
        async def stream() -> AsyncIterator[StreamEvent]:
            yield TextDelta(text="one ")
            yield TextDelta(text="two ")
            yield TextDelta(text="three")
            yield StreamDone()

        assert (await collect(stream())).text == "one two three"

    async def test_keeps_the_action(self) -> None:
        payload = TurnAction(action="ask_user")

        async def stream() -> AsyncIterator[StreamEvent]:
            yield Action(payload=payload)
            yield StreamDone()

        assert (await collect(stream())).action.payload is payload

    async def test_missing_terminator_still_returns_a_result(self) -> None:
        async def truncated() -> AsyncIterator[StreamEvent]:
            yield TextDelta(text="partial")

        result = await collect(truncated())
        assert result.text == "partial"
        assert result.cancelled is False

    async def test_cancellation_survives_collection(self) -> None:
        async def stream() -> AsyncIterator[StreamEvent]:
            yield TextDelta(text="partial")
            yield StreamDone(text="partial", cancelled=True)

        assert (await collect(stream())).cancelled is True

    async def test_a_second_action_is_still_rejected(self) -> None:
        async def stream() -> AsyncIterator[StreamEvent]:
            yield Action(payload=TurnAction(action="a"))
            yield Action(payload=TurnAction(action="b"))

        with pytest.raises(MultipleActionsError):
            await collect(stream())


def _async_iter(items: list[object]) -> AsyncIterator[object]:
    async def gen() -> AsyncIterator[object]:
        for item in items:
            yield item

    return gen()


def _stream(body: str, chunk: int = 400) -> str | None:
    """Feed a body through the guard as it would arrive, returning its verdict."""
    guard = RunawayGuard()
    for start in range(0, len(body), chunk):
        stopped = guard.feed(body[start : start + chunk])
        if stopped:
            return stopped
    return None


def _prose(words: int, seed: int = 7) -> str:
    """Filler that reads like content rather than like a repetition."""
    import random

    vocabulary = [
        "customer",
        "warranty",
        "return",
        "policy",
        "appliance",
        "delivery",
        "contractor",
        "refund",
        "escalate",
        "specification",
        "installation",
        "receipt",
        "store",
        "aisle",
        "order",
        "patient",
    ]
    rng = random.Random(seed)
    return " ".join(rng.choice(vocabulary) for _ in range(words))


class TestRunawayGuard:
    """Stopping a body that has stopped finishing, without stopping a long one.

    Length looks like the obvious signal and is the wrong one. A structured body
    grows with the work it describes - revising 300 drafts by name measures 74,000
    characters and is entirely valid - while the runaways measured came in at
    64,000, 218,000 and 447,000. They overlap, so no limit separates them.
    """

    def test_it_catches_the_loop_that_started_this(self) -> None:
        # The tail of a real failure: the six digits 910482, over and over, until
        # the output ceiling stopped it 240,000 characters later.
        body = '{\n  "message": "reference numbers (e.g., 123' + "910482" * 40_000
        stopped = _stream(body)

        assert stopped and "6 character sequence" in stopped

    def test_it_stops_near_the_start_of_the_loop(self) -> None:
        # The point of catching it at all. Left alone this runs to the model's
        # ceiling, which is two minutes of waiting and a full body of tokens
        # billed for an answer that cannot be parsed.
        body = '{\n  "message": "' + _prose(600) + "910482" * 40_000
        guard = RunawayGuard()
        consumed = 0
        for start in range(0, len(body), 400):
            consumed += len(body[start : start + 400])
            if guard.feed(body[start : start + 400]):
                break

        assert consumed < 12_000, "the loop should be caught, not merely detected"

    def test_a_long_answer_about_a_lot_of_work_is_not_a_loop(self) -> None:
        # The case that killed the length limit: 450 revisions, each naming a
        # different draft. Long because the session is large, not because the
        # model is stuck, and no threshold tells those apart by size - this body
        # is larger than the smallest runaway measured.
        entries = ",".join(
            f'{{"draft_id": "{index:08x}-aaaa-bbbb-cccc-{index:012x}", '
            f'"instruction": "{_prose(9, seed=index)}"}}'
            for index in range(450)
        )
        body = '{"action": "revise", "revise_drafts": [' + entries + "]}"

        assert len(body) > 63_829, "smaller than a real runaway proves nothing"
        assert _stream(body) is None

    def test_ordinary_prose_of_any_length_passes(self) -> None:
        assert _stream('{"message": "' + _prose(20_000) + '"}') is None

    def test_a_short_body_is_never_examined(self) -> None:
        # Nothing under a few KB can be the failure this looks for, and checking
        # it would only be a chance to be wrong.
        assert (
            _stream('{"action": "ask_user", "message": "' + "ab" * 500 + '"}') is None
        )


class TestProviderCompatibility:
    """Every provider must produce the same event shape from its own SDK events."""

    async def test_openai_streams_text_deltas_and_tracks_usage(self) -> None:
        from arksim.llms.chat.providers.openai import OpenAILLM

        final = SimpleNamespace(
            output_text="hi there",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )
        sdk_events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="hi "),
            SimpleNamespace(type="response.output_text.delta", delta="there"),
            SimpleNamespace(type="response.completed"),
        ]

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = _returns(_FakeStream(sdk_events, final))
        stream_ctx.__aexit__ = _returns(False)

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="gpt-4o")
        llm.async_client = MagicMock()
        llm.async_client.responses.stream = MagicMock(return_value=stream_ctx)

        with patch("arksim.llms.chat.providers.openai.track_usage") as usage:
            events = await _events(llm.stream_async("hi"))

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["hi ", "there"]
        assert events[-1].text == "hi there"
        usage.assert_called_once()
        assert llm.supports_token_streaming is True

    async def test_openai_streams_the_named_field_as_prose(self) -> None:
        # The other half of the same problem: suppressing the JSON entirely left a
        # structured turn with no visible text until it finished. text_field
        # recovers the prose without ever forwarding syntax.
        from arksim.llms.chat.providers.openai import OpenAILLM

        body = '{"action":"ask_user","reason":"Got it, one moment"}'
        final = SimpleNamespace(
            output_text=body,
            output_parsed=TurnAction(action="ask_user", reason="Got it, one moment"),
            usage=None,
        )
        # Split inside the prose, which is where a real stream breaks and the only
        # place that proves text is emitted before the value is complete.
        cut = body.index("one")
        sdk_events = [
            SimpleNamespace(type="response.output_text.delta", delta=body[:cut]),
            SimpleNamespace(type="response.output_text.delta", delta=body[cut:]),
        ]
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = _returns(_FakeStream(sdk_events, final))
        stream_ctx.__aexit__ = _returns(False)

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="gpt-4o")
        llm.async_client = MagicMock()
        llm.async_client.responses.stream = MagicMock(return_value=stream_ctx)

        events = await _events(
            llm.stream_async("hi", schema=TurnAction, text_field="reason")
        )

        deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert "".join(deltas) == "Got it, one moment"
        assert len(deltas) > 1, "arrived in one piece, so nothing streamed"
        assert all("{" not in d and '"' not in d for d in deltas)
        assert isinstance(events[-1], StreamDone)
        assert events[-1].text == "Got it, one moment"

    async def test_openai_does_not_leak_schema_json_as_assistant_text(self) -> None:
        # With text_format set, output_text IS the JSON body. Forwarding those
        # deltas printed raw JSON into the conversation, which is what a live run
        # showed before this was fixed.
        from arksim.llms.chat.providers.openai import OpenAILLM

        final = SimpleNamespace(
            output_text='{"action":"ask_user"}',
            output_parsed=TurnAction(action="ask_user"),
            usage=None,
        )
        sdk_events = [
            SimpleNamespace(type="response.output_text.delta", delta='{"action"'),
            SimpleNamespace(type="response.output_text.delta", delta=':"ask_user"}'),
        ]
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = _returns(_FakeStream(sdk_events, final))
        stream_ctx.__aexit__ = _returns(False)

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="gpt-4o")
        llm.async_client = MagicMock()
        llm.async_client.responses.stream = MagicMock(return_value=stream_ctx)

        events = await _events(llm.stream_async("hi", schema=TurnAction))

        assert not [e for e in events if isinstance(e, TextDelta)]
        assert [type(e) for e in events] == [Action, StreamDone]
        assert events[0].payload.action == "ask_user"

    @requires_google
    async def test_google_rejects_a_structured_body_that_loops(self) -> None:
        # The provider boundary, not the helper: RunawayGuard is tested directly,
        # but nothing else pins which providers use it.
        from arksim.llms.chat.providers.google import GoogleLLM

        chunks = [
            SimpleNamespace(
                text='{"action": "generate", "reason": "' + "910482" * 2_000,
                usage_metadata=None,
            )
        ]

        with patch("arksim.llms.chat.providers.google.genai"):
            llm = GoogleLLM(model="gemini-2.5-pro")
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = _returns(_async_iter(chunks))

        with pytest.raises(TruncatedActionError) as caught:
            await _events(llm.stream_async("hi", schema=TurnAction))

        assert "repeated" in str(caught.value)

    async def test_openai_accepts_a_structured_body_that_repeats(self) -> None:
        # The other half of the same decision. Repetition detection cannot tell a
        # loop from content that legitimately repeats, so it is Google-only and a
        # complete OpenAI body parses however repetitive it is.
        from arksim.llms.chat.providers.openai import OpenAILLM

        body = '{"action": "generate", "reason": "' + "910482" * 2_000 + '"}'
        final = SimpleNamespace(
            output_text=body,
            output_parsed=TurnAction(action="generate", reason="910482" * 2_000),
            usage=None,
        )
        sdk_events = [
            SimpleNamespace(type="response.output_text.delta", delta=body[i : i + 400])
            for i in range(0, len(body), 400)
        ]
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = _returns(_FakeStream(sdk_events, final))
        stream_ctx.__aexit__ = _returns(False)

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="gpt-4o")
        llm.async_client = MagicMock()
        llm.async_client.responses.stream = MagicMock(return_value=stream_ctx)

        events = await _events(llm.stream_async("hi", schema=TurnAction))

        assert [type(e) for e in events] == [Action, StreamDone]
        assert events[0].payload.action == "generate"

    @requires_anthropic
    async def test_anthropic_streams_content_block_deltas(self) -> None:
        from arksim.llms.chat.providers.anthropic import AnthropicLLM

        final = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=5,
                output_tokens=3,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
        )
        sdk_events = [
            SimpleNamespace(type="message_start"),
            SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(text="one ")
            ),
            SimpleNamespace(
                type="content_block_delta", delta=SimpleNamespace(text="two")
            ),
            SimpleNamespace(type="message_stop"),
        ]

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = _returns(
            _FakeStream(sdk_events, final, final_attr="get_final_message")
        )
        stream_ctx.__aexit__ = _returns(False)

        with patch("arksim.llms.chat.providers.anthropic.anthropic"):
            llm = AnthropicLLM(model="claude-opus-5")
        llm.async_client = MagicMock()
        llm.async_client.messages.stream = MagicMock(return_value=stream_ctx)

        with patch("arksim.llms.chat.providers.anthropic.track_usage") as usage:
            events = await _events(llm.stream_async("hi"))

        assert [e.text for e in events if isinstance(e, TextDelta)] == ["one ", "two"]
        assert events[-1].text == "one two"
        usage.assert_called_once()

    @requires_google
    async def test_google_does_not_leak_schema_json_as_assistant_text(self) -> None:
        from arksim.llms.chat.providers.google import GoogleLLM

        # Google streams the JSON body itself when a schema is set. Forwarding it
        # as text would print raw JSON into the conversation.
        chunks = [
            SimpleNamespace(text='{"action": "gen', usage_metadata=None),
            SimpleNamespace(
                text='erate"}',
                usage_metadata=SimpleNamespace(
                    prompt_token_count=8,
                    total_token_count=12,
                    candidates_token_count=4,
                    thoughts_token_count=0,
                    cached_content_token_count=0,
                ),
            ),
        ]

        with patch("arksim.llms.chat.providers.google.genai"):
            llm = GoogleLLM(model="gemini-2.5-pro")
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = _returns(_async_iter(chunks))

        with patch("arksim.llms.chat.providers.google.track_usage"):
            events = await _events(llm.stream_async("hi", schema=TurnAction))

        assert not [e for e in events if isinstance(e, TextDelta)]
        assert [type(e) for e in events] == [Action, StreamDone]
        assert events[0].payload.action == "generate"

    @requires_google
    async def test_google_says_so_when_the_body_was_cut_off(self) -> None:
        from arksim.llms.chat.providers.google import GoogleLLM

        # A body that ran out of room mid-string. Pydantic calls this an ordinary
        # validation error and quotes the whole body back in it; observed at
        # 218KB, which then travelled to the browser as the reason the turn
        # failed. The caller can only ask again, so it is told that and no more.
        chunks = [
            SimpleNamespace(
                text='{"action": "generate", "reason": "because', usage_metadata=None
            ),
            SimpleNamespace(
                text=" it is",
                usage_metadata=None,
                candidates=[
                    SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))
                ],
            ),
        ]

        with patch("arksim.llms.chat.providers.google.genai"):
            llm = GoogleLLM(model="gemini-2.5-pro")
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = _returns(_async_iter(chunks))

        with pytest.raises(TruncatedActionError) as caught:
            await _events(llm.stream_async("hi", schema=TurnAction))

        assert "MAX_TOKENS" in str(caught.value)
        # The body itself stays out of it, whatever length it reached.
        assert "because it is" not in str(caught.value)

    @requires_google
    async def test_google_still_reports_a_schema_mismatch_as_one(self) -> None:
        from arksim.llms.chat.providers.google import GoogleLLM

        # Complete JSON that does not match the schema is a prompt or a model to
        # fix, not something to ask again for. It has to keep raising what it
        # always raised, or a real bug arrives dressed as a transient one.
        chunks = [
            SimpleNamespace(text='{"reason": "no action here"}', usage_metadata=None)
        ]

        with patch("arksim.llms.chat.providers.google.genai"):
            llm = GoogleLLM(model="gemini-2.5-pro")
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = _returns(_async_iter(chunks))

        with pytest.raises(ValidationError):
            await _events(llm.stream_async("hi", schema=TurnAction))

    @requires_google
    async def test_google_streams_prose_when_no_schema(self) -> None:
        from arksim.llms.chat.providers.google import GoogleLLM

        chunks = [
            SimpleNamespace(text="one ", usage_metadata=None),
            SimpleNamespace(text="two", usage_metadata=None),
        ]

        with patch("arksim.llms.chat.providers.google.genai"):
            llm = GoogleLLM(model="gemini-2.5-pro")
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = _returns(_async_iter(chunks))

        events = await _events(llm.stream_async("hi"))
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["one ", "two"]
        assert events[-1].text == "one two"

    @requires_google
    async def test_google_cancellation_stops_mid_stream(self) -> None:
        from arksim.llms.chat.providers.google import GoogleLLM

        chunks = [
            SimpleNamespace(text="one ", usage_metadata=None),
            SimpleNamespace(text="two", usage_metadata=None),
        ]
        cancelled = {"value": False}

        with patch("arksim.llms.chat.providers.google.genai"):
            llm = GoogleLLM(model="gemini-2.5-pro")
        llm.client = MagicMock()
        llm.client.aio.models.generate_content_stream = _returns(_async_iter(chunks))

        events: list[StreamEvent] = []
        async for event in llm.stream_async("hi", cancel=lambda: cancelled["value"]):
            events.append(event)
            cancelled["value"] = True

        assert isinstance(events[-1], StreamDone)
        assert events[-1].cancelled is True


def _returns(value: object) -> Callable[..., object]:
    """An async callable returning ``value``, for stubbing SDK entry points."""

    async def call(*args: object, **kwargs: object) -> object:
        return value

    return call


class _FakeStream:
    """Stands in for an SDK streaming context's iterator plus final accessor."""

    def __init__(
        self,
        events: list[object],
        final: object,
        final_attr: str = "get_final_response",
    ) -> None:
        self._events = events
        self.closed = False
        setattr(self, final_attr, _returns(final))

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True
