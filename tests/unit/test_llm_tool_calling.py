# SPDX-License-Identifier: Apache-2.0
"""BaseLLM's tool-calling channel, and the Google and OpenAI implementations.

The channel exists because ``call`` and ``call_async`` have nowhere to put a
tool declaration: a provider builds its request from messages and a schema and
drops the rest. The two things worth pinning here are that a provider without
the channel says so rather than answering as though no tool were on offer, and
that a completed round replays in a shape the model accepts.

That second one is not theoretical. A thinking model returns an opaque
``thought_signature`` with each function call and rejects any later turn that
replays the call without it:

    400 INVALID_ARGUMENT: Function call is missing a thought_signature in
    functionCall parts ... function call `default_api:return_policy`, position 2

Position 2 is the tell. The first round always worked and the second failed the
moment the first call was replayed stripped of its signature, so any caller
making two rounds of calls never got past its first lookup.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from arksim.llms.chat.base.base_llm import BaseLLM
from arksim.llms.chat.base.tools import (
    ToolCall,
    ToolCallingUnsupported,
    ToolDeclaration,
    ToolExchange,
)


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

pytestmark = pytest.mark.unit

SIGNATURE = b"opaque-thought-signature-bytes"


def _part(name: str, args: dict[str, Any], signature: bytes | None) -> SimpleNamespace:
    return SimpleNamespace(
        function_call=SimpleNamespace(name=name, args=args),
        thought_signature=signature,
    )


def _response(*parts: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
        usage_metadata=None,
    )


class FakeClient:
    """Records what the provider sent and replays queued responses."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Models:
            async def generate_content(self, **kwargs: object) -> SimpleNamespace:
                outer.calls.append(kwargs)
                return outer._responses.pop(0)

        self.aio = SimpleNamespace(models=_Models())


def _google(responses: list[SimpleNamespace]) -> tuple[Any, FakeClient]:
    from arksim.llms.chat.providers.google import GoogleLLM

    client = FakeClient(responses)
    return GoogleLLM("gemini-3.5-flash", client=client), client


def _price_tool() -> ToolDeclaration:
    return ToolDeclaration(
        name="product_price",
        description="Look up a price.",
        parameters={"type": "object", "properties": {"sku": {"type": "string"}}},
    )


def _function_call_parts(contents: list[Any]) -> list[Any]:
    out = []
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "function_call", None) is not None:
                out.append(part)
    return out


class TestTheDefaultRefuses:
    async def test_a_provider_without_the_channel_says_so(self) -> None:
        class TextOnly(BaseLLM):
            def call(self, messages, schema=None, **kwargs) -> str:  # noqa: ANN001, ANN003
                return ""

            async def call_async(self, messages, schema=None, **kwargs) -> str:  # noqa: ANN001, ANN003
                return ""

        llm = TextOnly("some-model")
        assert llm.supports_tool_calling is False
        # Refused rather than answered. Returning no calls would be
        # indistinguishable from the model deciding it needed none, and a
        # caller depending on retrieved data would score against nothing.
        with pytest.raises(ToolCallingUnsupported):
            await llm.call_with_tools_async("hi", tools=[_price_tool()])

    @requires_google
    def test_google_advertises_the_channel(self) -> None:
        from arksim.llms.chat.providers.google import GoogleLLM

        assert GoogleLLM.supports_tool_calling is True


@requires_google
class TestGoogleToolCalling:
    async def test_it_returns_the_calls_and_runs_none_of_them(self) -> None:
        llm, client = _google(
            [_response(_part("product_price", {"sku": "PAINT-1G"}, SIGNATURE))]
        )
        calls = await llm.call_with_tools_async(
            [{"role": "system", "content": "p"}, {"role": "user", "content": "c"}],
            tools=[_price_tool()],
        )
        assert [c.name for c in calls] == ["product_price"]
        assert calls[0].arguments == {"sku": "PAINT-1G"}
        assert calls[0].provider_state == SIGNATURE

        config = client.calls[0]["config"]
        # The SDK will execute a function for the model given the chance, which
        # would act before the caller could authorize it.
        assert config.automatic_function_calling.disable is True
        assert config.system_instruction == "p"

    async def test_prose_with_no_call_means_done(self) -> None:
        llm, _ = _google([_response()])
        assert (await llm.call_with_tools_async("c", tools=[_price_tool()])) == []

    async def test_a_replayed_call_carries_the_signature_it_was_issued_with(
        self,
    ) -> None:
        llm, client = _google(
            [
                _response(_part("product_price", {"sku": "PAINT-1G"}, SIGNATURE)),
                _response(),
            ]
        )
        tools = [_price_tool()]
        issued = await llm.call_with_tools_async("c", tools=tools)

        # The caller hands back the call it was given, which is what carries
        # the signature. This is the whole bug in one round trip.
        await llm.call_with_tools_async(
            "c",
            tools=tools,
            history=[ToolExchange(call=issued[0], observation='{"price_usd": 42.98}')],
        )

        replayed = _function_call_parts(client.calls[1]["contents"])
        assert len(replayed) == 1
        assert replayed[0].function_call.name == "product_price"
        assert replayed[0].thought_signature == SIGNATURE

    async def test_each_signature_follows_its_own_call(self) -> None:
        llm, client = _google(
            [
                _response(
                    _part("product_price", {"sku": "A"}, b"sig-A"),
                    _part("product_price", {"sku": "B"}, b"sig-B"),
                ),
                _response(),
            ]
        )
        tools = [_price_tool()]
        issued = await llm.call_with_tools_async("c", tools=tools)
        by_sku = {c.arguments["sku"]: c for c in issued}

        # Replayed out of order on purpose: a signature that is looked up
        # rather than carried can answer for the wrong call.
        await llm.call_with_tools_async(
            "c",
            tools=tools,
            history=[
                ToolExchange(call=by_sku["B"], observation="b"),
                ToolExchange(call=by_sku["A"], observation="a"),
            ],
        )

        replayed = _function_call_parts(client.calls[1]["contents"])
        assert [p.function_call.args["sku"] for p in replayed] == ["B", "A"]
        assert [p.thought_signature for p in replayed] == [b"sig-B", b"sig-A"]

    async def test_calls_issued_in_one_turn_replay_as_one_turn(self) -> None:
        """One signature per turn, carried on its first part.

        Splitting a two-call turn leaves the second call with none, and the
        API refuses the request. The turn is the unit that goes back.
        """
        llm, client = _google(
            [
                _response(
                    _part("product_price", {"sku": "A"}, SIGNATURE),
                    # As the API sends it: the sibling part carries none.
                    _part("product_price", {"sku": "B"}, None),
                ),
                _response(),
            ]
        )
        tools = [_price_tool()]
        issued = await llm.call_with_tools_async("c", tools=tools)
        one_turn = [replace(call, turn_index=1) for call in issued]

        await llm.call_with_tools_async(
            "c",
            tools=tools,
            history=[
                ToolExchange(call=one_turn[0], observation="a"),
                ToolExchange(call=one_turn[1], observation="b"),
            ],
        )

        contents = client.calls[1]["contents"]
        model_turns = [c for c in contents if getattr(c, "role", None) == "model"]
        assert len(model_turns) == 1, "two calls, one turn, one model content"
        assert len(model_turns[0].parts) == 2
        assert [p.thought_signature for p in model_turns[0].parts] == [SIGNATURE, None]

        # The answers pair with the calls positionally, so they travel together
        # and in the same order.
        responses = [
            part
            for content in contents
            for part in getattr(content, "parts", None) or []
            if getattr(part, "function_response", None) is not None
        ]
        assert [r.function_response.response["result"] for r in responses] == ["a", "b"]

    async def test_calls_with_no_turn_recorded_replay_one_per_turn(self) -> None:
        # Grouping on a default would merge rounds the model issued
        # separately into a turn it never took.
        llm, client = _google(
            [
                _response(_part("product_price", {"sku": "A"}, SIGNATURE)),
                _response(_part("product_price", {"sku": "B"}, b"sig-B")),
                _response(),
            ]
        )
        tools = [_price_tool()]
        first = await llm.call_with_tools_async("c", tools=tools)
        second = await llm.call_with_tools_async(
            "c", tools=tools, history=[ToolExchange(call=first[0], observation="a")]
        )
        await llm.call_with_tools_async(
            "c",
            tools=tools,
            history=[
                ToolExchange(call=first[0], observation="a"),
                ToolExchange(call=second[0], observation="b"),
            ],
        )

        contents = client.calls[2]["contents"]
        model_turns = [c for c in contents if getattr(c, "role", None) == "model"]
        assert len(model_turns) == 2
        assert all(len(t.parts) == 1 for t in model_turns)

    async def test_a_model_that_returns_no_signature_still_replays(self) -> None:
        # Non-thinking models omit it, and None is what the SDK expects then.
        llm, client = _google(
            [_response(_part("product_price", {"sku": "A"}, None)), _response()]
        )
        tools = [_price_tool()]
        issued = await llm.call_with_tools_async("c", tools=tools)
        assert issued[0].provider_state is None

        await llm.call_with_tools_async(
            "c", tools=tools, history=[ToolExchange(call=issued[0], observation="a")]
        )
        replayed = _function_call_parts(client.calls[1]["contents"])
        assert replayed[0].thought_signature is None

    async def test_the_provider_remembers_nothing_between_rounds(self) -> None:
        """State rides on the call, so it cannot leak or be misattributed.

        One provider instance may serve many unrelated loops. A signature map
        kept on the instance would hand a later loop a signature issued during
        an earlier one, which is why this is carried rather than remembered.
        """
        llm, client = _google(
            [_response(_part("product_price", {"sku": "A"}, SIGNATURE)), _response()]
        )
        tools = [_price_tool()]
        await llm.call_with_tools_async("c", tools=tools)

        # A call rebuilt without the state gets no signature, even though this
        # instance has seen one for exactly these arguments.
        await llm.call_with_tools_async(
            "c",
            tools=tools,
            history=[
                ToolExchange(
                    call=ToolCall(name="product_price", arguments={"sku": "A"}),
                    observation="a",
                )
            ],
        )
        replayed = _function_call_parts(client.calls[1]["contents"])
        assert replayed[0].thought_signature is None

    async def test_a_bare_message_is_not_replayed_as_a_tool_result(self) -> None:
        """A nudge is the caller's own turn, not an answer to any call.

        Dressed as a function_response it would claim to answer a call the
        model never made, which providers reject.
        """
        llm, client = _google([_response()])
        await llm.call_with_tools_async(
            "c",
            tools=[_price_tool()],
            history=[ToolExchange(observation="still need the price", is_error=True)],
        )
        contents = client.calls[0]["contents"]
        assert _function_call_parts(contents) == []
        texts = [
            part.text
            for content in contents
            for part in (getattr(content, "parts", None) or [])
            if getattr(part, "text", None)
        ]
        assert "still need the price" in texts


# ---- OpenAI ----
#
# The Responses API has no signature to carry, but it has an id: an output item
# must name the call it answers, so the call_id rides in provider_state and the
# two are rebuilt as a pair on replay.


def _openai_call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call", name=name, arguments=arguments, call_id=call_id
    )


def _openai_response(*items: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(output=list(items), usage=None)


class FakeOpenAIClient:
    """Records what the provider sent and replays queued responses."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Responses:
            async def create(self, **kwargs: object) -> SimpleNamespace:
                outer.calls.append(kwargs)
                return outer._responses.pop(0)

        self.responses = _Responses()


def _openai(responses: list[SimpleNamespace]) -> tuple[Any, FakeOpenAIClient]:
    from arksim.llms.chat.providers.openai import OpenAILLM

    # The constructor builds real clients from ambient credentials, which a
    # test has none of; the async one is swapped for the fake below.
    with (
        patch("arksim.llms.chat.providers.openai.OpenAI"),
        patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
    ):
        llm = OpenAILLM("gpt-5")
    client = FakeOpenAIClient(responses)
    llm.async_client = client
    return llm, client


def _items_of_type(items: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [item for item in items if item.get("type") == kind]


class TestOpenAIToolCalling:
    def test_openai_advertises_the_channel(self) -> None:
        from arksim.llms.chat.providers.openai import OpenAILLM

        assert OpenAILLM.supports_tool_calling is True

    async def test_it_returns_the_calls_and_runs_none_of_them(self) -> None:
        llm, client = _openai(
            [
                _openai_response(
                    _openai_call("product_price", '{"sku": "PAINT-1G"}', "call_1")
                )
            ]
        )
        calls = await llm.call_with_tools_async(
            [{"role": "system", "content": "p"}, {"role": "user", "content": "c"}],
            tools=[_price_tool()],
        )
        assert [c.name for c in calls] == ["product_price"]
        assert calls[0].arguments == {"sku": "PAINT-1G"}
        assert calls[0].provider_state == b"call_1"

        sent = client.calls[0]
        assert sent["input"][0] == {"role": "system", "content": "p"}
        assert sent["tool_choice"] == "auto"
        # Third-party schemas as written. Strict mode would refuse any object
        # that does not forbid extra keys, which most tool schemas do not.
        assert sent["tools"] == [
            {
                "type": "function",
                "name": "product_price",
                "description": "Look up a price.",
                "parameters": _price_tool().parameters,
                "strict": False,
            }
        ]

    async def test_prose_with_no_call_means_done(self) -> None:
        llm, _ = _openai([_openai_response()])
        assert (await llm.call_with_tools_async("c", tools=[_price_tool()])) == []

    async def test_a_replayed_call_and_its_output_share_the_issued_id(
        self,
    ) -> None:
        llm, client = _openai(
            [
                _openai_response(
                    _openai_call("product_price", '{"sku": "PAINT-1G"}', "call_1")
                ),
                _openai_response(),
            ]
        )
        tools = [_price_tool()]
        issued = await llm.call_with_tools_async("c", tools=tools)

        await llm.call_with_tools_async(
            "c",
            tools=tools,
            history=[ToolExchange(call=issued[0], observation='{"price_usd": 42.98}')],
        )

        sent = client.calls[1]["input"]
        [call] = _items_of_type(sent, "function_call")
        [output] = _items_of_type(sent, "function_call_output")
        assert call["call_id"] == output["call_id"] == "call_1"
        assert call["name"] == "product_price"
        assert json.loads(call["arguments"]) == {"sku": "PAINT-1G"}
        assert output["output"] == '{"price_usd": 42.98}'
        # The pair follows the messages, in the order the rounds happened.
        assert sent.index(call) < sent.index(output)

    async def test_a_call_rebuilt_without_its_id_still_pairs(self) -> None:
        llm, client = _openai([_openai_response()])
        await llm.call_with_tools_async(
            "c",
            tools=[_price_tool()],
            history=[
                ToolExchange(
                    call=ToolCall(name="product_price", arguments={"sku": "A"}),
                    observation="a",
                )
            ],
        )
        sent = client.calls[0]["input"]
        [call] = _items_of_type(sent, "function_call")
        [output] = _items_of_type(sent, "function_call_output")
        assert call["call_id"] and call["call_id"] == output["call_id"]

    async def test_a_bare_message_is_not_replayed_as_a_tool_result(self) -> None:
        llm, client = _openai([_openai_response()])
        await llm.call_with_tools_async(
            "c",
            tools=[_price_tool()],
            history=[ToolExchange(observation="still need the price", is_error=True)],
        )
        sent = client.calls[0]["input"]
        assert _items_of_type(sent, "function_call") == []
        assert _items_of_type(sent, "function_call_output") == []
        assert {"role": "user", "content": "still need the price"} in sent
