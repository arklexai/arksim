# SPDX-License-Identifier: Apache-2.0
"""Tests for the hosted web tool offered to the utterance generator.

The point is to react to a link the agent sent. anthropic sends its fetch tool
(`web_fetch`); google and openai send search (`google_search`, `web_search`),
whose reasoning models open pages as part of searching. A live fetch of a
bot-protected page comes back empty, so search is the more reliable route where
both exist.

A wrong tool spec is a request-time 400, not a local failure, so these pin the
exact spec each provider sends. Anthropic also gets coverage for the two
response shapes tools introduce: interleaved blocks, and a paused turn.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel

from arksim.llms.chat.providers.azure_openai import AzureOpenAILLM
from arksim.llms.chat.providers.openai import OpenAILLM
from arksim.scenario.entities import Scenarios
from arksim.simulation_engine import simulator as simulator_module
from arksim.simulation_engine.entities import SimulationInput, SimulationParams
from arksim.simulation_engine.simulator import run_simulation
from arksim.simulation_engine.utils.prompts import (
    DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE as SIMULATED_USER_PROMPT,
)

# The provider SDKs are optional extras, and CI installs only ``.[dev]``.
try:
    from arksim.llms.chat.providers.anthropic import (
        MAX_FETCH_CONTENT_TOKENS,
        MAX_PAUSE_CONTINUATIONS,
        MAX_WEB_FETCHES,
        WEB_FETCH_TOOL,
        AnthropicLLM,
        _extract_text,
    )
except ImportError:
    AnthropicLLM = None
try:
    from arksim.llms.chat.providers.google import GoogleLLM
except ImportError:
    GoogleLLM = None

requires_anthropic = pytest.mark.skipif(
    AnthropicLLM is None, reason="anthropic not installed"
)
requires_google = pytest.mark.skipif(
    GoogleLLM is None, reason="google-genai not installed"
)

_MESSAGES = [{"role": "user", "content": "any good vacuums?"}]


class _Reply(BaseModel):
    text: str


def _openai_llm(model: str = "gpt-5.1") -> OpenAILLM:
    with (
        patch("arksim.llms.chat.providers.openai.OpenAI"),
        patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
    ):
        return OpenAILLM(model=model)


def _anthropic_llm(model: str = "claude-opus-5") -> AnthropicLLM:
    with (
        patch("anthropic.Anthropic"),
        patch("anthropic.AsyncAnthropic"),
    ):
        return AnthropicLLM(model=model)


def _google_llm(model: str = "gemini-2.5-pro") -> GoogleLLM:
    with patch("arksim.llms.chat.providers.google.genai.Client"):
        return GoogleLLM(model=model)


def _azure_llm() -> AzureOpenAILLM:
    with (
        patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
                "AZURE_OPENAI_API_VERSION": "2024-02-01",
                "AZURE_CLIENT_ID": "fake-client-id",
            },
        ),
        patch(
            "arksim.llms.chat.providers.azure_openai.get_azure_token_provider",
            return_value=lambda: "fake-token",
        ),
        patch("arksim.llms.chat.providers.azure_openai.AzureOpenAI"),
        patch("arksim.llms.chat.providers.azure_openai.AsyncAzureOpenAI"),
    ):
        return AzureOpenAILLM(model="gpt-4")


class TestSupportsWebSearchFlag:
    """Callers branch on this flag, so it has to match reality."""

    @requires_anthropic
    @requires_google
    def test_hosted_search_providers_advertise_support(self) -> None:
        assert OpenAILLM.supports_web_search is True
        assert AnthropicLLM.supports_web_search is True
        assert GoogleLLM.supports_web_search is True

    def test_azure_does_not_advertise_support(self) -> None:
        assert AzureOpenAILLM.supports_web_search is False


class TestOpenAIWebSearch:
    def test_no_tool_by_default(self) -> None:
        params = _openai_llm()._prepare_params(_MESSAGES)
        assert "tools" not in params

    def test_offers_the_hosted_tool_when_enabled(self) -> None:
        params = _openai_llm()._prepare_params(_MESSAGES, web_search=True)
        assert params["tools"] == [{"type": "web_search"}]

    def test_coexists_with_structured_output(self) -> None:
        """Unlike Gemini, the Responses API takes a tool and a text format."""
        params = _openai_llm()._prepare_params(
            _MESSAGES, schema=_Reply, web_search=True
        )
        assert params["tools"] == [{"type": "web_search"}]
        assert params["text_format"] is _Reply


@requires_anthropic
class TestAnthropicWebFetch:
    def test_no_tool_by_default(self) -> None:
        params = _anthropic_llm()._prepare_params(_MESSAGES)
        assert "tools" not in params

    # The revision is dated and there is no unversioned alias, so the exact
    # string matters: a revision the model rejects is a request-time 400.
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "some-unreleased-model",
        ],
    )
    def test_same_tool_for_every_model(self, model: str) -> None:
        """Pinned to the broadly available revision, so no model needs a lookup
        and a new model id cannot go stale into the wrong one."""
        params = _anthropic_llm(model)._prepare_params(_MESSAGES, web_search=True)
        assert params["tools"] == [WEB_FETCH_TOOL]

    def test_the_tool_is_fetch_not_search(self) -> None:
        """Search would answer a keyword query about the product; the agent sent
        a URL, and fetch is the tool that reads that page. It is also free."""
        assert WEB_FETCH_TOOL["type"] == "web_fetch_20250910"
        assert WEB_FETCH_TOOL["name"] == "web_fetch"

    def test_tool_spec_caps_both_uses_and_page_size(self) -> None:
        """One utterance should not turn into a research session, and one long
        page should not crowd the conversation out of the context window."""
        assert WEB_FETCH_TOOL["max_uses"] == MAX_WEB_FETCHES
        assert WEB_FETCH_TOOL["max_content_tokens"] == MAX_FETCH_CONTENT_TOKENS

    def test_citations_stay_off(self) -> None:
        """Off is the fetch default, unlike search where they are forced on: a
        customer reacts to a page, it does not cite one."""
        assert "citations" not in WEB_FETCH_TOOL


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _fetch_blocks() -> list[SimpleNamespace]:
    """The block pair Anthropic returns for one server-side fetch."""
    return [
        SimpleNamespace(type="server_tool_use", name="web_fetch", input={}),
        SimpleNamespace(type="web_fetch_tool_result", content=[]),
    ]


def _response(
    blocks: list[SimpleNamespace], stop_reason: str = "end_turn"
) -> SimpleNamespace:
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


@requires_anthropic
class TestAnthropicTextExtraction:
    def test_joins_text_blocks_and_skips_tool_blocks(self) -> None:
        response = _response(
            [_text_block("Let me check. "), *_fetch_blocks(), _text_block("It is up.")]
        )
        assert _extract_text(response) == "Let me check. It is up."

    def test_survives_a_leading_tool_block(self) -> None:
        """A fetch-first turn puts a block with no .text at index 0."""
        response = _response([*_fetch_blocks(), _text_block("Found it.")])
        assert _extract_text(response) == "Found it."

    def test_plain_response_is_unchanged(self) -> None:
        assert _extract_text(_response([_text_block("hello")])) == "hello"


@requires_anthropic
class TestAnthropicPauseTurn:
    """Sending a paused turn's partial as the utterance would hand the agent a
    truncated or empty message, so the call resumes it instead."""

    def test_resumes_a_paused_turn(self) -> None:
        llm = _anthropic_llm()
        paused = _response(
            [_text_block("Checking that link"), *_fetch_blocks()],
            stop_reason="pause_turn",
        )
        finished = _response([_text_block(" - it is a vacuum, yes.")])
        llm.async_client = MagicMock()
        llm.async_client.messages.create = AsyncMock(side_effect=[paused, finished])

        result = asyncio.run(llm.call_async(_MESSAGES, web_search=True))

        assert result == "Checking that link - it is a vacuum, yes."
        assert llm.async_client.messages.create.await_count == 2
        # The resumed request has to carry the paused turn back.
        resumed = llm.async_client.messages.create.await_args.kwargs["messages"]
        assert resumed[-1]["role"] == "assistant"
        assert resumed[-1]["content"] == paused.content

    def test_continuations_are_bounded(self) -> None:
        """A turn that never settles must not spin."""
        llm = _anthropic_llm()
        llm.async_client = MagicMock()
        llm.async_client.messages.create = AsyncMock(
            return_value=_response([_text_block("x")], stop_reason="pause_turn")
        )

        asyncio.run(llm.call_async(_MESSAGES, web_search=True))

        assert (
            llm.async_client.messages.create.await_count == MAX_PAUSE_CONTINUATIONS + 1
        )

    def test_finished_turn_makes_one_call(self) -> None:
        llm = _anthropic_llm()
        llm.async_client = MagicMock()
        llm.async_client.messages.create = AsyncMock(
            return_value=_response([_text_block("done")])
        )

        assert asyncio.run(llm.call_async(_MESSAGES)) == "done"
        assert llm.async_client.messages.create.await_count == 1


@requires_google
class TestGoogleUrlContext:
    def test_no_tool_by_default(self) -> None:
        config = _google_llm()._prepare_params(_MESSAGES).get("config")
        assert config is None or not getattr(config, "tools", None)

    def test_offers_google_search_when_enabled(self) -> None:
        """google_search reads the index, so it still reaches pages whose
        origin refuses a live url_context fetch."""
        params = _google_llm()._prepare_params(_MESSAGES, web_search=True)
        tools = params["config"].tools
        assert len(tools) == 1
        assert tools[0].google_search is not None
        assert tools[0].url_context is None

    def test_rejects_structured_output_with_a_tool(self) -> None:
        """Gemini 400s on schema + tool; say so here instead of mid-run."""
        with pytest.raises(ValueError, match="structured output and hosted tools"):
            _google_llm()._prepare_params(_MESSAGES, schema=_Reply, web_search=True)


class TestAzureRejectsWebSearch:
    """Chat Completions has no hosted search; fail rather than skip silently."""

    @patch("time.sleep", return_value=None)
    def test_call_raises(self, _sleep: MagicMock) -> None:
        with pytest.raises(ValueError, match="does not support hosted web search"):
            _azure_llm().call(_MESSAGES, web_search=True)

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_call_async_raises(self, _sleep: AsyncMock) -> None:
        with pytest.raises(ValueError, match="does not support hosted web search"):
            asyncio.run(_azure_llm().call_async(_MESSAGES, web_search=True))


def _render_prompt(web_search: bool) -> str:
    return (
        SandboxedEnvironment()
        .from_string(SIMULATED_USER_PROMPT)
        .render(
            scenario={"user_profile": "You are shopping for a vacuum."},
            web_search=web_search,
        )
    )


class TestSimulatedUserPrompt:
    """The tool is advertised by the API, so the prompt only has to not forbid
    using it. The permission is phrased as customer behavior, not tool usage."""

    def test_opening_a_link_is_allowed_when_the_tool_is_offered(self) -> None:
        prompt = _render_prompt(web_search=True)
        assert "you can open it before replying" in prompt

    def test_the_permission_is_scoped_to_links(self) -> None:
        """A fetch tool can only follow a URL. Inviting a check of a bare
        product or price is inviting an invented "I looked it up and..."."""
        prompt = _render_prompt(web_search=True)
        assert "When the agent sends you a link" in prompt

    def test_no_such_rule_without_the_tool(self) -> None:
        """Otherwise the prompt invites a lookup the model cannot perform, and
        an invented "I checked and it says..." is worse than not checking."""
        assert "open it before replying" not in _render_prompt(web_search=False)

    def test_never_names_a_tool(self) -> None:
        assert "web search" not in _render_prompt(web_search=True).lower()

    def test_fabrication_rule_is_scoped_to_the_user_s_own_details(self) -> None:
        """It used to ban information beyond the profile outright, which reads
        as a ban on looking anything up - the tool then went unused."""
        for flag in (True, False):
            prompt = _render_prompt(web_search=flag)
            assert "Do not invent details about yourself or your situation" in prompt
            assert "Do not make up information beyond your profile" not in prompt

    def test_rules_stay_one_clean_list(self) -> None:
        """The conditional must not leave a blank line mid-list."""
        rules = [
            ln
            for ln in _render_prompt(web_search=True).splitlines()
            if ln.startswith("-")
        ]
        body = _render_prompt(web_search=True).split("Rules:\n", 1)[1]
        assert len(rules) == 6
        assert "\n\n" not in body.strip()


def _settings(provider: str) -> SimulationInput:
    return SimulationInput(
        provider=provider,
        agent_config={
            "agent_type": "chat_completions",
            "agent_name": "test-agent",
            "api_config": {
                "endpoint": "http://localhost/v1/chat/completions",
                "headers": {},
                "body": {"model": "test", "messages": []},
            },
        },
    )


def _resolve_web_search(
    provider: str, supports_web_search: bool
) -> tuple[bool, list[str]]:
    """Return the web_search value handed to the Simulator, plus any warnings.

    With no scenarios the run stops at the "nothing completed" error, well past
    the resolution step.
    """
    llm = MagicMock()
    llm.supports_web_search = supports_web_search
    captured: list[bool] = []
    warnings: list[str] = []

    real_params = SimulationParams

    def spy(**kwargs: object) -> SimulationParams:
        captured.append(bool(kwargs["web_search"]))
        return real_params(**kwargs)

    with (
        patch("arksim.simulation_engine.simulator.LLM", return_value=llm),
        patch("arksim.simulation_engine.simulator.SimulationParams", side_effect=spy),
        patch.object(
            simulator_module.logger,
            "warning",
            side_effect=lambda msg, *a: warnings.append(msg % a),
        ),
        pytest.raises(RuntimeError, match="no conversations"),
    ):
        asyncio.run(
            run_simulation(
                _settings(provider),
                Scenarios(schema_version="v1", scenarios=[]),
            )
        )
    return captured[0], warnings


class TestWebSearchResolution:
    """On by default, so an unsupported provider is a configuration, not a
    failed run."""

    def test_on_by_default_for_a_supported_provider(self) -> None:
        resolved, warnings = _resolve_web_search("openai", supports_web_search=True)
        assert resolved is True
        assert warnings == []

    def test_dropped_with_a_warning_for_an_unsupported_provider(self) -> None:
        resolved, warnings = _resolve_web_search("azure", supports_web_search=False)
        assert resolved is False
        assert any("azure" in w and "web fetch or search" in w for w in warnings)

    def test_default_is_on(self) -> None:
        assert _settings("openai").web_search is True
