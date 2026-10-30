# SPDX-License-Identifier: Apache-2.0
"""Tests for per-request headers and turn hooks around ``agent.execute()``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from arksim.config import AgentConfig
from arksim.scenario import Scenario
from arksim.simulation_engine.agent.clients.chat_completions import (
    ChatCompletionsAgent,
)
from arksim.simulation_engine.entities import ConversationExecutionError
from arksim.simulation_engine.simulator import Simulator
from arksim.tracing.context import (
    request_headers,
    reset_request_headers,
    set_request_headers,
)

HEADERS = {"traceparent": "00-abc-def-01", "baggage": "arklex.capture.id=cap-1"}


# ── Contextvar helpers and the chat_completions connector ──


def test_set_and_reset_request_headers() -> None:
    assert request_headers.get() is None
    token = set_request_headers(HEADERS)
    assert request_headers.get() == HEADERS
    reset_request_headers(token)
    assert request_headers.get() is None


async def test_chat_completions_headers_merged_per_request_and_reset() -> None:
    endpoint = "https://api.example.com/v1/chat/completions"
    agent = ChatCompletionsAgent(
        AgentConfig(
            agent_type="chat_completions",
            agent_name="cc",
            api_config={
                "endpoint": endpoint,
                "headers": {"Authorization": "Bearer test-token"},
                "body": {"model": "m", "messages": []},
            },
        )
    )
    agent._client = AsyncMock()
    agent._client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi"}}]},
            request=httpx.Request("POST", endpoint),
        )
    )

    token = set_request_headers(HEADERS)
    try:
        await agent.execute("hello")
    finally:
        reset_request_headers(token)
    await agent.execute("again")

    first, second = (c.kwargs["headers"] for c in agent._client.post.await_args_list)
    assert first == {"Authorization": "Bearer test-token", **HEADERS}
    assert second == {"Authorization": "Bearer test-token"}
    assert agent.chat_headers == {"Authorization": "Bearer test-token"}


# ── Simulator turn hooks ──


class _Hooks:
    """Recording TurnHooks implementation."""

    def __init__(
        self,
        headers: Mapping[str, str] | None = HEADERS,
        fail_before: Exception | None = None,
        fail_after: Exception | None = None,
    ) -> None:
        self.headers = headers
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.calls: list[tuple[Any, ...]] = []

    async def before_agent_execute(self, turn_id: int) -> Mapping[str, str] | None:
        self.calls.append(("before", turn_id))
        if self.fail_before is not None:
            raise self.fail_before
        return self.headers

    async def after_agent_execute(
        self, turn_id: int, error: BaseException | None
    ) -> None:
        # Headers must already be reset when the after-hook runs.
        self.calls.append(("after", turn_id, error, request_headers.get()))
        if self.fail_after is not None:
            raise self.fail_after


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="s1",
        user_id="u1",
        goal="test goal",
        agent_context="context",
        knowledge=[{"content": "k1"}],
        user_profile="test user",
    )


def _make_simulator() -> Simulator:
    simulator_params = MagicMock()
    simulator_params.simulated_user_prompt_template = None
    simulator_params.web_search = False
    return Simulator(
        agent_config=MagicMock(),
        simulator_params=simulator_params,
        llm=MagicMock(),
        trace_receiver=None,
    )


def _mock_agent(execute: AsyncMock) -> AsyncMock:
    agent = AsyncMock()
    agent.get_chat_id = AsyncMock(return_value="conv-1")
    agent.execute = execute
    agent.close = AsyncMock()
    return agent


def _patched(agent: AsyncMock) -> Any:  # noqa: ANN401
    """Patch the simulator's collaborators; the judge never terminates."""
    return (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch(
            "arksim.simulation_engine.simulator.judge_turn",
            AsyncMock(return_value=None),
        ),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            AsyncMock(return_value="hello"),
        ),
    )


async def test_hooks_run_in_order_and_headers_scoped_to_request() -> None:
    sim = _make_simulator()
    seen: list[Mapping[str, str] | None] = []

    async def _execute(user_query: str, **kwargs: object) -> str:
        seen.append(request_headers.get())
        return "reply"

    agent = _mock_agent(AsyncMock(side_effect=_execute))
    hooks = _Hooks()
    a, b, c = _patched(agent)
    with a, b, c:
        state = await sim.run_single_conversation(
            _scenario(), max_turns=2, turn_hooks=hooks
        )

    assert hooks.calls == [
        ("before", 0),
        ("after", 0, None, None),
        ("before", 1),
        ("after", 1, None, None),
    ]
    assert seen == [HEADERS, HEADERS]
    assert request_headers.get() is None

    # Hook turn ids equal the assistant Message.turn_id in the output.
    convo = sim.to_conversation_output(state)
    assistant_turns = [
        m.turn_id for m in convo.conversation_history if m.role == "assistant"
    ]
    assert assistant_turns == [0, 1]


async def test_after_hook_receives_execute_error() -> None:
    sim = _make_simulator()
    boom = RuntimeError("HTTP 500")
    agent = _mock_agent(AsyncMock(side_effect=boom))
    hooks = _Hooks()
    a, b, c = _patched(agent)
    with a, b, c, pytest.raises(ConversationExecutionError, match="HTTP 500"):
        await sim.run_single_conversation(_scenario(), max_turns=2, turn_hooks=hooks)

    assert hooks.calls == [("before", 0), ("after", 0, boom, None)]
    assert request_headers.get() is None


async def test_after_hook_failure_does_not_mask_execute_error() -> None:
    sim = _make_simulator()
    agent = _mock_agent(AsyncMock(side_effect=RuntimeError("original")))
    hooks = _Hooks(fail_after=ValueError("hook broke"))
    a, b, c = _patched(agent)
    with a, b, c, pytest.raises(ConversationExecutionError, match="original"):
        await sim.run_single_conversation(_scenario(), max_turns=1, turn_hooks=hooks)


async def test_after_hook_failure_on_success_is_logged_and_ignored() -> None:
    sim = _make_simulator()
    agent = _mock_agent(AsyncMock(return_value="reply"))
    hooks = _Hooks(fail_after=ValueError("hook broke"))
    a, b, c = _patched(agent)
    with a, b, c:
        state = await sim.run_single_conversation(
            _scenario(), max_turns=1, turn_hooks=hooks
        )

    assert state is not None
    assert len(hooks.calls) == 2


async def test_before_hook_failure_aborts_before_request() -> None:
    sim = _make_simulator()
    agent = _mock_agent(AsyncMock(return_value="reply"))
    hooks = _Hooks(fail_before=RuntimeError("registration failed"))
    a, b, c = _patched(agent)
    with (
        a,
        b,
        c,
        pytest.raises(ConversationExecutionError, match="registration failed"),
    ):
        await sim.run_single_conversation(_scenario(), max_turns=2, turn_hooks=hooks)

    agent.execute.assert_not_awaited()
    assert hooks.calls == [("before", 0)]
    assert request_headers.get() is None
    agent.close.assert_awaited_once()


async def test_chat_id_override_used_for_conversation_and_metadata() -> None:
    sim = _make_simulator()
    agent = _mock_agent(AsyncMock(return_value="reply"))
    a, b, c = _patched(agent)
    with a, b, c:
        state = await sim.run_single_conversation(
            _scenario(), max_turns=1, chat_id="platform-chat-42"
        )

    assert state is not None
    assert state.conversation_id == "platform-chat-42"
    assert agent.execute.await_args.kwargs["metadata"]["chat_id"] == "platform-chat-42"


async def test_no_hooks_leaves_headers_unset() -> None:
    sim = _make_simulator()
    seen: list[Mapping[str, str] | None] = []

    async def _execute(user_query: str, **kwargs: object) -> str:
        seen.append(request_headers.get())
        return "reply"

    agent = _mock_agent(AsyncMock(side_effect=_execute))
    a, b, c = _patched(agent)
    with a, b, c:
        state = await sim.run_single_conversation(_scenario(), max_turns=1)

    assert state is not None
    assert state.conversation_id == "conv-1"
    assert seen == [None]
