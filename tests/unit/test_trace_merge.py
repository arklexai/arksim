# SPDX-License-Identifier: Apache-2.0
"""Tests for trace receiver merge logic in the simulator turn loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arksim.simulation_engine.simulator import Simulator
from arksim.simulation_engine.tool_types import AgentResponse, ToolCall


def _make_simulator(
    trace_receiver: AsyncMock | None = None,
) -> Simulator:
    """Create a Simulator with mocked dependencies."""

    agent_config = MagicMock()
    simulator_params = MagicMock()
    simulator_params.simulated_user_prompt_template = None
    llm = MagicMock()

    return Simulator(
        agent_config=agent_config,
        simulator_params=simulator_params,
        llm=llm,
        trace_receiver=trace_receiver,
    )


# ── Merge logic ──


@pytest.mark.asyncio
async def test_trace_merge_appends_new_tool_calls() -> None:
    """Traced tool calls not in AgentResponse are appended."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(id="traced-1", name="search", arguments={"q": "test"}),
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    # Mock dependencies for _run_single_conversation
    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(id="explicit-1", name="lookup", arguments={"id": "42"}),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    # LLM returns non-STOP first so the agent gets a turn, then STOP
    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    # Explicit (lookup) + traced (search) = 2 tool calls
    assert len(tc_list) == 2
    names = {tc["name"] for tc in tc_list}
    assert names == {"lookup", "search"}


@pytest.mark.asyncio
async def test_trace_merge_deduplicates_by_id() -> None:
    """Traced tool calls with the same ID as explicit ones are skipped."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(id="shared-id", name="search", arguments={"q": "test"}),
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    # Agent returns the same tool call ID
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(id="shared-id", name="search", arguments={"q": "test"}),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    # LLM returns non-STOP first, then STOP
    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    # Find the assistant message (role="user" in internal history = agent)
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    # Should have only 1 tool call (deduped), not 2
    assert len(tc_list) == 1
    assert tc_list[0]["id"] == "shared-id"


@pytest.mark.asyncio
async def test_trace_merge_deduplicates_by_name_args() -> None:
    """Traced tool calls matching (name, arguments) of explicit ones are skipped.

    This covers the case where the trace receiver uses spanId as the tool call
    ID while AgentResponse carries an SDK-assigned ID.
    """
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            # Different ID, same (name, args)
            ToolCall(id="span-id-abc", name="search", arguments={"q": "test"}),
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(id="call_xyz", name="search", arguments={"q": "test"}),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    # Deduped by (name, args) despite different IDs
    assert len(tc_list) == 1
    assert tc_list[0]["id"] == "call_xyz"  # explicit one wins


@pytest.mark.asyncio
async def test_trace_merge_adds_unique_traced_calls() -> None:
    """Traced tool calls with different name/args are added alongside explicit ones."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(id="traced-1", name="fetch_data", arguments={"url": "/api"}),
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(id="explicit-1", name="search", arguments={"q": "test"}),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    assert len(tc_list) == 2
    names = {tc["name"] for tc in tc_list}
    assert names == {"search", "fetch_data"}


@pytest.mark.asyncio
async def test_no_trace_receiver_preserves_existing_behavior() -> None:
    """Without a trace receiver, tool calls come only from AgentResponse."""
    sim = _make_simulator(trace_receiver=None)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(id="tc-1", name="search", arguments={}),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    assert len(tc_list) == 1
    assert tc_list[0]["name"] == "search"


@pytest.mark.asyncio
async def test_turn_id_passed_in_metadata() -> None:
    """The turn_id is included in metadata passed to agent.execute()."""
    sim = _make_simulator(trace_receiver=None)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(return_value="response")
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    # agent.execute was called once (before STOP)
    assert mock_agent.execute.call_count == 1
    call_kwargs = mock_agent.execute.call_args
    metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
    assert metadata["turn_id"] == 0
    assert "chat_id" in metadata


@pytest.mark.asyncio
async def test_trace_merge_nested_dict_arguments() -> None:
    """Dedup handles nested dict arguments without crashing."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(
                id="traced-1",
                name="search",
                arguments={"filter": {"price": 100, "category": "laptop"}},
            ),
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(
                    id="explicit-1",
                    name="search",
                    arguments={"filter": {"price": 100, "category": "laptop"}},
                ),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    # Should have 1 tool call (deduped by name+args), not crash with TypeError
    assert len(tc_list) == 1


@pytest.mark.asyncio
async def test_trace_merge_three_way_overlap() -> None:
    """Agent returns A+B, traces return B'+C. Result should be A, B, C."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(id="span-b", name="lookup", arguments={"id": "42"}),
            ToolCall(id="c", name="cancel", arguments={"id": "99"}),
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="result",
            tool_calls=[
                ToolCall(id="a", name="search", arguments={"q": "test"}),
                ToolCall(id="b", name="lookup", arguments={"id": "42"}),
            ],
        )
    )
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="test goal",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    assert len(tc_list) == 3
    names = [tc["name"] for tc in tc_list]
    assert names == ["search", "lookup", "cancel"]
    ids = [tc["id"] for tc in tc_list]
    assert "b" in ids
    assert "span-b" not in ids


@pytest.mark.asyncio
async def test_dedup_prefers_toolcall_with_result() -> None:
    """When a traced ToolCall has a result and the explicit one does not, traced wins."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    # Traced tool call with result
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(
                id="span_1",
                name="search",
                arguments={"q": "test"},
                result='["item1"]',
                source="otel_trace",
            )
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    # Explicit tool call (from response parsing) with no result
    mock_agent.execute = AsyncMock(
        return_value=AgentResponse(
            content="results",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="search",
                    arguments={"q": "test"},
                    source="response_parse",
                )
            ],
        )
    )
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="find items",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    # The traced version with result should replace the explicit one
    assert len(tc_list) == 1
    assert tc_list[0]["result"] == '["item1"]'
    assert tc_list[0]["source"] == "otel_trace"


@pytest.mark.asyncio
async def test_source_field_preserved_through_merge() -> None:
    """Source field survives the merge/dedup process for purely traced tool calls."""
    receiver = AsyncMock()
    receiver.signal_turn_complete = MagicMock()
    receiver.wait_for_traces = AsyncMock(
        return_value=[
            ToolCall(id="t1", name="unique_tool", arguments={}, source="otel_trace")
        ]
    )

    sim = _make_simulator(trace_receiver=receiver)

    mock_agent = AsyncMock()
    mock_agent.get_chat_id = AsyncMock(return_value="conv-1")
    # Agent returns a plain string (no tool calls in response)
    mock_agent.execute = AsyncMock(return_value="text only")
    mock_agent.close = AsyncMock()

    sim.llm.call_async = AsyncMock(side_effect=["hello", "###STOP###"])

    with patch(
        "arksim.simulation_engine.simulator.create_agent",
        return_value=mock_agent,
    ):
        state = await sim._run_single_conversation(
            profile="test user",
            goal="do something",
            knowledge=[{"content": "k1"}],
            agent_context="context",
            max_turns=3,
            scenario_id="s1",
        )

    assert state is not None
    agent_msgs = [m for m in state.conversation_history if m.get("role") == "user"]
    assert len(agent_msgs) == 1
    tc_list = agent_msgs[0].get("tool_calls", [])
    assert len(tc_list) == 1
    assert tc_list[0]["source"] == "otel_trace"
