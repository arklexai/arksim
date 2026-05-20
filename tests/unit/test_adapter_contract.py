# SPDX-License-Identifier: Apache-2.0
"""Cross-adapter contract: events in produce ToolCall out for every SDK adapter.

This is the capstone test for the 8-adapter tracing batch. Each adapter has its
own quirks (split-event correlation, async vs sync, event-bus dispatch, etc.),
but the contract is uniform: fire one synthetic tool-use event into the
adapter while routing context is set, and the configured receiver receives a
single ``ToolCall`` tagged with the correct ``ToolCallSource``.

One factory per adapter. Each factory returns ``(fire_callable,
expected_source)``. The parameterized test sets routing context, invokes
``fire()``, and asserts a single ``ToolCall`` lands on the receiver with the
expected source.

If you add a new adapter, add a factory below and one ``pytest.param`` entry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _clear_trace_context, _set_trace_context

Factory = Callable[[], tuple[Callable[[], None], ToolCallSource]]


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    _clear_trace_context()
    yield
    _clear_trace_context()


# --- Per-adapter factories ---------------------------------------------------


def _make_langchain() -> tuple[Callable[[], None], ToolCallSource]:
    from arksim.tracing.integrations.langchain import ArksimLangChainHandler

    handler = ArksimLangChainHandler()
    run_id = uuid4()

    def fire() -> None:
        handler.on_tool_start({"name": "get_weather"}, '{"city": "NYC"}', run_id=run_id)
        handler.on_tool_end("sunny 75F", run_id=run_id)

    return fire, ToolCallSource.LANGCHAIN


def _make_crewai() -> tuple[Callable[[], None], ToolCallSource]:
    from crewai.events import ToolUsageFinishedEvent, crewai_event_bus

    from arksim.tracing.integrations.crewai import ArksimCrewEventListener

    # ``crewai_event_bus`` is a module-level singleton; enter a scoped-handlers
    # block inside the factory so this listener does not leak to other tests,
    # then exit it after the dispatched future completes.
    scope = crewai_event_bus.scoped_handlers()
    scope.__enter__()
    ArksimCrewEventListener()
    now = datetime.now(timezone.utc)
    event = ToolUsageFinishedEvent(
        timestamp=now,
        tool_name="get_weather",
        tool_args={"city": "NYC"},
        started_at=now,
        finished_at=now,
        output="sunny 75F",
    )

    def fire() -> None:
        try:
            future = crewai_event_bus.emit(source=object(), event=event)
            if future is not None:
                # Dispatcher uses a ThreadPoolExecutor; wait to avoid racing
                # the assertion against the worker thread.
                future.result(timeout=5.0)
        finally:
            scope.__exit__(None, None, None)

    return fire, ToolCallSource.CREWAI


def _make_claude_agent_sdk() -> tuple[Callable[[], None], ToolCallSource]:
    import asyncio

    from arksim.tracing.integrations.claude_agent_sdk import ArksimClaudeHooks

    hooks = ArksimClaudeHooks()

    def fire() -> None:
        asyncio.run(
            hooks.post_tool_use(
                {
                    "tool_name": "lookup_order",
                    "tool_input": {"order_id": "12345"},
                    "tool_response": "shipped",
                },
                "t1",
                cast("Any", {}),
            )
        )

    return fire, ToolCallSource.CLAUDE_AGENT_SDK


def _make_google_adk() -> tuple[Callable[[], None], ToolCallSource]:
    import asyncio

    from arksim.tracing.integrations.google_adk import ArksimADKPlugin

    plugin = ArksimADKPlugin()
    tool = MagicMock()
    tool.name = "lookup_order"
    tool_context = MagicMock(spec=["invocation_id"])
    tool_context.invocation_id = "inv-1"

    def fire() -> None:
        asyncio.run(
            plugin.after_tool_callback(
                tool=tool,
                tool_args={"order_id": "12345"},
                tool_context=tool_context,
                result={"status": "shipped"},
            )
        )

    return fire, ToolCallSource.GOOGLE_ADK


def _make_livekit() -> tuple[Callable[[], None], ToolCallSource]:
    from livekit.agents.llm.chat_context import FunctionCall, FunctionCallOutput
    from livekit.agents.voice.events import FunctionToolsExecutedEvent

    from arksim.tracing.integrations.livekit import ArksimLiveKitHandler

    handler = ArksimLiveKitHandler()
    event = FunctionToolsExecutedEvent(
        function_calls=[
            FunctionCall(
                call_id="call_1", arguments='{"city": "NYC"}', name="get_weather"
            )
        ],
        function_call_outputs=[
            FunctionCallOutput(
                call_id="call_1",
                name="get_weather",
                output="sunny 75F",
                is_error=False,
            )
        ],
    )

    def fire() -> None:
        handler.on_function_tools_executed(event)

    return fire, ToolCallSource.LIVEKIT


def _make_strands() -> tuple[Callable[[], None], ToolCallSource]:
    from strands.hooks import AfterToolCallEvent, HookRegistry
    from strands.types.tools import ToolResult, ToolUse

    from arksim.tracing.integrations.strands import ArksimStrandsHookProvider

    registry = HookRegistry()
    ArksimStrandsHookProvider().register_hooks(registry)
    tool_use = cast(
        "ToolUse",
        {"name": "get_weather", "toolUseId": "u1", "input": {"city": "NYC"}},
    )
    result = cast(
        "ToolResult",
        {
            "toolUseId": "u1",
            "status": "success",
            "content": [{"text": "sunny 75F"}],
        },
    )
    event = AfterToolCallEvent(
        agent=cast("Any", MagicMock()),
        selected_tool=None,
        tool_use=tool_use,
        invocation_state={},
        result=result,
        exception=None,
    )

    def fire() -> None:
        registry.invoke_callbacks(event)

    return fire, ToolCallSource.STRANDS


def _make_llamaindex() -> tuple[Callable[[], None], ToolCallSource]:
    from llama_index.core.agent.workflow import ToolCall as LIToolCall
    from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
    from llama_index.core.tools.types import ToolOutput

    from arksim.tracing.integrations.llamaindex import ArksimLlamaIndexObserver

    observer = ArksimLlamaIndexObserver()
    args = {"city": "NYC"}
    start = LIToolCall(tool_name="get_weather", tool_kwargs=args, tool_id="t1")
    tool_output = ToolOutput(
        tool_name="get_weather",
        content="sunny 75F",
        raw_input=args,
        raw_output="sunny 75F",
        is_error=False,
        exception=None,
    )
    result = LIToolCallResult(
        tool_name="get_weather",
        tool_kwargs=args,
        tool_id="t1",
        tool_output=tool_output,
        return_direct=False,
    )

    def fire() -> None:
        observer.observe(start)
        observer.observe(result)

    return fire, ToolCallSource.LLAMAINDEX


def _make_smolagents() -> tuple[Callable[[], None], ToolCallSource]:
    from smolagents.memory import ActionStep
    from smolagents.memory import ToolCall as SmolToolCall
    from smolagents.monitoring import Timing

    from arksim.tracing.integrations.smolagents import ArksimSmolagentsCallback

    observer = ArksimSmolagentsCallback()
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        tool_calls=[
            SmolToolCall(name="get_weather", arguments={"city": "NYC"}, id="call_1")
        ],
        observations="sunny 75F",
    )

    def fire() -> None:
        observer(step)

    return fire, ToolCallSource.SMOLAGENTS


# --- The contract test ------------------------------------------------------


_FACTORIES: list[Any] = [
    pytest.param(_make_langchain, id="langchain"),
    pytest.param(_make_crewai, id="crewai"),
    pytest.param(_make_claude_agent_sdk, id="claude_agent_sdk"),
    pytest.param(_make_google_adk, id="google_adk"),
    pytest.param(_make_livekit, id="livekit"),
    pytest.param(_make_strands, id="strands"),
    pytest.param(_make_llamaindex, id="llamaindex"),
    pytest.param(_make_smolagents, id="smolagents"),
]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_adapter_produces_tool_call_with_correct_source(factory: Factory) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)

    fire, expected_source = factory()
    fire()

    receiver.submit_tool_calls.assert_called_once()
    args, _ = receiver.submit_tool_calls.call_args
    conv, turn, tool_calls = args
    assert conv == "conv-1"
    assert turn == 0
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].source == expected_source


@pytest.mark.parametrize("factory", _FACTORIES)
def test_adapter_no_routing_context_drops(factory: Factory) -> None:
    """With no trace context set, no adapter must call the receiver.

    Routing context is the only signal that arksim is observing; without
    it the adapter is being invoked outside a simulation and must stay
    silent. This is the cross-cutting drop-on-no-context contract.
    """
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.

    fire, _expected_source = factory()
    fire()

    receiver.submit_tool_calls.assert_not_called()


# --- Error-path factories ---------------------------------------------------


def _make_langchain_error() -> tuple[Callable[[], None], ToolCallSource]:
    from arksim.tracing.integrations.langchain import ArksimLangChainHandler

    handler = ArksimLangChainHandler()
    run_id = uuid4()

    def fire() -> None:
        handler.on_tool_start({"name": "get_weather"}, '{"city": "NYC"}', run_id=run_id)
        handler.on_tool_error(ValueError("nope"), run_id=run_id)

    return fire, ToolCallSource.LANGCHAIN


def _make_crewai_error() -> tuple[Callable[[], None], ToolCallSource]:
    from crewai.events import ToolUsageErrorEvent, crewai_event_bus

    from arksim.tracing.integrations.crewai import ArksimCrewEventListener

    scope = crewai_event_bus.scoped_handlers()
    scope.__enter__()
    ArksimCrewEventListener()
    now = datetime.now(timezone.utc)
    event = ToolUsageErrorEvent(
        timestamp=now,
        tool_name="get_weather",
        tool_args={"city": "NYC"},
        error=ValueError("nope"),
    )

    def fire() -> None:
        try:
            future = crewai_event_bus.emit(source=object(), event=event)
            if future is not None:
                future.result(timeout=5.0)
        finally:
            scope.__exit__(None, None, None)

    return fire, ToolCallSource.CREWAI


def _make_livekit_error() -> tuple[Callable[[], None], ToolCallSource]:
    from livekit.agents.llm.chat_context import FunctionCall, FunctionCallOutput
    from livekit.agents.voice.events import FunctionToolsExecutedEvent

    from arksim.tracing.integrations.livekit import ArksimLiveKitHandler

    handler = ArksimLiveKitHandler()
    event = FunctionToolsExecutedEvent(
        function_calls=[
            FunctionCall(
                call_id="call_1", arguments='{"city": "NYC"}', name="get_weather"
            )
        ],
        function_call_outputs=[
            FunctionCallOutput(
                call_id="call_1",
                name="get_weather",
                output="ValueError: nope",
                is_error=True,
            )
        ],
    )

    def fire() -> None:
        handler.on_function_tools_executed(event)

    return fire, ToolCallSource.LIVEKIT


def _make_strands_error() -> tuple[Callable[[], None], ToolCallSource]:
    from strands.hooks import AfterToolCallEvent, HookRegistry
    from strands.types.tools import ToolResult, ToolUse

    from arksim.tracing.integrations.strands import ArksimStrandsHookProvider

    registry = HookRegistry()
    ArksimStrandsHookProvider().register_hooks(registry)
    tool_use = cast(
        "ToolUse",
        {"name": "get_weather", "toolUseId": "u1", "input": {"city": "NYC"}},
    )
    result = cast(
        "ToolResult",
        {
            "toolUseId": "u1",
            "status": "error",
            "content": [{"text": "boom"}],
        },
    )
    event = AfterToolCallEvent(
        agent=cast("Any", MagicMock()),
        selected_tool=None,
        tool_use=tool_use,
        invocation_state={},
        result=result,
        exception=ValueError("nope"),
    )

    def fire() -> None:
        registry.invoke_callbacks(event)

    return fire, ToolCallSource.STRANDS


def _make_llamaindex_error() -> tuple[Callable[[], None], ToolCallSource]:
    from llama_index.core.agent.workflow import ToolCall as LIToolCall
    from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
    from llama_index.core.tools.types import ToolOutput

    from arksim.tracing.integrations.llamaindex import ArksimLlamaIndexObserver

    observer = ArksimLlamaIndexObserver()
    args = {"city": "NYC"}
    start = LIToolCall(tool_name="get_weather", tool_kwargs=args, tool_id="t1")
    tool_output = ToolOutput(
        tool_name="get_weather",
        content="",
        raw_input=args,
        raw_output=None,
        is_error=True,
        exception=ValueError("nope"),
    )
    result = LIToolCallResult(
        tool_name="get_weather",
        tool_kwargs=args,
        tool_id="t1",
        tool_output=tool_output,
        return_direct=False,
    )

    def fire() -> None:
        observer.observe(start)
        observer.observe(result)

    return fire, ToolCallSource.LLAMAINDEX


_ERROR_FACTORIES: list[Any] = [
    pytest.param(_make_langchain_error, id="langchain"),
    pytest.param(_make_crewai_error, id="crewai"),
    pytest.param(_make_livekit_error, id="livekit"),
    pytest.param(_make_strands_error, id="strands"),
    pytest.param(_make_llamaindex_error, id="llamaindex"),
    pytest.param(
        None,
        id="claude_agent_sdk",
        marks=pytest.mark.skip(reason="SDK has no separate error event"),
    ),
    pytest.param(
        None,
        id="google_adk",
        marks=pytest.mark.skip(reason="SDK has no separate error event"),
    ),
    pytest.param(
        None,
        id="smolagents",
        marks=pytest.mark.skip(reason="SDK has no separate error event"),
    ),
]


@pytest.mark.parametrize("factory", _ERROR_FACTORIES)
def test_adapter_error_path_populates_error_field(factory: Factory) -> None:
    """Adapters that expose a distinct error event must surface it on
    ToolCall.error rather than ToolCall.result. Skipped for SDKs whose
    callbacks carry only the success/result shape (Claude Agent SDK,
    Google ADK, Smolagents).
    """
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)

    fire, expected_source = factory()
    fire()

    receiver.submit_tool_calls.assert_called_once()
    args, _ = receiver.submit_tool_calls.call_args
    _, _, tool_calls = args
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.source == expected_source
    assert tc.error is not None and tc.error != ""
    assert tc.result is None
