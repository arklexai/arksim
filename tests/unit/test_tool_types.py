# SPDX-License-Identifier: Apache-2.0
"""Tests for tool_types module."""

from __future__ import annotations

from arksim.simulation_engine.tool_types import ToolCallSource


def test_tool_call_source_has_new_sdk_adapter_variants() -> None:
    """8 new variants for SDK tracing adapters added in 2026-05 rollout."""
    assert ToolCallSource.LANGCHAIN.value == "langchain"
    assert ToolCallSource.CREWAI.value == "crewai"
    assert ToolCallSource.CLAUDE_AGENT_SDK.value == "claude_agent_sdk"
    assert ToolCallSource.GOOGLE_ADK.value == "google_adk"
    assert ToolCallSource.LIVEKIT.value == "livekit"
    assert ToolCallSource.STRANDS.value == "strands"
    assert ToolCallSource.LLAMAINDEX.value == "llamaindex"
    assert ToolCallSource.SMOLAGENTS.value == "smolagents"
