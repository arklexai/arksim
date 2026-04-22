# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from arksim.simulation_engine.tool_types import ToolCallSource


class TestToolCallSource:
    def test_chat_completions_variant_exists(self) -> None:
        assert ToolCallSource.CHAT_COMPLETIONS.value == "chat_completions"

    def test_chat_completions_serializes_via_pydantic(self) -> None:
        """Pydantic serializes the enum to a plain string in model_dump()."""
        from arksim.simulation_engine.tool_types import ToolCall

        tc = ToolCall(
            id="c1",
            name="f",
            source=ToolCallSource.CHAT_COMPLETIONS,
        )
        dumped = tc.model_dump()
        assert dumped["source"] == "chat_completions"
        assert isinstance(dumped["source"], str)

    def test_existing_variants_unchanged(self) -> None:
        assert ToolCallSource.A2A_PROTOCOL.value == "a2a_protocol"
        assert ToolCallSource.OPENAI_AGENTS.value == "openai_agents"
        assert ToolCallSource.OTEL_TRACE.value == "otel_trace"
