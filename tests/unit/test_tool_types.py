# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from arksim.simulation_engine.tool_types import ToolCallSource


class TestToolCallSource:
    def test_chat_completions_variant_exists(self) -> None:
        assert ToolCallSource.CHAT_COMPLETIONS.value == "chat_completions"

    def test_chat_completions_serializes_to_string(self) -> None:
        """Enum inherits from str so pydantic/json serialize it as plain string."""
        assert str(ToolCallSource.CHAT_COMPLETIONS.value) == "chat_completions"

    def test_existing_variants_unchanged(self) -> None:
        assert ToolCallSource.A2A_PROTOCOL.value == "a2a_protocol"
        assert ToolCallSource.OPENAI_AGENTS.value == "openai_agents"
        assert ToolCallSource.OTEL_TRACE.value == "otel_trace"
