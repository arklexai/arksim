# SPDX-License-Identifier: Apache-2.0
"""Tests for evaluator prompt templates and prompt-related models."""

from __future__ import annotations

from arksim.evaluator.base_metric import QualResult
from arksim.evaluator.utils.prompts import (
    agent_behavior_failure_system_prompt,
    agent_behavior_failure_user_prompt,
    goal_completion_user_prompt,
    tool_call_behavior_failure_user_prompt,
)


class TestQualResult:
    def test_accepts_metadata(self) -> None:
        result = QualResult(
            name="agent_behavior_failure",
            value="false information",
            reason="unsupported claim",
            metadata={"failure_type": "hallucination", "severity": "critical"},
        )

        assert result.metadata == {
            "failure_type": "hallucination",
            "severity": "critical",
        }

    def test_metadata_round_trips_through_model_dump_and_validate(self) -> None:
        result = QualResult(
            name="agent_behavior_failure",
            value="false information",
            reason="unsupported claim",
            metadata={"failure_type": "hallucination", "severity": "critical"},
        )

        dumped = result.model_dump()
        restored = QualResult.model_validate(dumped)

        assert restored == result
        assert restored.metadata == result.metadata


class TestPromptTemplates:
    def test_goal_completion_prompt_uses_user_goal_placeholder(self) -> None:
        assert "{user_goal}" in goal_completion_user_prompt
        assert "{goal}" not in goal_completion_user_prompt

    def test_behavior_failure_user_prompts_use_chat_history_placeholder(self) -> None:
        assert "{chat_history}" in agent_behavior_failure_user_prompt
        assert "{chat_history}" in tool_call_behavior_failure_user_prompt

    def test_false_information_definition_mentions_unsupported_knowledge(self) -> None:
        assert (
            "not supported by the provided knowledge or context"
            in agent_behavior_failure_system_prompt
        )

    def test_lack_of_specific_information_no_longer_requires_knowledge(self) -> None:
        assert "only applies when knowledge is provided" not in (
            agent_behavior_failure_system_prompt
        )
