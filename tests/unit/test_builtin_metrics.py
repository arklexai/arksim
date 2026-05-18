# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.builtin_metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arksim.evaluator.base_metric import ChatMessage, ScoreInput
from arksim.evaluator.builtin_metrics import (
    AgentBehaviorFailureMetric,
    CoherenceMetric,
    FaithfulnessMetric,
    GoalCompletionMetric,
    HelpfulnessMetric,
    RelevanceMetric,
    VerbosityMetric,
)
from arksim.evaluator.utils.schema import QualSchema, ScoreSchema


def _mock_llm(score: int = 4, reason: str = "ok") -> MagicMock:
    """Return a mock LLM that returns a ScoreSchema on .call()."""
    llm = MagicMock()
    llm.call.return_value = ScoreSchema(score=score, reason=reason)
    return llm


def _mock_llm_qual(label: str = "no failure", reason: str = "fine") -> MagicMock:
    """Return a mock LLM that returns a QualSchema on .call()."""
    llm = MagicMock()
    llm.call.return_value = QualSchema(label=label, reason=reason)
    return llm


def _score_input(**overrides: object) -> ScoreInput:
    defaults: dict[str, object] = {
        "chat_history": [ChatMessage(role="user", content="hi")],
        "current_turn": [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ],
        "knowledge": "some knowledge",
        "user_goal": "help user",
    }
    defaults.update(overrides)
    return ScoreInput(**defaults)


class TestHelpfulnessMetric:
    def test_name(self) -> None:
        m = HelpfulnessMetric(_mock_llm())
        assert m.name == "helpfulness"

    def test_score_returns_llm_value(self) -> None:
        llm = _mock_llm(score=3, reason="decent")
        m = HelpfulnessMetric(llm)
        result = m.score(_score_input())
        assert result.value == 3
        assert result.reason == "decent"
        assert result.name == "helpfulness"
        llm.call.assert_called_once()


@pytest.mark.parametrize(
    ("factory", "method_name", "expected_fragments"),
    [
        (
            HelpfulnessMetric,
            "score",
            [
                "Here is the conversation between user and AI assistant:",
                "user: Need help\nassistant: Happy to help\n",
                "Evaluation Form:",
            ],
        ),
        (
            CoherenceMetric,
            "score",
            [
                "Here is the conversation between user and AI assistant:",
                "user: Need help\nassistant: Happy to help\n",
                "Evaluation Form:",
            ],
        ),
        (
            VerbosityMetric,
            "score",
            [
                "Here is the conversation between user and AI assistant:",
                "user: Need help\nassistant: Happy to help\n",
                "Evaluation Form:",
            ],
        ),
        (
            RelevanceMetric,
            "score",
            [
                "Here is the conversation between user and AI assistant:",
                "user: Need help\nassistant: Happy to help\n",
                "Evaluation Form:",
            ],
        ),
        (
            FaithfulnessMetric,
            "score",
            [
                "policy knowledge",
                "user: Need help\nassistant: Happy to help\n",
                "Evaluation Form:",
            ],
        ),
        (
            GoalCompletionMetric,
            "score",
            [
                "Here is user's goal: resolve billing issue",
                "user: Need help\nassistant: Happy to help\n",
                "Evaluation Form:",
            ],
        ),
        (
            AgentBehaviorFailureMetric,
            "evaluate",
            [
                "resolve billing issue",
                "policy knowledge",
                "user: Need help\nassistant: Happy to help\n",
            ],
        ),
    ],
)
def test_builtin_metrics_format_generic_score_input_context(
    factory: type[
        HelpfulnessMetric
        | CoherenceMetric
        | VerbosityMetric
        | RelevanceMetric
        | FaithfulnessMetric
        | GoalCompletionMetric
        | AgentBehaviorFailureMetric
    ],
    method_name: str,
    expected_fragments: list[str],
) -> None:
    llm = _mock_llm() if method_name == "score" else _mock_llm_qual()
    score_input = _score_input(
        chat_history=[
            ChatMessage(role="user", content="Need help"),
            ChatMessage(role="assistant", content="Happy to help"),
        ],
        current_turn=[
            ChatMessage(role="user", content="Current question"),
            ChatMessage(role="assistant", content="Current answer"),
        ],
        knowledge="policy knowledge",
        user_goal="resolve billing issue",
        profile="frustrated premium customer",
    )

    getattr(factory(llm), method_name)(score_input)

    user_msg = llm.call.call_args.args[0][1]["content"]
    for fragment in expected_fragments:
        assert fragment in user_msg


class TestCoherenceMetric:
    def test_name(self) -> None:
        assert CoherenceMetric(_mock_llm()).name == "coherence"

    def test_score(self) -> None:
        llm = _mock_llm(score=5)
        result = CoherenceMetric(llm).score(_score_input())
        assert result.value == 5


class TestVerbosityMetric:
    def test_name(self) -> None:
        assert VerbosityMetric(_mock_llm()).name == "verbosity"

    def test_score_inverts(self) -> None:
        """LLM score of 1 (verbose) should be stored as 5 (good)."""
        llm = _mock_llm(score=1)
        result = VerbosityMetric(llm).score(_score_input())
        assert result.value == 5

    def test_score_inverts_high(self) -> None:
        """LLM score of 5 (concise) should be stored as 1."""
        llm = _mock_llm(score=5)
        result = VerbosityMetric(llm).score(_score_input())
        assert result.value == 1

    def test_score_midpoint(self) -> None:
        """LLM score of 3 stays at 3 (6-3=3)."""
        llm = _mock_llm(score=3)
        result = VerbosityMetric(llm).score(_score_input())
        assert result.value == 3


class TestRelevanceMetric:
    def test_name(self) -> None:
        assert RelevanceMetric(_mock_llm()).name == "relevance"

    def test_score(self) -> None:
        result = RelevanceMetric(_mock_llm(score=2)).score(_score_input())
        assert result.value == 2


class TestFaithfulnessMetric:
    def test_name(self) -> None:
        assert FaithfulnessMetric(_mock_llm()).name == "faithfulness"

    def test_score_includes_knowledge(self) -> None:
        llm = _mock_llm(score=4)
        FaithfulnessMetric(llm).score(_score_input(knowledge="important fact"))
        call_args = llm.call.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "important fact" in user_msg

    def test_score_includes_formatted_chat_history(self) -> None:
        llm = _mock_llm(score=4)
        FaithfulnessMetric(llm).score(
            _score_input(
                knowledge="policy knowledge",
                chat_history=[
                    ChatMessage(role="user", content="What is covered?"),
                    ChatMessage(role="assistant", content="Water damage is covered."),
                ],
            )
        )
        user_msg = llm.call.call_args.args[0][1]["content"]
        assert "policy knowledge" in user_msg
        assert (
            "user: What is covered?\nassistant: Water damage is covered.\n" in user_msg
        )


class TestGoalCompletionMetric:
    def test_name(self) -> None:
        assert GoalCompletionMetric(_mock_llm()).name == "goal_completion"

    def test_score_uses_chat_history_and_goal(self) -> None:
        llm = _mock_llm(score=5)
        si = _score_input(user_goal="book a flight")
        GoalCompletionMetric(llm).score(si)
        call_args = llm.call.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "book a flight" in user_msg
        assert "user: hi\n" in user_msg


class TestAgentBehaviorFailureMetric:
    def test_name(self) -> None:
        m = AgentBehaviorFailureMetric(_mock_llm_qual())
        assert m.name == "agent_behavior_failure"

    def test_evaluate_returns_label(self) -> None:
        llm = _mock_llm_qual(label="repetition", reason="said same thing")
        result = AgentBehaviorFailureMetric(llm).evaluate(_score_input())
        assert result.value == "repetition"
        assert result.reason == "said same thing"

    def test_evaluate_includes_goal_knowledge_and_chat_history(self) -> None:
        llm = _mock_llm_qual(label="no failure", reason="fine")
        AgentBehaviorFailureMetric(llm).evaluate(
            _score_input(
                user_goal="update my address",
                knowledge="Address changes require account verification.",
                chat_history=[
                    ChatMessage(role="user", content="Please update my address"),
                    ChatMessage(
                        role="assistant",
                        content="I can help with that after verification.",
                    ),
                ],
            )
        )

        user_msg = llm.call.call_args.args[0][1]["content"]
        assert "update my address" in user_msg
        assert "Address changes require account verification." in user_msg
        assert (
            "user: Please update my address\n"
            "assistant: I can help with that after verification.\n"
        ) in user_msg
