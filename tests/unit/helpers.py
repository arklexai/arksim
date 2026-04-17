# SPDX-License-Identifier: Apache-2.0
"""Shared mock-building helpers for unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock


def make_mock_convo(
    convo_id: str,
    metric_scores: dict[str, list[float]] | None = None,
    user_goal_completion_score: float = -1.0,
    overall_agent_score: float = 0.9,
    abf_labels: list[str] | None = None,
    qual_scores: dict[str, list[str]] | None = None,
) -> MagicMock:
    """Build a mock ConversationEvaluation suitable for threshold tests.

    Args:
        convo_id: Conversation identifier.
        metric_scores: Per-metric numeric turn scores, e.g. ``{"faithfulness": [4.0, 3.5]}``.
        user_goal_completion_score: Conversation-level goal completion score (0-1, or -1 if skipped).
        overall_agent_score: Conversation-level overall agent score (0-1).
        abf_labels: Per-turn ``agent_behavior_failure`` labels.
        qual_scores: Per-metric qualitative turn labels, e.g. ``{"prohibited": ["ok", "violated"]}``.
    """
    metric_scores = metric_scores or {}
    max_turns = max(
        max((len(v) for v in metric_scores.values()), default=0),
        len(abf_labels) if abf_labels else 0,
        *(len(v) for v in (qual_scores or {}).values()),
        0,
    )
    turns = []
    for i in range(max_turns):
        turn = MagicMock()
        turn.turn_id = i

        scores = []
        for name, values in metric_scores.items():
            if i < len(values):
                r = MagicMock()
                r.name = name
                r.value = values[i]
                scores.append(r)
        turn.scores = scores

        turn.turn_behavior_failure = (
            abf_labels[i] if abf_labels and i < len(abf_labels) else "no failure"
        )

        qs = []
        for name, labels in (qual_scores or {}).items():
            if i < len(labels):
                q = MagicMock()
                q.name = name
                q.value = labels[i]
                qs.append(q)
        turn.qual_scores = qs

        turns.append(turn)

    convo = MagicMock()
    convo.conversation_id = convo_id
    convo.turn_scores = turns
    convo.user_goal_completion_score = user_goal_completion_score
    convo.overall_agent_score = overall_agent_score
    return convo


def make_mock_evaluation(convos: list) -> MagicMock:
    """Wrap a list of mock conversations in a mock Evaluation."""
    ev = MagicMock()
    ev.conversations = convos
    return ev
