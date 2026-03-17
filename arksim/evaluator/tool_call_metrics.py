# SPDX-License-Identifier: Apache-2.0
"""Tool call behavior failure evaluation metric."""

from __future__ import annotations

import json

from .base_metric import (
    QualResult,
    ScoreInput,
    format_chat_history,
)
from .builtin_metrics import AgentBehaviorFailureMetric
from .utils.enums import AgentBehaviorFailureType, AgentMetrics
from .utils.prompts import (
    tool_call_behavior_failure_system_prompt,
    tool_call_behavior_failure_user_prompt,
)
from .utils.schema import QualSchema


class ToolCallBehaviorFailureMetric(AgentBehaviorFailureMetric):
    """Detects tool call behavior failures using LLM-as-judge.

    Extends AgentBehaviorFailureMetric because tool call failures are
    a specialized form of agent behavior failure. Shares the same
    failure labels, severity mapping, and pipeline (TSR, Unique Errors).

    Evaluates: tool selection, parameter correctness, call necessity,
    result usage, action safety, and response integrity.
    """

    DESCRIPTION = (
        "Detects tool call failures: wrong tool selection, incorrect parameters, "
        "unnecessary calls, ignored results, unsafe actions, and unsafe state. "
        "Labels: disobey user request, lack of specific information, "
        "failure to ask for clarification, false information, repetition, "
        "unsafe action, unsafe state, no failure."
    )

    def __init__(self, llm: LLM) -> None:  # noqa: F821
        super().__init__(llm)
        self.name = AgentMetrics.TOOL_CALL_BEHAVIOR_FAILURE.value
        self.description = self.DESCRIPTION

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        tool_calls = getattr(score_input, "tool_calls", None)
        if not tool_calls:
            return QualResult(
                name=self.name,
                value=AgentBehaviorFailureType.NO_FAILURE.value,
                reason="No tool calls in this turn",
            )

        response = self._llm.call(
            [
                {
                    "role": "system",
                    "content": tool_call_behavior_failure_system_prompt,
                },
                {
                    "role": "user",
                    "content": tool_call_behavior_failure_user_prompt.format(
                        user_goal=score_input.user_goal,
                        conversation=format_chat_history(score_input.current_turn),
                        knowledge=score_input.knowledge,
                        tool_calls=json.dumps(
                            [
                                tc.model_dump() if hasattr(tc, "model_dump") else tc
                                for tc in tool_calls
                            ],
                            indent=2,
                        ),
                    ),
                },
            ],
            schema=QualSchema,
        )
        return QualResult(
            name=self.name,
            value=response.label,
            reason=response.reason,
        )
