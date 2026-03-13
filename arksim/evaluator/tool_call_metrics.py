# SPDX-License-Identifier: Apache-2.0
"""Tool call behavior failure evaluation metric."""

from __future__ import annotations

import json

from arksim.llms.chat import LLM

from .base_metric import (
    QualitativeMetric,
    QualResult,
    ScoreInput,
    format_chat_history,
)
from .utils.enums import AgentMetrics
from .utils.prompts import (
    tool_call_behavior_failure_system_prompt,
    tool_call_behavior_failure_user_prompt,
)
from .utils.schema import QualSchema


class ToolCallBehaviorFailureMetric(QualitativeMetric):
    """Detects tool call behavior failures using LLM-as-judge.

    Evaluates: tool selection, parameter correctness, call necessity,
    and result usage. Maps failures to AgentBehaviorFailureType labels
    including unsafe action and unsafe state for tool-specific failure modes.
    """

    DESCRIPTION = (
        "Detects tool call failures: wrong tool selection, incorrect parameters, "
        "unnecessary calls, ignored results, unsafe actions, and unsafe state. "
        "Labels: disobey user request, lack of specific information, "
        "failure to ask for clarification, false information, repetition, "
        "unsafe action, unsafe state, no failure."
    )

    def __init__(self, llm: LLM) -> None:
        super().__init__(
            name=AgentMetrics.TOOL_CALL_BEHAVIOR_FAILURE.value,
            description=self.DESCRIPTION,
        )
        self._llm = llm

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        tool_calls = getattr(score_input, "tool_calls", None)
        if not tool_calls:
            return QualResult(
                name=self.name,
                value="no failure",
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
                        conversation=format_chat_history(score_input.current_turn),
                        knowledge=score_input.knowledge,
                        tool_calls=json.dumps(tool_calls, indent=2),
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
