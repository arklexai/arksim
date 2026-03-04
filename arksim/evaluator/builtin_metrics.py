# SPDX-License-Identifier: Apache-2.0
"""Built-in evaluation metrics implemented as QuantitativeMetric subclasses."""

from __future__ import annotations

from arksim.llms.chat import LLM

from .base_metric import (
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
    ScoreInput,
    format_chat_history,
)
from .utils.enums import AgentMetrics
from .utils.prompts import (
    agent_behavior_failure_system_prompt,
    agent_behavior_failure_user_prompt,
    coherence_system_prompt,
    coherence_user_prompt,
    faithfulness_system_prompt,
    faithfulness_user_prompt,
    goal_completion_system_prompt,
    goal_completion_user_prompt,
    helpfulness_system_prompt,
    helpfulness_user_prompt,
    relevance_system_prompt,
    relevance_user_prompt,
    verbosity_system_prompt,
    verbosity_user_prompt,
)
from .utils.schema import QualSchema, ScoreSchema


class HelpfulnessMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="helpfulness")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        response = self._llm.call(
            [
                {"role": "system", "content": helpfulness_system_prompt},
                {
                    "role": "user",
                    "content": helpfulness_user_prompt.format(
                        full_conversation=format_chat_history(score_input.current_turn),
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        return QuantResult(name=self.name, value=response.score, reason=response.reason)


class CoherenceMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="coherence")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        response = self._llm.call(
            [
                {"role": "system", "content": coherence_system_prompt},
                {
                    "role": "user",
                    "content": coherence_user_prompt.format(
                        full_conversation=format_chat_history(score_input.current_turn),
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        return QuantResult(name=self.name, value=response.score, reason=response.reason)


class VerbosityMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="verbosity")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        response = self._llm.call(
            [
                {"role": "system", "content": verbosity_system_prompt},
                {
                    "role": "user",
                    "content": verbosity_user_prompt.format(
                        full_conversation=format_chat_history(score_input.current_turn),
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        # Invert scale: LLM scores 1=verbose (bad) → stored as 5; 5=concise (good) → 1
        return QuantResult(
            name=self.name, value=6 - response.score, reason=response.reason
        )


class RelevanceMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="relevance")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        response = self._llm.call(
            [
                {"role": "system", "content": relevance_system_prompt},
                {
                    "role": "user",
                    "content": relevance_user_prompt.format(
                        full_conversation=format_chat_history(score_input.current_turn),
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        return QuantResult(name=self.name, value=response.score, reason=response.reason)


class FaithfulnessMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="faithfulness")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        response = self._llm.call(
            [
                {"role": "system", "content": faithfulness_system_prompt},
                {
                    "role": "user",
                    "content": faithfulness_user_prompt.format(
                        knowledge=score_input.knowledge,
                        full_conversation=format_chat_history(score_input.current_turn),
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        return QuantResult(name=self.name, value=response.score, reason=response.reason)


class GoalCompletionMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="goal_completion")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        response = self._llm.call(
            [
                {"role": "system", "content": goal_completion_system_prompt},
                {
                    "role": "user",
                    "content": goal_completion_user_prompt.format(
                        full_conversation=format_chat_history(score_input.chat_history),
                        goal=score_input.user_goal,
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        return QuantResult(name=self.name, value=response.score, reason=response.reason)


class AgentBehaviorFailureMetric(QualitativeMetric):
    DESCRIPTION = (
        "Distribution of agent behavior failure types across all evaluated turns."
        " Labels: false information, disobey user request,"
        " lack of specific information, failure to ask for clarification,"
        " repetition, no failure."
    )

    def __init__(self, llm: LLM) -> None:
        super().__init__(name="agent_behavior_failure", description=self.DESCRIPTION)
        self._llm = llm

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response = self._llm.call(
            [
                {"role": "system", "content": agent_behavior_failure_system_prompt},
                {
                    "role": "user",
                    "content": agent_behavior_failure_user_prompt.format(
                        conversation=format_chat_history(score_input.current_turn),
                        knowledge=score_input.knowledge,
                    ),
                },
            ],
            schema=QualSchema,
        )
        return QualResult(
            name=AgentMetrics.AGENT_BEHAVIOR_FAILURE.value,
            value=response.label,
            reason=response.reason,
        )
