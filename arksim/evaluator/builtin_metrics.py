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
    constraint_violation_system_prompt,
    constraint_violation_user_prompt,
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
from .utils.schema import ConstraintViolationSchema, QualSchema, ScoreSchema


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
                        full_conversation=format_chat_history(score_input.chat_history),
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
                        full_conversation=format_chat_history(score_input.chat_history),
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
                        full_conversation=format_chat_history(score_input.chat_history),
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
                        full_conversation=format_chat_history(score_input.chat_history),
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
                        full_conversation=format_chat_history(score_input.chat_history),
                    ),
                },
            ],
            schema=ScoreSchema,
        )
        return QuantResult(name=self.name, value=response.score, reason=response.reason)


class GoalCompletionMetric(QuantitativeMetric):
    def __init__(self, llm: LLM) -> None:
        super().__init__(name="user_goal_completion")
        self._llm = llm

    def score(self, score_input: ScoreInput) -> QuantResult:
        agent_context = score_input.model_extra.get("agent_context", "") if score_input.model_extra else ""
        response = self._llm.call(
            [
                {"role": "system", "content": goal_completion_system_prompt},
                {
                    "role": "user",
                    "content": goal_completion_user_prompt.format(
                        agent_context=agent_context or "(none)",
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
    # Default colors keyed by severity: green for clean, red for critical, etc.
    DEFAULT_LABEL_COLORS: dict[str, str] = {
        "no failure": "#22c55e",  # green  — all good
        "repetition": "#94a3b8",  # slate  — low
        "failure to ask for clarification": "#f59e0b",  # amber  — medium
        "lack of specific information": "#f59e0b",  # amber  — medium
        "disobey user request": "#f97316",  # orange — high
        "false information": "#ef4444",  # red    — critical
        "unsafe action": "#dc2626",  # dark red — critical
        "unsafe state": "#dc2626",  # dark red — critical
    }

    def __init__(self, llm: LLM) -> None:
        super().__init__(
            name="agent_behavior_failure",
            description=self.DESCRIPTION,
            label_colors=self.DEFAULT_LABEL_COLORS,
        )
        self._llm = llm

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        extra = score_input.model_extra or {}
        agent_context = extra.get("agent_context", "")
        agent_constraints = extra.get("agent_constraints", [])
        expected_behavior = extra.get("expected_behavior", "")

        constraints_str = (
            "\n".join(f"- {c}" for c in agent_constraints) if agent_constraints else "(none)"
        )
        response = self._llm.call(
            [
                {"role": "system", "content": agent_behavior_failure_system_prompt},
                {
                    "role": "user",
                    "content": agent_behavior_failure_user_prompt.format(
                        agent_context=agent_context or "(none)",
                        agent_constraints=constraints_str,
                        expected_behavior=expected_behavior or "(none)",
                        user_goal=score_input.user_goal,
                        conversation=format_chat_history(score_input.chat_history),
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


class ConstraintViolationMetric(QualitativeMetric):
    """Evaluates constraint adherence using a single all-at-once LLM call.

    Combines global agent_constraints from the eval config with the
    scenario-level expected_behavior from an agent_response assertion into
    one unified constraints list. Emits a constraint_violation label if any
    constraint is violated; otherwise returns no_failure.

    Constraint results (fulfilled / violated) are stored separately from the
    label so callers can surface per-turn constraints_fulfilled data.
    """

    def __init__(self, llm: LLM) -> None:
        super().__init__(name="constraint_violation")
        self._llm = llm

    @staticmethod
    def build_constraints_list(
        agent_constraints: list[str], expected_behavior: str
    ) -> list[str]:
        """Combine global constraints and scenario-level expected behavior.

        Global agent_constraints and the scenario-level expected_behavior
        (from an agent_response assertion) are merged into a single flat
        list so the LLM evaluates them together in one prompt call.
        """
        constraints: list[str] = list(agent_constraints)
        if expected_behavior:
            constraints.append(expected_behavior)
        return constraints

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        extra = score_input.model_extra or {}
        agent_context = extra.get("agent_context", "")
        agent_constraints: list[str] = extra.get("agent_constraints", [])
        expected_behavior: str = extra.get("expected_behavior", "")

        all_constraints = self.build_constraints_list(agent_constraints, expected_behavior)
        if not all_constraints:
            from .utils.enums import AgentBehaviorFailureType

            return QualResult(
                name=self.name,
                value=AgentBehaviorFailureType.NO_FAILURE.value,
                reason="No constraints defined for this scenario.",
            )

        constraints_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(all_constraints))
        response = self._llm.call(
            [
                {"role": "system", "content": constraint_violation_system_prompt},
                {
                    "role": "user",
                    "content": constraint_violation_user_prompt.format(
                        agent_context=agent_context or "(none)",
                        constraints_list=constraints_list,
                        conversation=format_chat_history(score_input.chat_history),
                    ),
                },
            ],
            schema=ConstraintViolationSchema,
        )

        from .utils.enums import AgentBehaviorFailureType

        if response.violated_constraints:
            label = AgentBehaviorFailureType.CONSTRAINT_VIOLATION.value
        else:
            label = AgentBehaviorFailureType.NO_FAILURE.value

        return QualResult(
            name=self.name,
            value=label,
            reason=response.reason,
            metadata={
                "fulfilled_constraints": response.fulfilled_constraints,
                "violated_constraints": response.violated_constraints,
            },
        )
