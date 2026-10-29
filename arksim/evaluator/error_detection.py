# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import uuid

from arksim.llms.chat import LLM

from .base_metric import QualResult, QuantResult
from .entities import ConversationEvaluation, Occurrence, UniqueError
from .utils.constants import BEHAVIOR_FAILURE_THRESHOLD
from .utils.enums import (
    AGENT_BEHAVIOR_FAILURE_SEVERITY,
    EvaluationOutcomes,
)
from .utils.prompts import find_unique_errors_prompt
from .utils.schema import (
    UniqueErrorSchema,
    UniqueErrorsSchema,
)

logger = logging.getLogger(__name__)


#: Category of unique errors built from failed scenario assertions. They are
#: listed ahead of behavior failures.
ASSERTION_FAILURE_CATEGORY = "assertion failed"
ASSERTION_FAILURE_SEVERITY = "high"

#: Range assumed for a quantitative metric missing from ``score_ranges``: the
#: built-in metrics' scale.
DEFAULT_SCORE_RANGE = (1.0, 5.0)
#: Most metric items sent for grouping, lowest scores first, so a large run
#: stays within one prompt.
# ponytail: a flat cap; batch the grouping call if runs routinely exceed it.
MAX_METRIC_ITEMS = 300
_GOAL_COMPLETION = "goal_completion"
#: Labels that mean nothing went wrong, and the metrics already collected
#: through ``turn_behavior_failure``.
_NO_PROBLEM_LABELS = {
    EvaluationOutcomes.AgentBehaviorFailureType.NO_FAILURE.value,
    EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
    EvaluationOutcomes.AGENT_API_ERROR.value,
    EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
    "unknown",
}
_BEHAVIOR_FAILURE_METRICS = {"agent_behavior_failure", "Agent Behavior Failure"}


def detect_agent_error(
    llm: LLM,
    convo_evaluations: list[ConversationEvaluation],
    score_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[UniqueError]:
    """Group failed assertions, behavior failures and low metric results into unique errors.

    Failed assertions come from every run; behavior failures only when the
    agent_behavior_failure metric ran; metric items from every other metric
    that ran (see ``collect_metric_failures``). A failure sharing a root cause
    with a failed assertion is merged into the assertion's error.

    ``score_ranges`` maps a quantitative metric's name to its (min, max).
    """
    try:
        agent_behavior_failure_categories_list = []
        for failure_type in EvaluationOutcomes.AgentBehaviorFailureType:
            if failure_type != EvaluationOutcomes.AgentBehaviorFailureType.NO_FAILURE:
                agent_behavior_failure_categories_list.append(failure_type.value)

        assertion_items = collect_assertion_failures(convo_evaluations)
        reasoning_items = collect_agent_behavior_failure_reasoning(
            convo_evaluations, agent_behavior_failure_categories_list
        )
        reasoning_items += collect_metric_failures(convo_evaluations, score_ranges)
        # Item key the LLM echoes back -> where it happened. Assertion items
        # are keyed by conversation, behavior items by conversation and turn.
        locations: dict[str, Occurrence] = {}
        for conv in convo_evaluations:
            locations[conv.conversation_id] = Occurrence(
                conversation_id=conv.conversation_id, turn_id=None
            )
            for turn in conv.turn_scores:
                locations[f"{conv.conversation_id}_{turn.turn_id}"] = Occurrence(
                    conversation_id=conv.conversation_id, turn_id=turn.turn_id
                )

        # Metric names by lowercase, so a metric category comes back as named.
        metric_names = {
            score.name.lower(): score.name
            for conv in convo_evaluations
            for score in [
                *conv.convo_scores,
                *conv.convo_qual_scores,
                *(
                    s
                    for turn in conv.turn_scores
                    for s in [*turn.scores, *turn.qual_scores]
                ),
            ]
            if score.name not in _BEHAVIOR_FAILURE_METRICS
        }
        # Plain arksim scores goal completion outside convo_scores.
        if any(c.goal_completion_score >= 0 for c in convo_evaluations):
            metric_names.setdefault(_GOAL_COMPLETION, _GOAL_COMPLETION)

        unique_errors: list[UniqueError] = []
        if assertion_items or reasoning_items:
            unique_errors_evaluator = UniqueErrors(llm)
            raw_errors = unique_errors_evaluator.evaluate(
                [*assertion_items, *reasoning_items]
            )
            for raw in raw_errors:
                occurrences = [
                    locations[key] for key in raw.occurrences if key in locations
                ]
                category = raw.agent_behavior_failure_category
                is_assertion = category.lower() == ASSERTION_FAILURE_CATEGORY
                metric_name = metric_names.get(category.lower())
                unique_errors.append(
                    UniqueError(
                        unique_error_id=str(uuid.uuid4()),
                        behavior_failure_category=(
                            ASSERTION_FAILURE_CATEGORY
                            if is_assertion
                            else metric_name or category
                        ),
                        source=(
                            "assertion"
                            if is_assertion
                            else "metric"
                            if metric_name
                            else "behavior"
                        ),
                        unique_error_description=raw.unique_error_description,
                        severity=(
                            ASSERTION_FAILURE_SEVERITY
                            if is_assertion
                            else AGENT_BEHAVIOR_FAILURE_SEVERITY.get(
                                category.lower(), "medium"
                            )
                        ),
                        occurrences=occurrences,
                    )
                )
        # Stable sort: assertion errors first, the LLM's order otherwise.
        unique_errors.sort(
            key=lambda e: e.behavior_failure_category != ASSERTION_FAILURE_CATEGORY
        )
        return unique_errors
    except Exception as e:
        logger.error(f"Error detecting agent errors: {e}")
        return []


def collect_assertion_failures(
    convo_evaluations: list[ConversationEvaluation],
) -> list[str]:
    """Collect failed assertions as items for unique error analysis.

    Assertions judge a whole conversation, so items are keyed by conversation
    rather than turn.
    """
    items: list[str] = []
    for conv in convo_evaluations:
        for result in conv.assertion_results:
            if result.status != "fail":
                continue
            items.append(
                f"Item {conv.conversation_id}: "
                f"{ASSERTION_FAILURE_CATEGORY}: {result.label}: {result.reason}"
            )
    return items


def collect_agent_behavior_failure_reasoning(
    convo_evaluations: list[ConversationEvaluation],
    failure_categories: list[str],
) -> list[str]:
    """Collect agent behavior failure reasoning items for unique error analysis.

    Only collects turns where agent behavior failure was detected (not
    'no failure' or 'skipped_good_performance').

    Args:
        convo_evaluations: List of conversation evaluation results.
        failure_categories: List of high-level failure categories to include.

    Returns:
        List of reasoning strings for error finding.
    """
    skip_labels = {
        EvaluationOutcomes.AgentBehaviorFailureType.NO_FAILURE.value,
        EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
        EvaluationOutcomes.AGENT_API_ERROR.value,
        EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
        "unknown",
    }

    reasoning_items = []
    for conv in convo_evaluations:
        for turn in conv.turn_scores:
            failure_label = turn.turn_behavior_failure
            failure_reason = turn.turn_behavior_failure_reason

            if failure_label in skip_labels:
                continue
            if failure_label not in failure_categories:
                continue

            reasoning_str = (
                f"Item {conv.conversation_id}_{turn.turn_id}: "
                f"agent_behavior_failure label and reason: "
                f"{failure_label}: {failure_reason}"
            )
            reasoning_items.append(reasoning_str)
    return reasoning_items


def _is_low(value: float, score_range: tuple[float, float]) -> bool:
    """Below the threshold share of the maximum, the same test that marks a turn
    as a behavior failure in ``evaluate``."""
    low, high = score_range
    if high <= low:
        return False
    return value < BEHAVIOR_FAILURE_THRESHOLD * high


def collect_metric_failures(
    convo_evaluations: list[ConversationEvaluation],
    score_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Collect metric results that point at a problem, for unique error analysis.

    A quantitative score counts when it falls below
    ``BEHAVIOR_FAILURE_THRESHOLD`` of its maximum. A qualitative label counts
    unless it is known to mean no problem; a custom label carries no polarity,
    so the grouping prompt drops the items whose reason describes none.
    Agent behavior failure is left out: it is collected per turn already.
    """
    ranges = score_ranges or {}
    scored: list[tuple[float, str]] = []

    def quant(
        key: str, score: QuantResult, score_range: tuple[float, float] | None = None
    ) -> None:
        if (score.metadata or {}).get("inconclusive_reason"):
            return
        score_range = score_range or ranges.get(score.name, DEFAULT_SCORE_RANGE)
        if not _is_low(score.value, score_range):
            return
        low, high = score_range
        relative = (score.value - low) / (high - low)
        scored.append(
            (
                relative,
                f"Item {key}: {score.name} scored {score.value:g} of {high:g}: "
                f"{score.reason or ''}",
            )
        )

    def qual(key: str, score: QualResult, behavior_reason: str = "") -> None:
        meta = score.metadata or {}
        if meta.get("inconclusive_reason") or score.value in _NO_PROBLEM_LABELS:
            return
        if score.name in _BEHAVIOR_FAILURE_METRICS or (
            meta.get("metric_id") == "agent_behavior_failure"
        ):
            return
        # A tool call failure is folded into turn_behavior_failure as well.
        if score.reason and score.reason in behavior_reason:
            return
        # Labels have no score; they sort after every low quantitative score.
        scored.append(
            (1.0, f"Item {key}: {score.name} label {score.value}: {score.reason or ''}")
        )

    for conv in convo_evaluations:
        cid = conv.conversation_id
        has_goal_score = False
        for score in conv.convo_scores:
            if _GOAL_COMPLETION in (
                score.name,
                (score.metadata or {}).get("metric_id"),
            ):
                has_goal_score = True
            quant(cid, score)
        for score in conv.convo_qual_scores:
            qual(cid, score)
        # Plain arksim keeps goal completion (0 to 1) off convo_scores.
        if not has_goal_score and conv.goal_completion_score >= 0:
            quant(
                cid,
                QuantResult(
                    name=_GOAL_COMPLETION,
                    value=conv.goal_completion_score,
                    reason=conv.goal_completion_reason,
                    scope="conversation",
                ),
                (0.0, 1.0),
            )
        for turn in conv.turn_scores:
            key = f"{cid}_{turn.turn_id}"
            for score in turn.scores:
                quant(key, score)
            for score in turn.qual_scores:
                qual(key, score, turn.turn_behavior_failure_reason)

    scored.sort(key=lambda item: item[0])
    return [item for _, item in scored[:MAX_METRIC_ITEMS]]


class UniqueErrors:
    """Find unique errors from agent behavior failure cases."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def evaluate(self, reasoning_items: list[str]) -> list[UniqueErrorSchema]:
        """Find unique errors from agent behavior failure reasoning items.

        Args:
            reasoning_items: List of reasoning strings for error finding.

        Returns:
            List of UniqueErrorSchema objects.
        """
        if not reasoning_items:
            return []

        items_str = "\n".join(reasoning_items)
        prompt = find_unique_errors_prompt.format(items=items_str)
        response = self.llm.call(
            [{"role": "system", "content": prompt}],
            schema=UniqueErrorsSchema,
        )
        return response.unique_errors
