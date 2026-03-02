import contextlib
import logging
import uuid

from arksim.llms.chat import LLM

from .entities import ConversationEvaluation, Occurrence, UniqueError
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


def detect_agent_error(
    llm: LLM, convo_evaluations: list[ConversationEvaluation]
) -> list[UniqueError]:
    try:
        agent_behavior_failure_categories_list = []
        for failure_type in EvaluationOutcomes.AgentBehaviorFailureType:
            if failure_type != EvaluationOutcomes.AgentBehaviorFailureType.NO_FAILURE:
                agent_behavior_failure_categories_list.append(failure_type.value)

        reasoning_items = collect_agent_behavior_failure_reasoning(
            convo_evaluations, agent_behavior_failure_categories_list
        )

        unique_errors: list[UniqueError] = []
        if reasoning_items:
            unique_errors_evaluator = UniqueErrors(llm)
            raw_errors = unique_errors_evaluator.evaluate(reasoning_items)
            for raw in raw_errors:
                occurrences = []
                for occ_str in raw.occurrences:
                    parts = occ_str.rsplit("_", 1)
                    if len(parts) == 2:
                        with contextlib.suppress(ValueError):
                            occurrences.append(
                                Occurrence(
                                    conversation_id=parts[0],
                                    turn_id=int(parts[1]),
                                )
                            )
                category = raw.agent_behavior_failure_category
                unique_errors.append(
                    UniqueError(
                        unique_error_id=str(uuid.uuid4()),
                        behavior_failure_category=category,
                        unique_error_description=raw.unique_error_description,
                        severity=AGENT_BEHAVIOR_FAILURE_SEVERITY.get(
                            category.lower(), "medium"
                        ),
                        occurrences=occurrences,
                    )
                )
        return unique_errors
    except Exception as e:
        logger.error(f"Error detecting agent errors: {e}")
        return []


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
