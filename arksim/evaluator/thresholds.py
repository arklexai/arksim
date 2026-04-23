# SPDX-License-Identifier: Apache-2.0
"""Public threshold gate functions for use after evaluation.

These can be imported directly when running ArkSim programmatically:

    from arksim.evaluator import check_numeric_thresholds, check_qualitative_failure_labels
"""

from __future__ import annotations

import logging

from .entities import Evaluation

logger = logging.getLogger(__name__)


# System outcomes for agent_behavior_failure that are not real failures.
QUAL_SKIP_OUTCOMES = frozenset(
    {"skipped_good_performance", "evaluation_run_failed", "agent_api_error"}
)


def check_numeric_thresholds(
    evaluator_output: Evaluation,
    numeric_thresholds: dict[str, float] | None,
) -> bool:
    """Check per-metric numeric score thresholds on each metric's native scale.

    Per-conversation score = mean of all per-turn scores for that metric (1–5 scale
    for built-in metrics). Every conversation must meet the threshold.
    ``goal_completion`` uses its per-conversation score directly (stored as 0–1).
    ``overall_score`` checks ``overall_agent_score`` directly (stored as 0–1).

    Args:
        evaluator_output: The ``Evaluation`` returned by ``Evaluator.evaluate()``.
        numeric_thresholds: Mapping of metric name to minimum required score.
            Use ``'overall_score'`` to gate on the per-conversation overall_agent_score (0–1).
            Pass ``None`` or an empty dict to skip all checks.

    Returns:
        ``True`` if every conversation meets every threshold, ``False`` otherwise.
    """
    if not numeric_thresholds:
        return True

    all_passed = True
    for metric_name, threshold in numeric_thresholds.items():
        failed_conversations = []
        for convo in evaluator_output.conversations:
            if metric_name == "overall_score":
                score: float | None = convo.overall_agent_score
            elif metric_name == "goal_completion":
                raw = convo.goal_completion_score
                score: float | None = raw if raw >= 0 else None
            else:
                # Conversation-level custom metrics live on conv.scores.
                convo_match = next(
                    (r for r in convo.scores if r.name == metric_name and r.value >= 0),
                    None,
                )
                if convo_match is not None:
                    score = convo_match.value
                else:
                    # Turn-level metrics: average across turns.
                    values = [
                        r.value
                        for turn in convo.turn_scores
                        for r in turn.scores
                        if r.name == metric_name and r.value >= 0
                    ]
                    score = sum(values) / len(values) if values else None

            if score is None:
                logger.warning(
                    f"Metric '{metric_name}' not found in conversation "
                    f"'{convo.conversation_id}', skipping threshold check for it."
                )
                continue

            if score < threshold:
                failed_conversations.append(
                    {"conversation_id": convo.conversation_id, "score": score}
                )

        if failed_conversations:
            all_passed = False
            logger.error(
                f"Metric threshold failed: '{metric_name}' requires >= {threshold}. "
                f"Failed conversations: {len(failed_conversations)}"
            )
            for fc in failed_conversations:
                logger.error(
                    f"  Conversation {fc['conversation_id']}: "
                    f"{metric_name}={fc['score']:.3f} < {threshold}"
                )
        else:
            logger.info(
                f"Metric threshold passed: '{metric_name}' >= {threshold} "
                f"for all {len(evaluator_output.conversations)} conversations"
            )

    return all_passed


def check_qualitative_failure_labels(
    evaluator_output: Evaluation,
    qualitative_failure_labels: dict[str, list[str]] | None,
) -> bool:
    """Check per-metric qualitative failure label gates.

    Any evaluated turn whose label appears in the failure list fails the run.
    Turns where the metric did not run are skipped. ``agent_behavior_failure`` uses
    the dedicated turn field; all other qualitative metrics use ``turn.qual_scores``.

    Args:
        evaluator_output: The ``Evaluation`` returned by ``Evaluator.evaluate()``.
        qualitative_failure_labels: Mapping of metric name to a list of labels
            that should trigger failure.
            Pass ``None`` or an empty dict to skip all checks.

    Returns:
        ``True`` if no failure labels were found, ``False`` otherwise.
    """
    if not qualitative_failure_labels:
        return True

    all_passed = True
    for metric_name, failure_labels in qualitative_failure_labels.items():
        failed_conversations = []
        for convo in evaluator_output.conversations:
            failures: list[dict[str, object]] = []

            # Check conversation-level qual scores.
            for qs in convo.qual_scores:
                if qs.name == metric_name and qs.value in failure_labels:
                    failures.append({"turn_id": "conversation", "label": qs.value})

            # Check turn-level qual scores.
            for turn in convo.turn_scores:
                if metric_name == "agent_behavior_failure":
                    label = turn.turn_behavior_failure
                    if label in QUAL_SKIP_OUTCOMES:
                        continue
                else:
                    match = next(
                        (q for q in turn.qual_scores if q.name == metric_name), None
                    )
                    if match is None:
                        continue
                    label = match.value

                if label in failure_labels:
                    failures.append({"turn_id": turn.turn_id, "label": label})

            if failures:
                failed_conversations.append(
                    {
                        "conversation_id": convo.conversation_id,
                        "failing_turns": failures,
                    }
                )

        if failed_conversations:
            all_passed = False
            logger.error(
                f"Qualitative gate failed: '{metric_name}' matched failure label(s) "
                f"{failure_labels}. Failed conversations: {len(failed_conversations)}"
            )
            for fc in failed_conversations:
                for t in fc["failing_turns"]:
                    logger.error(
                        f"  Conversation {fc['conversation_id']} turn {t['turn_id']}: "
                        f"{metric_name}='{t['label']}' is in {failure_labels}"
                    )
        else:
            logger.info(
                f"Qualitative gate passed: '{metric_name}' - no failure labels found"
            )

    return all_passed
