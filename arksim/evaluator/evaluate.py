# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from arksim.llms.chat import LLM
from arksim.simulation_engine import combine_knowledge

from .base_metric import (
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
    ScoreInput,
)
from .builtin_metrics import (
    AgentBehaviorFailureMetric,
    CoherenceMetric,
    FaithfulnessMetric,
    GoalCompletionMetric,
    HelpfulnessMetric,
    RelevanceMetric,
    VerbosityMetric,
)
from .entities import (
    ConversationEvaluation,
    ConvoItem,
    TurnEvaluation,
    TurnItem,
)
from .tool_call_metrics import ToolCallBehaviorFailureMetric
from .utils.constants import (
    BEHAVIOR_FAILURE_THRESHOLD,
    EVALUATION_PARTIAL_FAILURE_THRESHOLD,
    GOAL_COMPLETION_SCORE_WEIGHT,
    SCORE_NOT_COMPUTED,
    SEVERITY_RANK,
    SKIP_OUTCOMES,
    TURN_SUCCESS_RATIO_SCORE_WEIGHT,
)
from .utils.enums import (
    AGENT_BEHAVIOR_FAILURE_SEVERITY,
    AgentMetrics,
    EvaluationOutcomes,
    EvaluationStatus,
)
from .utils.error_messages import SKIPPED_GOOD_PERFORMANCE_REASON

logger = logging.getLogger(__name__)


def _should_run(name: str, metrics_to_run: list[str] | None) -> bool:
    return not metrics_to_run or name in metrics_to_run


def evaluate_turn(
    llm: LLM,
    turn_item: TurnItem,
    custom_metrics: list[QuantitativeMetric] | None = None,
    custom_qualitative_metrics: list[QualitativeMetric] | None = None,
    metrics_to_run: list[str] | None = None,
    num_workers: int = 0,
) -> TurnEvaluation:
    """Evaluate all metrics for a single conversation turn.

    Args:
        llm: The LLM instance to use for evaluation.
        turn_item: The turn to evaluate.
        custom_metrics: Optional list of user-defined quantitative metrics.
        custom_qualitative_metrics: Optional list of user-defined qualitative metrics.
        metrics_to_run: Optional list of metric names to run.
        num_workers: Max concurrent metric LLM calls. 0 means unlimited
            (all metrics in parallel).
    """
    score_input = ScoreInput(
        chat_history=turn_item.conversation_history,
        current_turn=turn_item.current_turn,
        knowledge=combine_knowledge(turn_item.knowledge or []),
        user_goal=turn_item.user_goal,
        profile=turn_item.profile,
    )

    builtin: list[QuantitativeMetric] = [
        HelpfulnessMetric(llm),
        CoherenceMetric(llm),
        VerbosityMetric(llm),
        RelevanceMetric(llm),
        FaithfulnessMetric(llm),
    ]
    # Builtins are filtered by metrics_to_run; custom metrics always run.
    all_metrics: list[QuantitativeMetric] = [
        m for m in builtin if _should_run(m.name, metrics_to_run)
    ] + (custom_metrics or [])

    def _run(metric: QuantitativeMetric) -> QuantResult:
        result = metric.score(
            ScoreInput(
                chat_history=score_input.chat_history,
                current_turn=score_input.current_turn,
                knowledge=score_input.knowledge,
                user_goal=score_input.user_goal,
                profile=score_input.profile,
                **metric.additional_input,
            )
        )
        return QuantResult(
            name=metric.name, value=result.value, reason=result.reason or ""
        )

    def _run_qual(metric: QualitativeMetric) -> QualResult:
        return metric.evaluate(
            ScoreInput(
                chat_history=score_input.chat_history,
                current_turn=score_input.current_turn,
                knowledge=score_input.knowledge,
                user_goal=score_input.user_goal,
                profile=score_input.profile,
                **getattr(metric, "additional_input", {}),
            )
        )

    metric_tasks: list[tuple[str, Callable[[], QuantResult]]] = [
        (m.name, lambda metric=m: _run(metric)) for m in all_metrics
    ]
    qual_tasks: list[tuple[str, Callable[[], QualResult]]] = [
        (m.name, lambda metric=m: _run_qual(metric))
        for m in (custom_qualitative_metrics or [])
    ]
    all_tasks = metric_tasks + qual_tasks

    scores: list[QuantResult] = []
    qual_scores: list[QualResult] = []

    if all_tasks:
        _max = min(num_workers, len(all_tasks)) if num_workers > 0 else len(all_tasks)
        # This inner pool is nested inside the outer per-turn pool in
        # evaluator.py. Total LLM concurrency is roughly
        # outer_workers * metrics_per_turn (unbounded in "auto" mode),
        # so large runs may hit rate limits.
        with ThreadPoolExecutor(max_workers=_max) as pool:
            score_futures = {pool.submit(fn): name for name, fn in metric_tasks}
            qual_futures = {pool.submit(fn): name for name, fn in qual_tasks}
            all_futures = {**score_futures, **qual_futures}

            for future in as_completed(all_futures):
                name = all_futures[future]
                try:
                    result = future.result()
                    if future in score_futures:
                        scores.append(result)
                    else:
                        qual_scores.append(result)
                except Exception as e:
                    logger.warning(
                        f"Metric '{name}' failed for turn {turn_item.turn_id}: {e}"
                    )

    metric_max = {m.name: m.score_range[1] for m in all_metrics}
    has_threshold_failure = any(
        s.value < BEHAVIOR_FAILURE_THRESHOLD * metric_max[s.name]
        for s in scores
        if s.name in metric_max
    )

    turn_score = (
        sum(s.value for s in scores) / len(scores) if scores else SCORE_NOT_COMPUTED
    )

    if _should_run("agent_behavior_failure", metrics_to_run) and has_threshold_failure:
        qual = AgentBehaviorFailureMetric(llm).evaluate(score_input)
        turn_behavior_failure = qual.value
        turn_behavior_failure_reason = qual.reason or ""
    else:
        turn_behavior_failure = EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value
        turn_behavior_failure_reason = SKIPPED_GOOD_PERFORMANCE_REASON

    # Tool call behavior failure check (independent of threshold, runs when turn has tool calls)
    if (
        _should_run("tool_call_behavior_failure", metrics_to_run)
        and turn_item.tool_calls
    ):
        tool_score_input = ScoreInput(
            chat_history=score_input.chat_history,
            current_turn=score_input.current_turn,
            knowledge=score_input.knowledge,
            user_goal=score_input.user_goal,
            profile=score_input.profile,
            tool_calls=turn_item.tool_calls,
        )
        tool_qual = ToolCallBehaviorFailureMetric(llm).evaluate(tool_score_input)

        # Surface tool call failures under agent_behavior_failure so the
        # HTML report and downstream consumers treat them uniformly.
        tool_qual_as_agent = QualResult(
            name=AgentMetrics.AGENT_BEHAVIOR_FAILURE.value,
            value=tool_qual.value,
            reason=f"[Tool call] {tool_qual.reason}" if tool_qual.reason else "",
        )
        qual_scores.append(tool_qual_as_agent)

        if tool_qual.value not in SKIP_OUTCOMES:
            if turn_behavior_failure in SKIP_OUTCOMES:
                turn_behavior_failure = tool_qual.value
                turn_behavior_failure_reason = tool_qual_as_agent.reason
            else:
                # Both agent and tool call failures detected. Keep the
                # higher-severity label so safety issues are not masked.
                tool_sev = SEVERITY_RANK.get(
                    AGENT_BEHAVIOR_FAILURE_SEVERITY.get(tool_qual.value, ""), 99
                )
                agent_sev = SEVERITY_RANK.get(
                    AGENT_BEHAVIOR_FAILURE_SEVERITY.get(turn_behavior_failure, ""), 99
                )
                if tool_sev < agent_sev:
                    turn_behavior_failure = tool_qual.value
                    turn_behavior_failure_reason = tool_qual_as_agent.reason
                else:
                    turn_behavior_failure_reason += (
                        f" [Tool call] {tool_qual.reason or ''}"
                    )

    return TurnEvaluation(
        turn_id=turn_item.turn_id,
        scores=scores,
        turn_score=turn_score,
        turn_behavior_failure=turn_behavior_failure,
        turn_behavior_failure_reason=turn_behavior_failure_reason,
        qual_scores=qual_scores,
    )


def evaluate_goal_completion(
    llm: LLM,
    convo_item: ConvoItem,
    turns_details: list[TurnEvaluation],
    metrics_to_run: list[str] | None = None,
) -> ConversationEvaluation:
    """Compute goal completion and build ConversationEvaluation.

    Args:
        llm: The LLM instance to use for evaluation.
        convo_item: The conversation item with full history and metadata.
        turns_details: Completed TurnEvaluation objects for this conversation.
        metrics_to_run: Optional list of metric names to run.
    """
    if _should_run("goal_completion", metrics_to_run):
        result = GoalCompletionMetric(llm).score(
            ScoreInput(
                chat_history=convo_item.chat_history,
                user_goal=convo_item.user_goal,
            )
        )
        goal_completion_score = result.value
        goal_completion_reason = result.reason
    else:
        goal_completion_score = SCORE_NOT_COMPUTED
        goal_completion_reason = ""

    behavior_failures = sum(
        1 for t in turns_details if t.turn_behavior_failure not in SKIP_OUTCOMES
    )
    num_turns = convo_item.turns
    turn_success_ratio = (
        (num_turns - behavior_failures) / num_turns if num_turns > 0 else 1.0
    )

    if goal_completion_score < 0:
        # goal_completion was skipped; score based only on turn success
        overall_agent_score = turn_success_ratio
    else:
        overall_agent_score = (
            turn_success_ratio * TURN_SUCCESS_RATIO_SCORE_WEIGHT
            + goal_completion_score * GOAL_COMPLETION_SCORE_WEIGHT
        )

    # Note: overall_agent_score cannot be SCORE_NOT_COMPUTED (-1) here because
    # if goal_completion is skipped (score < 0), we fall back to turn_success_ratio
    # which is always in [0.0, 1.0].
    if overall_agent_score == 1.0:
        status = EvaluationStatus.DONE.value
    elif overall_agent_score >= EVALUATION_PARTIAL_FAILURE_THRESHOLD:
        status = EvaluationStatus.PARTIAL_FAILURE.value
    else:
        status = EvaluationStatus.FAILED.value

    return ConversationEvaluation(
        conversation_id=convo_item.chat_id,
        goal_completion_score=goal_completion_score,
        goal_completion_reason=goal_completion_reason,
        turn_success_ratio=turn_success_ratio,
        overall_agent_score=overall_agent_score,
        evaluation_status=status,
        turn_scores=turns_details,
    )
