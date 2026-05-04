# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextvars
import inspect
import logging
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tqdm import tqdm

from arksim.llms.chat import LLM
from arksim.llms.chat.base.usage import usage_label, usage_scope
from arksim.scenario import Scenarios
from arksim.scenario.entities import AssertionType, ExpectedToolCall

if TYPE_CHECKING:
    from arksim.scenario.entities import Scenario

    from .entities import ErrorScenarioMapping
from arksim.simulation_engine import Conversation, Simulation
from arksim.utils.module_loader import load_module_from_file
from arksim.utils.output import load_json_file, save_json_file

from .base_metric import ChatMessage, QualitativeMetric, QualResult, QuantitativeMetric
from .entities import (
    ConversationEvaluation,
    ConvoItem,
    Evaluation,
    EvaluationInput,
    EvaluationParams,
    TokenUsage,
    TurnEvaluation,
    TurnItem,
    UniqueError,
)
from .error_detection import detect_agent_error
from .evaluate import (
    evaluate_conversation,
    evaluate_turn,
)
from .trajectory_matching import match_trajectory
from .utils.constants import (
    SCORE_NOT_COMPUTED,
    SEVERITY_RANK,
    SKIP_OUTCOMES,
    score_label,
)
from .utils.enums import (
    AGENT_BEHAVIOR_FAILURE_SEVERITY,
    AgentBehaviorFailureType,
    AgentMetrics,
    EvaluationOutcomes,
)

logger = logging.getLogger(__name__)


EVALUATION_SCHEMA_VERSION = "v1.1"
EVALUATOR_VERSION = "v1"


class Evaluator:
    def __init__(
        self,
        params: EvaluationParams,
        llm: LLM | None = None,
        scenarios: Scenarios | None = None,
    ) -> None:
        self.params = params
        self.llm = llm
        self.scenarios = scenarios
        self.evaluation_results: Evaluation | None = None
        self.total_turns: int = 0
        self.total_conversations: int = 0
        self.chat_id_to_label: dict[str, str] = {}
        self._conv_to_scenario: dict[str, str] = {}
        # Build scenario_id -> (expected_tool_calls, match_mode) mapping
        self._scenario_expected: dict[str, tuple[list[ExpectedToolCall], str]] = {}
        if scenarios:
            for s in scenarios.scenarios:
                tc_assertion = s.find_assertion(AssertionType.TOOL_CALLS)
                if tc_assertion:
                    self._scenario_expected[s.scenario_id] = (
                        tc_assertion.expected,
                        tc_assertion.match_mode,
                    )

        logger.info(f"Evaluator initialized: num_workers={params.num_workers}")

    def _process_input(self, entry: Conversation) -> tuple[list[TurnItem], ConvoItem]:
        variables = (
            entry.simulated_user_prompt.variables if entry.simulated_user_prompt else {}
        )
        system_prompt = variables.get("scenario.agent_context", "")
        knowledge = variables.get("scenario.knowledge", [])
        profile = variables.get("scenario.user_profile", "")
        user_goal = variables.get("scenario.goal", "")

        convo_list = []
        all_messages: list[ChatMessage] = []
        turn_id = 0
        input_msg = None

        for msg in entry.conversation_history:
            if msg.role == "simulated_user":
                input_msg = msg.content
            elif msg.role == "assistant":
                output_msg = msg.content
                if input_msg is not None:
                    turn_messages = [
                        ChatMessage(role="user", content=input_msg),
                        ChatMessage(role="assistant", content=output_msg),
                    ]
                    all_messages.extend(turn_messages)

                    turn_tool_calls = msg.tool_calls if msg.tool_calls else None

                    convo_list.append(
                        TurnItem(
                            chat_id=entry.conversation_id,
                            turn_id=turn_id,
                            current_turn=turn_messages,
                            conversation_history=list(all_messages),
                            system_prompt=system_prompt,
                            knowledge=knowledge,
                            profile=profile,
                            user_goal=user_goal,
                            tool_calls=turn_tool_calls,
                        )
                    )
                    turn_id += 1
                    input_msg = None

        convo_item = ConvoItem(
            chat_id=entry.conversation_id,
            chat_history=all_messages,
            system_prompt=system_prompt,
            knowledge=knowledge,
            profile=profile,
            user_goal=user_goal,
            turns=turn_id,
        )
        return convo_list, convo_item

    def _apply_trajectory_matching(
        self,
        conversations: list[Conversation],
        processed_entries: list[tuple[list[TurnItem], ConvoItem]],
        turn_results: dict[str, list[TurnEvaluation]],
    ) -> None:
        """Run conversation-level trajectory matching and attribute failures.

        Aggregates all tool calls across a conversation's turns, matches
        against the scenario's tool_calls assertions, and attributes any
        failure to the last turn that had tool calls.

        Note: when ``num_convos_per_scenario > 1``, every conversation
        for the same scenario is compared against the same expected
        trajectory.  If the simulated user can diverge (e.g. decline a
        cancellation), use ``contains`` mode instead of ``strict`` so the
        agent is not penalised for correctly adapting to user input.
        """
        for entry, (convo_list, _) in zip(
            conversations, processed_entries, strict=False
        ):
            expected_info = self._scenario_expected.get(entry.scenario_id)
            if not expected_info:
                continue
            expected_calls, match_mode_val = expected_info

            # Aggregate tool calls across all turns in the conversation
            all_tool_calls = []
            last_tool_turn_id = -1
            for turn_item in convo_list:
                if turn_item.tool_calls:
                    all_tool_calls.extend(turn_item.tool_calls)
                    last_tool_turn_id = turn_item.turn_id

            traj_result = match_trajectory(
                all_tool_calls, expected_calls, match_mode_val
            )
            if traj_result.matched:
                continue

            turns = turn_results.get(entry.conversation_id, [])
            if last_tool_turn_id >= 0:
                target_turn = next(
                    (t for t in turns if t.turn_id == last_tool_turn_id), None
                )
            else:
                # No tool calls at all; attribute to the last turn
                target_turn = max(turns, key=lambda t: t.turn_id) if turns else None
            if target_turn is None:
                continue

            traj_qual = QualResult(
                name=AgentMetrics.AGENT_BEHAVIOR_FAILURE.value,
                value=traj_result.failure_label,
                reason=f"[Trajectory] {traj_result.reason}",
            )
            target_turn.qual_scores.append(traj_qual)

            if target_turn.turn_behavior_failure in SKIP_OUTCOMES:
                target_turn.turn_behavior_failure = traj_result.failure_label
                target_turn.turn_behavior_failure_reason = traj_qual.reason
            else:
                traj_sev = SEVERITY_RANK.get(
                    AGENT_BEHAVIOR_FAILURE_SEVERITY.get(
                        traj_result.failure_label or "", ""
                    ),
                    99,
                )
                current_sev = SEVERITY_RANK.get(
                    AGENT_BEHAVIOR_FAILURE_SEVERITY.get(
                        target_turn.turn_behavior_failure, ""
                    ),
                    99,
                )
                if traj_sev < current_sev:
                    target_turn.turn_behavior_failure = traj_result.failure_label
                    target_turn.turn_behavior_failure_reason = traj_qual.reason
                else:
                    target_turn.turn_behavior_failure_reason += f"\n{traj_qual.reason}"

    def evaluate(
        self,
        simulation: Simulation,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Evaluation:
        """Evaluate conversations and return results."""
        conversations = simulation.conversations
        simulation_id = simulation.simulation_id
        logger.info(f"Starting evaluation of {len(conversations)} conversations")

        # Pre-process to count total turns for accurate progress tracking
        processed_entries: list[tuple[list[TurnItem], ConvoItem]] = []
        total_turns = 0
        for entry in conversations:
            convo_list, convo_item = self._process_input(entry)
            processed_entries.append((convo_list, convo_item))
            total_turns += len(convo_list)

        logger.info(f"Preprocessing complete: {total_turns} total turns to evaluate")

        # When auto: enough workers for all turns + per-conversation evaluate_conversation
        # calls at once. Each evaluate_conversation call may itself spawn an inner pool
        # for convo-scope custom metrics, but that pool is unbounded in auto mode so
        # additional workers beyond total_turns + len(conversations) are not needed here.
        # When explicit int: use that value as-is.
        if self.params.num_workers == "auto":
            num_workers = total_turns + len(conversations)
        else:
            num_workers = self.params.num_workers

        post_processing_steps = 1  # detect errors
        pbar = tqdm(
            total=total_turns + len(conversations) + post_processing_steps,
            desc="Evaluating...",
            unit="step",
        )

        def _on_turn_complete() -> None:
            pbar.update(1)
            if on_progress:
                on_progress(int(pbar.n), total_turns + len(conversations))

        # Route custom metrics to turn- or conversation-level execution by scope.
        _buckets: dict[tuple[str, str], list] = defaultdict(list)
        for _m in self.params.custom_metrics:
            _kind = "quant" if isinstance(_m, QuantitativeMetric) else "qual"
            _buckets[(_m.scope, _kind)].append(_m)
        _turn_quant = _buckets[("turn", "quant")]
        _turn_qual = _buckets[("turn", "qual")]
        _convo_quant = _buckets[("conversation", "quant")]
        _convo_qual = _buckets[("conversation", "qual")]

        # Phase 1: evaluate all turns in parallel across all conversations
        all_turn_tasks = [
            (turn_item, convo_item)
            for convo_list, convo_item in processed_entries
            for turn_item in convo_list
        ]

        turn_results: dict[str, list[TurnEvaluation]] = defaultdict(list)

        if all_turn_tasks:
            with ThreadPoolExecutor(
                max_workers=min(num_workers, len(all_turn_tasks))
            ) as executor:
                # auto → unlimited inner parallelism (0 = unlimited).
                # explicit int → metrics capped at min(num_workers, num_metrics).
                inner_workers = 0 if self.params.num_workers == "auto" else num_workers
                turn_futures = {
                    executor.submit(
                        contextvars.copy_context().run,
                        evaluate_turn,
                        self.llm,
                        turn_item,
                        _turn_quant or None,
                        _turn_qual or None,
                        self.params.metrics_to_run,
                        inner_workers,
                    ): turn_item
                    for turn_item, _ in all_turn_tasks
                }
                for future in as_completed(turn_futures):
                    turn_item = turn_futures[future]
                    try:
                        turn_eval = future.result()
                        turn_results[turn_item.chat_id].append(turn_eval)
                    except Exception as e:
                        logger.error(
                            f"Error evaluating turn {turn_item.turn_id} "
                            f"of conversation {turn_item.chat_id}: {e}"
                        )
                    _on_turn_complete()

        # Phase 1.5: Conversation-level trajectory matching (deterministic,
        # no LLM cost). Runs before goal completion so trajectory failures
        # are reflected in TSR. Only runs when tool_call_behavior_failure
        # is in metrics_to_run (or metrics_to_run is None/empty, meaning
        # "run all").
        _mtrs = self.params.metrics_to_run
        if self._scenario_expected and (
            not _mtrs or "tool_call_behavior_failure" in _mtrs
        ):
            self._apply_trajectory_matching(
                conversations, processed_entries, turn_results
            )

        # Phase 2: conversation-level evaluation (parallel)
        convo_score_list: list[ConversationEvaluation] = []

        convo_max_workers = max(1, min(num_workers, len(processed_entries)))
        convo_inner_workers = 0 if self.params.num_workers == "auto" else num_workers
        with ThreadPoolExecutor(max_workers=convo_max_workers) as executor:
            convo_futures = {
                executor.submit(
                    contextvars.copy_context().run,
                    evaluate_conversation,
                    self.llm,
                    convo_item,
                    sorted(
                        turn_results.get(convo_item.chat_id, []),
                        key=lambda t: t.turn_id,
                    ),
                    _convo_quant or None,
                    _convo_qual or None,
                    self.params.metrics_to_run,
                    convo_inner_workers,
                ): convo_item
                for _, convo_item in processed_entries
            }
            for future in as_completed(convo_futures):
                convo_item = convo_futures[future]
                try:
                    convo_eval = future.result()
                    convo_score_list.append(convo_eval)
                except Exception as e:
                    logger.error(
                        f"Error evaluating conversation {convo_item.chat_id}: {e}"
                    )
                pbar.update(1)

        pbar.set_description("Detecting agent errors")
        logger.info("Detecting agent errors")
        with usage_label(component="evaluation", phase="error_detection"):
            unique_errors = detect_agent_error(self.llm, convo_score_list)
        logger.info(f"Detected {len(unique_errors)} unique errors")

        # Link unique_error_ids back to TurnEvaluation objects
        turn_to_errors: dict[tuple[str, int], list[str]] = defaultdict(list)
        for error in unique_errors:
            for occ in error.occurrences:
                turn_to_errors[(occ.conversation_id, occ.turn_id)].append(
                    error.unique_error_id
                )
        for conv_eval in convo_score_list:
            for turn in conv_eval.turn_scores:
                turn.unique_error_ids = turn_to_errors.get(
                    (conv_eval.conversation_id, turn.turn_id), []
                )

        pbar.update(1)
        pbar.set_description("Evaluation complete")
        pbar.close()
        logger.info("Post-processing complete")

        # Build chat_id -> label mapping
        self.chat_id_to_label = {
            conv.conversation_id: f"Conversation {i}"
            for i, conv in enumerate(conversations, 1)
        }

        self.evaluation_results = Evaluation(
            schema_version=EVALUATION_SCHEMA_VERSION,
            evaluator_version=EVALUATOR_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            evaluation_id=str(uuid.uuid4()),
            simulation_id=simulation_id,
            conversations=convo_score_list,
            unique_errors=unique_errors,
        )
        self.total_turns = sum(len(conv.turn_scores) for conv in convo_score_list)
        self.total_conversations = len(conversations)

        # Build conv->scenario mapping and compute error-to-scenario mappings
        self._conv_to_scenario = (
            {c.conversation_id: c.scenario_id for c in conversations}
            if self.scenarios
            else {}
        )
        if unique_errors and self.scenarios and self._conv_to_scenario:
            from .error_scenarios import build_error_scenario_data

            mappings, _ = build_error_scenario_data(
                unique_errors=unique_errors,
                conv_to_scenario=self._conv_to_scenario,
                scenarios=self.scenarios,
            )
            self.evaluation_results.error_scenario_mappings = mappings

        logger.info(
            f"Evaluation complete: {self.total_conversations} conversations, "
            f"{self.total_turns} turns, {len(unique_errors)} unique errors"
        )
        return self.evaluation_results

    def save_results(self) -> None:
        """Save evaluation results and focus files to disk.

        Writes ``evaluation.json`` (always) and per-error focus files
        under ``focus/`` (when ``error_scenario_mappings`` is populated
        and the original scenario set is available).

        Raises:
            ValueError: If evaluate() has not been called yet.
        """
        if self.evaluation_results is None:
            logger.error(
                "Attempted to save results but no evaluation results available"
            )
            raise ValueError("No evaluation results to save. Call evaluate() first.")

        eval_path = os.path.join(self.params.output_dir, "evaluation.json")
        if os.path.exists(eval_path):
            logger.warning(f"Overwriting existing file: {eval_path}")
        logger.info(f"Saving evaluation results to {eval_path}")

        save_json_file(
            self.evaluation_results.model_dump(),
            eval_path,
            overwrite=True,
        )
        logger.info("Evaluation results saved successfully")

        # Write focus files (best-effort; failures must not crash the save)
        mappings = self.evaluation_results.error_scenario_mappings
        if mappings and self.scenarios:
            try:
                self._write_focus_files(mappings)
            except Exception:
                logger.exception(
                    "Focus file writing failed; evaluation.json was saved successfully"
                )

    def _write_focus_files(
        self,
        mappings: list[ErrorScenarioMapping],
    ) -> None:
        """Write per-error and combined focus files to disk."""
        scenario_lookup: dict[str, Scenario] = {
            s.scenario_id: s for s in self.scenarios.scenarios
        }
        focus_dir = os.path.join(self.params.output_dir, "focus")
        all_scenario_ids: set[str] = set()

        for mapping in mappings:
            matched = [
                scenario_lookup[sid]
                for sid in mapping.scenario_ids
                if sid in scenario_lookup
            ]
            if not matched:
                continue

            file_path = os.path.join(focus_dir, f"error_{mapping.error_index}.json")
            filtered = Scenarios(
                schema_version=self.scenarios.schema_version,
                scenarios=matched,
            )
            save_json_file(filtered.model_dump(), file_path, overwrite=True)
            all_scenario_ids.update(mapping.scenario_ids)

        if all_scenario_ids:
            all_scenarios = [
                scenario_lookup[sid]
                for sid in sorted(all_scenario_ids)
                if sid in scenario_lookup
            ]
            all_path = os.path.join(focus_dir, "all_failures.json")
            all_bundle = Scenarios(
                schema_version=self.scenarios.schema_version,
                scenarios=all_scenarios,
            )
            save_json_file(all_bundle.model_dump(), all_path, overwrite=True)

        logger.info(
            "Generated %d focus file(s) in %s",
            len(mappings),
            os.path.join(self.params.output_dir, "focus/"),
        )

    @staticmethod
    def _truncate_reason(reason: str | None, max_words: int = 10) -> str:
        """Truncate reason text to max_words with ellipsis."""
        if not reason:
            return ""
        words = reason.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

    def _display_turn_by_turn_metrics(
        self, convo_evaluations: list[ConversationEvaluation]
    ) -> None:
        """Display turn-by-turn evaluation metrics grouped by conversation."""
        title = "TURN-BY-TURN EVALUATION:"
        logger.info(f"\n{title}\n" + "-" * len(title))
        logger.info(
            "Note: Scores range from 1 to 5, where 1 indicates poor "
            "performance and 5 indicates excellent performance.\n",
        )

        skip_labels = {
            EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
            "skipped_all_failed",
            AgentBehaviorFailureType.NO_FAILURE.value,
        }

        for conv_num, conv in enumerate(convo_evaluations, 1):
            logger.info(
                f"Conversation {conv_num}:",
            )
            for turn in sorted(conv.turn_scores, key=lambda t: t.turn_id):
                issues = []

                if (
                    turn.turn_behavior_failure
                    and turn.turn_behavior_failure not in skip_labels
                ):
                    preview = self._truncate_reason(turn.turn_behavior_failure_reason)
                    issues.append(
                        f"Agent Behavior Failure: {turn.turn_behavior_failure}"
                        + (f" ({preview})" if preview else "")
                    )

                for score in turn.scores:
                    if score.value <= 1:
                        preview = self._truncate_reason(score.reason)
                        issues.append(
                            f"{score.name.title()}: {score.value:.1f}"
                            + (f" ({preview})" if preview else "")
                        )

                status = (
                    f"Issues: {', '.join(issues)}" if issues else "No issues detected"
                )
                logger.info(
                    f"  Turn {turn.turn_id + 1}: {status}",
                )
            logger.info("\n")

    def _display_conversation_metrics(
        self, convo_evaluations: list[ConversationEvaluation]
    ) -> None:
        """Display conversation-level evaluation metrics."""
        title = "CONVERSATION-LEVEL METRICS:"
        logger.info(f"\n{title}\n" + "-" * len(title))
        logger.info(
            "Note: Scores range 0-1. Overall Agent Score is a weighted "
            "average of Turn Success Ratio and Goal Completion Rate.",
        )

        for i, c in enumerate(convo_evaluations, 1):
            logger.info(
                f"\nConversation {i} (Turns: {len(c.turn_scores)}):\n"
                f"   - Goal Completion Rate: {c.goal_completion_score:.2f}\n"
                f"   - Turn Success Ratio: {c.turn_success_ratio:.2f}\n"
                f"   - Overall Agent Score: {c.overall_agent_score:.2f}\n"
                f"   - Status: {c.evaluation_status}",
            )

    def _display_top_unique_errors(
        self,
        unique_errors: list[UniqueError],
    ) -> None:
        """Display top unique errors sorted by severity then occurrence count."""
        if not unique_errors:
            logger.info(
                "UNIQUE ERRORS: No agent behavior failures found.",
            )
            return

        top_errors = sorted(
            unique_errors,
            key=lambda e: (
                SEVERITY_RANK.get(e.severity, len(SEVERITY_RANK)),
                -len(e.occurrences),
            ),
        )[:5]
        title = (
            "UNIQUE ERRORS FOUND:"
            if len(top_errors) < 5
            else "TOP 5 UNIQUE ERRORS (by severity):"
        )
        logger.info(f"\n{title}\n" + "-" * len(title))

        # Build lookup from error ID to mapping for focus file paths
        mappings = (
            self.evaluation_results.error_scenario_mappings
            if self.evaluation_results
            else []
        )
        mapping_by_id = {m.unique_error_id: m for m in (mappings or [])}

        for i, err in enumerate(top_errors, 1):
            desc = err.unique_error_description or "N/A"
            category = (
                (err.behavior_failure_category or "N/A").replace("_", " ").title()
            )
            severity = (err.severity or "medium").upper()

            occ_convs = {occ.conversation_id for occ in err.occurrences}
            labels = [self.chat_id_to_label.get(c, c) for c in sorted(occ_convs)]
            occ_line = (
                f"{len(err.occurrences)} occurrences ({', '.join(labels)})"
                if labels
                else f"{len(err.occurrences)} occurrences"
            )

            logger.info(f"\n{i}. [{severity}] {occ_line}")
            logger.info(f"Unique Error: {desc}")
            logger.info(f"Agent Behavior Failure Category: {category}")

            # Show scenario IDs and focus file path from error_scenario_mappings
            mapping = mapping_by_id.get(err.unique_error_id)
            if mapping:
                logger.info(f"Scenarios: {', '.join(mapping.scenario_ids)}")
                logger.info(
                    "Focus file: %s",
                    os.path.join(
                        self.params.output_dir,
                        "focus",
                        f"error_{mapping.error_index}.json",
                    ),
                )
            elif self._conv_to_scenario:
                scenario_ids = sorted(
                    {
                        self._conv_to_scenario[c]
                        for c in occ_convs
                        if c in self._conv_to_scenario
                    }
                )
                if scenario_ids:
                    logger.info(f"Scenarios: {', '.join(scenario_ids)}")

    def _format_metric_score(self, value: float, use_label: bool = True) -> str:
        """Format a metric score, handling not-computed sentinel."""
        if value == SCORE_NOT_COMPUTED:
            return "N/A (Not computed)"
        if use_label:
            return f"{value:.1f} ({score_label(value)})"
        return f"{value:.1f}"

    def _display_failure_breakdown(self, failure_counts: dict) -> None:
        """Display agent behavior failure type breakdown."""
        skip_types = {
            EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
            EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
            EvaluationOutcomes.AGENT_API_ERROR.value,
        }
        filtered = {
            k: v for k, v in failure_counts.items() if k not in skip_types and v > 0
        }
        if not filtered:
            return

        total = sum(filtered.values())
        title = "AGENT BEHAVIOR FAILURE BREAKDOWN:"
        logger.info(f"\n{title}\n" + "-" * len(title))
        for label, count in sorted(filtered.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100 if total > 0 else 0
            logger.info(f"• {label}: {count} ({pct:.1f}%)")
        logger.info(f"Total evaluations: {total}")

    def display_evaluation_summary(self) -> None:
        """Display evaluation summary with agent-specific metrics."""
        logger.info("Displaying evaluation summary")
        results = self.evaluation_results

        logger.info(f"\n{'=' * 60}\nEVALUATION SUMMARY\n{'=' * 60}")

        avg_turns = (
            f"{self.total_turns / self.total_conversations:.1f}"
            if self.total_conversations
            else "N/A"
        )
        logger.info(f"Conversations: {self.total_conversations}")
        logger.info(f"Total Turns: {self.total_turns}")
        logger.info(f"Average Turns per Conversation: {avg_turns}")

        if not results:
            logger.info("\nEvaluation: Not performed or no results available")
            logger.info(f"\n{'=' * 60}")
            return

        # Compute aggregates on-the-fly from Evaluation.
        # Key by (name, scope) so a turn-scope and convo-scope metric that
        # share a name are never pooled into the same bucket.
        metric_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        failure_counts: Counter = Counter()
        for conv in results.conversations:
            for turn in conv.turn_scores:
                for score in turn.scores:
                    metric_scores[(score.name, "turn")].append(score.value)
                failure_counts[turn.turn_behavior_failure] += 1
            for score in conv.convo_scores:
                metric_scores[(score.name, "convo")].append(score.value)

        averages: dict[tuple[str, str], float] = {
            key: sum(vals) / len(vals) for key, vals in metric_scores.items() if vals
        }

        if results.conversations:
            self._display_turn_by_turn_metrics(results.conversations)
            self._display_conversation_metrics(results.conversations)

        # Overall Performance Analysis
        builtin_metrics = [
            "helpfulness",
            "coherence",
            "verbosity",
            "relevance",
            "faithfulness",
        ]
        valid_scores = [
            averages[(m, "turn")]
            for m in builtin_metrics
            if (m, "turn") in averages and averages[(m, "turn")] != SCORE_NOT_COMPUTED
        ]

        if valid_scores:
            title = "OVERALL PERFORMANCE ANALYSIS:"
            logger.info(f"\n{title}\n" + "-" * len(title))
            logger.info(
                "Note: Scores range from 1 to 5, where 1 indicates poor "
                "performance and 5 indicates excellent performance.\n"
            )
            for metric in builtin_metrics:
                key = (metric, "turn")
                if key in averages and averages[key] != SCORE_NOT_COMPUTED:
                    logger.info(
                        f"• {metric.title()}: "
                        f"{self._format_metric_score(averages[key])}"
                    )

            # Custom metrics -- label with scope when the same name appears in
            # both turn and convo scope to avoid ambiguity.
            builtin_set = set(builtin_metrics)
            turn_names = {name for name, scope in averages if scope == "turn"}
            convo_names = {name for name, scope in averages if scope == "convo"}
            shared_names = turn_names & convo_names
            for (name, scope), score in averages.items():
                if name not in builtin_set and score != SCORE_NOT_COMPUTED:
                    label = name.replace("_", " ").title()
                    if name in shared_names:
                        label = f"{label} ({scope})"
                    logger.info(
                        f"• {label}: "
                        f"{self._format_metric_score(score, use_label=False)}"
                    )

            overall_avg = sum(valid_scores) / len(valid_scores)
            logger.info(f"\nOverall Average: {self._format_metric_score(overall_avg)}")
        else:
            logger.info(
                "Evaluation data incomplete - unable to provide detailed analysis."
            )

        if failure_counts:
            self._display_failure_breakdown(dict(failure_counts))

        if results.unique_errors:
            self._display_top_unique_errors(results.unique_errors)

        if results.error_scenario_mappings:
            logger.info("\nFOCUS FILES FOR TARGETED RERUNS:")
            logger.info(
                "  Rerun all failures: arksim simulate-evaluate <config> "
                "--scenario_file_path %s",
                os.path.join(self.params.output_dir, "focus", "all_failures.json"),
            )
            logger.info(
                "  Or target a specific error: --scenario_file_path %s",
                os.path.join(self.params.output_dir, "focus", "error_N.json"),
            )
            logger.info("  Tip: pass --output_dir to avoid overwriting these results.")

        logger.info(f"\n{'=' * 60}")


def _instantiate_with_optional_llm(cls: type, llm: object | None) -> object:
    """Instantiate a metric class, injecting ``llm`` if the signature accepts it.

    Inspects ``cls.__init__`` directly (not the resolved MRO) so that legacy
    metrics overriding ``__init__`` without an ``llm`` param are instantiated
    without injection. Metrics that do not override ``__init__`` inherit the
    base class signature (which includes ``llm``) and receive it automatically.

    Emits a warning when ``**kwargs`` is present in the signature, because
    ``llm`` would be swallowed by ``**kwargs`` without being forwarded to
    ``super().__init__``, causing silent injection failure.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except ValueError:
        return cls()

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        logger.warning(
            "Metric '%s' uses **kwargs in __init__; LLM injection may silently "
            "fail if 'llm' is not explicitly forwarded to super().__init__().",
            cls.__name__,
        )

    if "llm" in params:
        return cls(llm=llm)
    return cls()


def _load_custom_metrics(
    file_paths: list[str],
    llm: object | None = None,
) -> tuple[list[QuantitativeMetric], list[QualitativeMetric]]:
    """Dynamically load QuantitativeMetric and QualitativeMetric subclasses from .py files.

    Args:
        file_paths: Paths to Python files containing
            QuantitativeMetric or QualitativeMetric subclass definitions.
        llm: Optional LLM instance to inject into metrics that accept it.
            Metrics whose ``__init__`` accepts a ``llm`` keyword argument receive
            this instance on ``self.llm``.  Metrics that do not accept ``llm``
            are instantiated without it for backward compatibility.

    Returns:
        Tuple of (quantitative metrics, qualitative metrics).
    """
    metrics: list[QuantitativeMetric] = []
    qual_metrics: list[QualitativeMetric] = []
    for path in file_paths:
        abs_path = os.path.abspath(path)
        module = load_module_from_file(abs_path)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, QuantitativeMetric) and obj is not QuantitativeMetric:
                if inspect.isabstract(obj):
                    logger.debug(
                        "Skipping abstract class '%s' from %s", obj.__name__, abs_path
                    )
                    continue
                try:
                    metrics.append(_instantiate_with_optional_llm(obj, llm))  # type: ignore[arg-type]
                    logger.info(
                        f"Loaded quantitative metric '{obj.__name__}' from {abs_path}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Could not instantiate quantitative metric '{obj.__name__}' "
                        f"from {abs_path}: {e}. "
                        "Make sure the class can be instantiated with at most an "
                        "`llm=` keyword argument (all other parameters must have defaults)."
                    ) from e
            elif issubclass(obj, QualitativeMetric) and obj is not QualitativeMetric:
                if inspect.isabstract(obj):
                    logger.debug(
                        "Skipping abstract class '%s' from %s", obj.__name__, abs_path
                    )
                    continue
                try:
                    qual_metrics.append(_instantiate_with_optional_llm(obj, llm))  # type: ignore[arg-type]
                    logger.info(
                        f"Loaded qualitative metric '{obj.__name__}' from {abs_path}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Could not instantiate qualitative metric '{obj.__name__}' "
                        f"from {abs_path}: {e}. "
                        "Make sure the class can be instantiated with at most an "
                        "`llm=` keyword argument (all other parameters must have defaults)."
                    ) from e
    return metrics, qual_metrics


def run_evaluation(
    settings: EvaluationInput,
    simulation: Simulation | None = None,
    scenarios: Scenarios | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Evaluation:
    """Evaluate simulation using EvaluationInput settings.

    Args:
        settings: EvaluationInput with evaluation settings
        simulation: Optional in-memory simulation output from run_simulation.
            If not provided, load from settings.simulation_file_path.
        scenarios: Optional in-memory scenarios. Used for trajectory matching
            (when scenarios define tool_calls assertions) and HTML report context.
            If not provided, load from settings.scenario_file_path.
    """
    if simulation is None:
        if not settings.simulation_file_path:
            raise ValueError(
                "Either pass Simulation object or set "
                "simulation_file_path in EvaluationInput"
            )
        simulation = Simulation.model_validate(
            load_json_file(settings.simulation_file_path)
        )

    # Load scenarios early so they're available for both trajectory matching
    # and (later) the HTML report.
    if scenarios is None and settings.scenario_file_path:
        try:
            scenarios = Scenarios.load(settings.scenario_file_path)
        except Exception:
            logger.warning(
                "Could not load scenarios; trajectory matching and report "
                "scenario context will be unavailable"
            )

    llm = LLM(
        model=settings.model,
        provider=settings.provider,
    )
    all_quant, all_qual = _load_custom_metrics(
        settings.custom_metrics_file_paths, llm=llm
    )

    params = EvaluationParams(
        output_dir=settings.output_dir,
        num_workers=settings.num_workers,
        custom_metrics=list(all_quant) + list(all_qual),
        metrics_to_run=settings.metrics_to_run or None,
    )

    evaluator = Evaluator(
        params=params,
        llm=llm,
        scenarios=scenarios,
    )

    with usage_scope() as tracker, usage_label(component="evaluation"):
        evaluator_output = evaluator.evaluate(simulation, on_progress=on_progress)

    evaluator_output.usage = TokenUsage(
        total_input_tokens=tracker.total_input_tokens,
        total_output_tokens=tracker.total_output_tokens,
        total_cached_tokens=tracker.total_cached_tokens,
        total_reasoning_tokens=tracker.total_reasoning_tokens,
        by_model=tracker.summary(),
        breakdowns={
            "by_phase": tracker.summary_by("phase", where={"component": "evaluation"}),
        },
    )
    tracker.log_summary()
    evaluator.save_results()
    evaluator.display_evaluation_summary()

    if settings.generate_html_report:
        from arksim.utils.html_report.generate_html_report import (
            HtmlReportParams,
            generate_html_report,
        )

        html_output_path = os.path.join(settings.output_dir, "final_report.html")

        logger.info("Generating HTML report...")
        all_custom = list(all_quant) + list(all_qual)
        metric_descriptions = {
            m.name: m.description for m in all_custom if m.description
        }
        metric_ranges = {m.name: tuple(m.score_range) for m in all_quant}
        qual_label_colors = {m.name: m.label_colors for m in all_qual if m.label_colors}
        report_params = HtmlReportParams(
            simulation=simulation,
            evaluation=evaluator_output,
            scenarios=scenarios,
            output_path=html_output_path,
            chat_id_to_label=evaluator.chat_id_to_label,
            metric_descriptions=metric_descriptions,
            metric_ranges=metric_ranges,
            qual_label_colors=qual_label_colors,
            evaluation_model=settings.model,
            evaluation_provider=settings.provider,
        )
        generate_html_report(report_params)

        logger.info("Successfully generated standalone HTML report!")
        logger.info(
            f"You can now open {os.path.abspath(html_output_path)} directly in your browser.",
        )

    return evaluator_output
