# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import logging
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm

from arksim.llms.chat import LLM
from arksim.scenario import Scenarios
from arksim.simulation_engine import Conversation, Simulation
from arksim.utils.module_loader import load_module_from_file
from arksim.utils.output import load_json_file, save_json_file

from .base_metric import ChatMessage, QualitativeMetric, QuantitativeMetric
from .entities import (
    ConversationEvaluation,
    ConvoItem,
    Evaluation,
    EvaluationInput,
    EvaluationParams,
    TurnEvaluation,
    TurnItem,
    UniqueError,
)
from .error_detection import detect_agent_error
from .evaluate import (
    evaluate_goal_completion,
    evaluate_turn,
)
from .utils.constants import SCORE_NOT_COMPUTED, score_label
from .utils.enums import AgentBehaviorFailureType, EvaluationOutcomes

logger = logging.getLogger(__name__)


EVALUATION_SCHEMA_VERSION = "v1"
EVALUATOR_VERSION = "v1"


class Evaluator:
    def __init__(
        self,
        params: EvaluationParams,
        llm: LLM | None = None,
    ) -> None:
        self.params = params
        self.llm = llm
        self.evaluation_results: Evaluation | None = None
        self.total_turns: int = 0
        self.total_conversations: int = 0
        self.chat_id_to_label: dict[str, str] = {}
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

        # When auto: enough workers for all turns + goal_completion calls at once.
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
                        evaluate_turn,
                        self.llm,
                        turn_item,
                        self.params.custom_metrics or None,
                        self.params.custom_qualitative_metrics or None,
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

        # Phase 2: goal_completion for each conversation (parallel)
        convo_score_list: list[ConversationEvaluation] = []

        gc_max_workers = max(1, min(num_workers, len(processed_entries)))
        with ThreadPoolExecutor(max_workers=gc_max_workers) as executor:
            gc_futures = {
                executor.submit(
                    evaluate_goal_completion,
                    self.llm,
                    convo_item,
                    sorted(
                        turn_results.get(convo_item.chat_id, []),
                        key=lambda t: t.turn_id,
                    ),
                    self.params.metrics_to_run,
                ): convo_item
                for _, convo_item in processed_entries
            }
            for future in as_completed(gc_futures):
                convo_item = gc_futures[future]
                try:
                    convo_eval = future.result()
                    convo_score_list.append(convo_eval)
                except Exception as e:
                    logger.error(
                        f"Error computing goal completion for {convo_item.chat_id}: {e}"
                    )
                pbar.update(1)

        pbar.set_description("Detecting agent errors")
        logger.info("Detecting agent errors")
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

        logger.info(
            f"Evaluation complete: {self.total_conversations} conversations, "
            f"{self.total_turns} turns, {len(unique_errors)} unique errors"
        )
        return self.evaluation_results

    def save_results(self) -> None:
        """Save evaluation results to evaluation.json.

        Raises:
            ValueError: If evaluate() has not been called yet.
        """
        if self.evaluation_results is None:
            logger.error(
                "Attempted to save results but no evaluation results available"
            )
            raise ValueError("No evaluation results to save. Call evaluate() first.")

        save_dir = os.path.join(self.params.output_dir, "evaluation.json")
        if os.path.exists(save_dir):
            logger.warning(f"Overwriting existing file: {save_dir}")
        logger.info(f"Saving evaluation results to {save_dir}")

        save_json_file(
            self.evaluation_results.model_dump(),
            save_dir,
            overwrite=True,
        )
        logger.info("Evaluation results saved successfully")

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

    _SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def _display_top_unique_errors(self, unique_errors: list[UniqueError]) -> None:
        """Display top unique errors sorted by severity then occurrence count."""
        if not unique_errors:
            logger.info(
                "UNIQUE ERRORS: No agent behavior failures found.",
            )
            return

        top_errors = sorted(
            unique_errors,
            key=lambda e: (
                self._SEVERITY_ORDER.get(e.severity, 2),
                -len(e.occurrences),
            ),
        )[:5]
        title = (
            "UNIQUE ERRORS FOUND:"
            if len(top_errors) < 5
            else "TOP 5 UNIQUE ERRORS (by severity):"
        )
        logger.info(f"\n{title}\n" + "-" * len(title))

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

        logger.info(
            "Note: Suggested fixes are module-level heuristics and may not apply to your architecture."
        )

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

        # Compute aggregates on-the-fly from Evaluation
        metric_scores: dict[str, list[float]] = defaultdict(list)
        failure_counts: Counter = Counter()
        for conv in results.conversations:
            for turn in conv.turn_scores:
                for score in turn.scores:
                    metric_scores[score.name].append(score.value)
                failure_counts[turn.turn_behavior_failure] += 1

        averages: dict[str, float] = {
            metric: sum(vals) / len(vals)
            for metric, vals in metric_scores.items()
            if vals
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
            averages[m]
            for m in builtin_metrics
            if m in averages and averages[m] != SCORE_NOT_COMPUTED
        ]

        if valid_scores:
            title = "OVERALL PERFORMANCE ANALYSIS:"
            logger.info(f"\n{title}\n" + "-" * len(title))
            logger.info(
                "Note: Scores range from 1 to 5, where 1 indicates poor "
                "performance and 5 indicates excellent performance.\n"
            )
            for metric in builtin_metrics:
                if metric in averages and averages[metric] != SCORE_NOT_COMPUTED:
                    logger.info(
                        f"• {metric.title()}: "
                        f"{self._format_metric_score(averages[metric])}"
                    )

            # Custom metrics
            for name, score in averages.items():
                if name not in builtin_metrics and score != SCORE_NOT_COMPUTED:
                    logger.info(
                        f"• {name.replace('_', ' ').title()}: "
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

        logger.info(f"\n{'=' * 60}")


def _load_custom_metrics(
    file_paths: list[str],
) -> tuple[list[QuantitativeMetric], list[QualitativeMetric]]:
    """Dynamically load QuantitativeMetric and QualitativeMetric subclasses from .py files.

    Args:
        file_paths: Paths to Python files containing
            QuantitativeMetric or QualitativeMetric subclass definitions.

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
                try:
                    metrics.append(obj())
                    logger.info(
                        f"Loaded quantitative metric '{obj.__name__}' from {abs_path}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Could not instantiate quantitative metric '{obj.__name__}' "
                        f"from {abs_path}: {e}. "
                        f"Make sure the class can be instantiated with no arguments."
                    ) from e
            elif issubclass(obj, QualitativeMetric) and obj is not QualitativeMetric:
                try:
                    qual_metrics.append(obj())
                    logger.info(
                        f"Loaded qualitative metric '{obj.__name__}' from {abs_path}"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Could not instantiate qualitative metric '{obj.__name__}' "
                        f"from {abs_path}: {e}. "
                        f"Make sure the class can be instantiated with no arguments."
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
        scenarios: Optional in-memory scenarios for HTML report context.
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

    llm = LLM(
        model=settings.model,
        provider=settings.provider,
    )
    custom_metrics, custom_qualitative_metrics = _load_custom_metrics(
        settings.custom_metrics_file_paths
    )

    params = EvaluationParams(
        output_dir=settings.output_dir,
        num_workers=settings.num_workers,
        custom_metrics=custom_metrics,
        custom_qualitative_metrics=custom_qualitative_metrics,
        metrics_to_run=settings.metrics_to_run or None,
    )

    evaluator = Evaluator(
        params=params,
        llm=llm,
    )
    evaluator_output = evaluator.evaluate(simulation, on_progress=on_progress)
    evaluator.display_evaluation_summary()
    evaluator.save_results()

    if settings.generate_html_report:
        from arksim.scenario import Scenarios
        from arksim.utils.html_report.generate_html_report import (
            HtmlReportParams,
            generate_html_report,
        )

        html_output_path = os.path.join(settings.output_dir, "final_report.html")

        logger.info("Generating HTML report...")

        if scenarios is None and settings.scenario_file_path:
            try:
                scenarios = Scenarios.load(settings.scenario_file_path)
            except Exception:
                logger.warning(
                    "Could not load scenarios; scenarios will be empty in report"
                )
        all_custom = list(custom_metrics) + list(custom_qualitative_metrics)
        metric_descriptions = {
            m.name: m.description for m in all_custom if m.description
        }
        metric_ranges = {m.name: tuple(m.score_range) for m in custom_metrics}
        qual_label_colors = {
            m.name: m.label_colors for m in custom_qualitative_metrics if m.label_colors
        }
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
