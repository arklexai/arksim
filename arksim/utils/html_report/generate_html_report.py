# SPDX-License-Identifier: Apache-2.0
"""
Generate a standalone HTML report by embedding data into the template.
This allows the HTML to be opened directly without needing an HTTP server.

This module is called from the evaluator with data objects.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any  # noqa: UP035

from pydantic import BaseModel, Field

from arksim.evaluator.builtin_metrics import AgentBehaviorFailureMetric
from arksim.evaluator.entities import Evaluation
from arksim.evaluator.utils.enums import AgentMetrics, EvaluationOutcomes
from arksim.scenario import Scenarios
from arksim.simulation_engine import Simulation

# ---------------------------------------------------------------------------
# Built-in metric descriptions (merged with user-supplied metric_descriptions)
# ---------------------------------------------------------------------------

_BUILTIN_METRIC_DESCRIPTIONS: dict[str, str] = {
    AgentMetrics.AGENT_BEHAVIOR_FAILURE.value: AgentBehaviorFailureMetric.DESCRIPTION,
}

# ---------------------------------------------------------------------------
# JSON data models — define the exact shape passed to the HTML template
# ---------------------------------------------------------------------------


class CustomMetricInfo(BaseModel):
    """Score and scale info for a single custom metric."""

    value: float
    min: float
    max: float


class PerformanceMetrics(BaseModel):
    """Aggregate performance scores shown in the report summary card."""

    helpfulness: float
    coherence: float
    verbosity: float
    relevance: float
    faithfulness: float
    custom_metric_scores: dict[str, CustomMetricInfo] = Field(default_factory=dict)


class BehaviourFailure(BaseModel):
    """Occurrence count and percentage for a single failure label."""

    occurrences: int
    percentage: float


class EvaluationPromptSummary(BaseModel):
    """Compact representation of a prompt category for the report."""

    category: str
    description: str
    prompts: list[dict[str, str]]


class ReportSummary(BaseModel):
    """Top-level summary object embedded as FINAL_REPORT_DATA in the template."""

    total_conversations: int
    total_turns: int
    average_turns_per_conversation: float
    performance_metrics: PerformanceMetrics | None = None
    agent_behavior_failures: dict[str, BehaviourFailure] | None = None
    qual_metric_distributions: dict[str, dict[str, BehaviourFailure]] = Field(
        default_factory=dict
    )
    evaluation_model: str | None = None
    evaluation_provider: str | None = None
    evaluation_prompts: list[EvaluationPromptSummary] | None = None


class ConvoRow(BaseModel):
    """One row in the CONVOS_DATA array — one entry per conversation."""

    chat_id: str
    scenario_id: str
    goal: str
    user_profile: str = ""
    goal_completion_score: float
    final_score: float
    status: str
    knowledge: list[str] = Field(default_factory=list)
    goal_completion_reason: str


class TurnRow(BaseModel):
    """One row in the TURNS_DATA array — one entry per turn."""

    chat_id: str
    turn_id: int
    agent_behavior_failure_label: str
    agent_behavior_failure_reason: str
    scores: list[dict[str, Any]] = Field(default_factory=list)
    qual_scores: list[dict[str, Any]] = Field(default_factory=list)


class OccurrenceSnippet(BaseModel):
    """A single turn transcript snippet shown inside an error card."""

    conversation_label: str
    turn_number: int
    user_message: str
    assistant_message: str


class ErrorRow(BaseModel):
    """One row in the UNIQUE_ERRORS_DATA array — one entry per unique error."""

    unique_error: str
    agent_behavior_failure_category: str
    severity: str
    occurrences: dict[str, list[str]]  # {conv_id: ["turn_0", "turn_1", ...]}
    suggested_fix: str = ""
    best_module_fix_reasoning: str = ""
    occurrence_snippets: list[OccurrenceSnippet] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Input parameters
# ---------------------------------------------------------------------------


class HtmlReportParams(BaseModel):
    """Parameters for HTML report generation."""

    simulation: Simulation
    evaluation: Evaluation
    scenarios: Scenarios | None = None
    output_path: str
    chat_id_to_label: dict[str, str] = Field(default_factory=dict)
    metric_descriptions: dict[str, str] = Field(default_factory=dict)
    metric_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    evaluation_model: str | None = None
    evaluation_provider: str | None = None


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_final_report_data(
    evaluation: Evaluation,
    metric_ranges: dict[str, tuple[float, float]] | None = None,
    evaluation_model: str | None = None,
    evaluation_provider: str | None = None,
) -> ReportSummary:
    """Build the summary report from an Evaluation.

    Args:
        evaluation: Evaluation result object.
        evaluation_model: LLM model used for evaluation.
        evaluation_provider: LLM provider used for evaluation.

    Returns:
        ReportSummary for the FINAL_REPORT_DATA template slot.
    """
    total_conversations = len(evaluation.conversations)
    total_turns = sum(len(c.turn_scores) for c in evaluation.conversations)
    average_turns = (
        total_turns / total_conversations if total_conversations > 0 else 0.0
    )

    metric_scores: dict[str, list[float]] = defaultdict(list)
    failure_counts: Counter = Counter()
    qual_label_counts: dict[str, Counter] = defaultdict(Counter)
    for conv in evaluation.conversations:
        for turn in conv.turn_scores:
            for score in turn.scores:
                metric_scores[score.name].append(score.value)
            for qs in turn.qual_scores:
                qual_label_counts[qs.name][qs.value] += 1
            failure_counts[turn.turn_behavior_failure] += 1

    def safe_avg(values: list[float]) -> float:
        valid = [v for v in values if v != -1]
        return sum(valid) / len(valid) if valid else -1

    builtin_keys = {
        "helpfulness",
        "coherence",
        "verbosity",
        "relevance",
        "faithfulness",
    }

    ranges = metric_ranges or {}
    helpfulness_avg = safe_avg(metric_scores.get("helpfulness", []))
    performance_metrics = None
    if helpfulness_avg != -1:
        custom_metric_scores = {
            k.replace("_", " ").title(): CustomMetricInfo(
                value=round(safe_avg(v), 1),
                min=ranges.get(k, (1, 5))[0],
                max=ranges.get(k, (1, 5))[1],
            )
            for k, v in metric_scores.items()
            if k not in builtin_keys and safe_avg(v) != -1
        }
        performance_metrics = PerformanceMetrics(
            helpfulness=round(safe_avg(metric_scores.get("helpfulness", [])), 1),
            coherence=round(safe_avg(metric_scores.get("coherence", [])), 1),
            verbosity=round(safe_avg(metric_scores.get("verbosity", [])), 1),
            relevance=round(safe_avg(metric_scores.get("relevance", [])), 1),
            faithfulness=round(safe_avg(metric_scores.get("faithfulness", [])), 1),
            custom_metric_scores=custom_metric_scores,
        )

    excluded_types = {
        EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
        EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
        EvaluationOutcomes.AGENT_API_ERROR.value,
        EvaluationOutcomes.NO_FAILURE.value,
    }
    displayed_failures = {
        k: v for k, v in failure_counts.items() if k not in excluded_types and v > 0
    }
    total_displayed = sum(displayed_failures.values())
    agent_behavior_failures = None
    behaviour_failures_dict: dict[str, BehaviourFailure] = {}
    for failure_type, count in displayed_failures.items():
        pct = (count / total_displayed) * 100 if total_displayed > 0 else 0
        behaviour_failures_dict[failure_type] = BehaviourFailure(
            occurrences=count,
            percentage=round(pct, 1),
        )
    if behaviour_failures_dict:
        agent_behavior_failures = behaviour_failures_dict

    custom_qual_distributions: dict[str, dict[str, BehaviourFailure]] = {}
    for metric_name, label_counter in qual_label_counts.items():
        total = sum(label_counter.values())
        custom_qual_distributions[metric_name] = {
            label: BehaviourFailure(
                occurrences=count,
                percentage=round((count / total) * 100, 1),
            )
            for label, count in sorted(label_counter.items(), key=lambda x: -x[1])
        }

    _abf_excluded = {
        EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
        EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
        EvaluationOutcomes.AGENT_API_ERROR.value,
    }
    abf_counts = {k: v for k, v in failure_counts.items() if k not in _abf_excluded}

    # agent_behavior_failure is always shown first
    qual_metric_distributions: dict[str, dict[str, BehaviourFailure]] = {}
    if abf_counts:
        _abf_total = sum(abf_counts.values())
        qual_metric_distributions[AgentMetrics.AGENT_BEHAVIOR_FAILURE.value] = {
            label: BehaviourFailure(
                occurrences=count,
                percentage=round((count / _abf_total) * 100, 1),
            )
            for label, count in sorted(abf_counts.items(), key=lambda x: -x[1])
        }
    qual_metric_distributions.update(custom_qual_distributions)

    # Build evaluation prompts summary from the registry
    eval_prompts: list[EvaluationPromptSummary] | None = None
    try:
        from arksim.evaluator.prompt_registry import PROMPT_REGISTRY

        eval_prompts = [
            EvaluationPromptSummary(
                category=cat.category,
                description=cat.description,
                prompts=[{"name": p.name, "text": p.text} for p in cat.prompts],
            )
            for cat in PROMPT_REGISTRY
        ]
    except Exception:
        pass

    return ReportSummary(
        total_conversations=total_conversations,
        total_turns=total_turns,
        average_turns_per_conversation=round(average_turns, 2),
        performance_metrics=performance_metrics,
        agent_behavior_failures=agent_behavior_failures,
        qual_metric_distributions=qual_metric_distributions,
        evaluation_model=evaluation_model,
        evaluation_provider=evaluation_provider,
        evaluation_prompts=eval_prompts,
    )


def _build_convo_rows(
    evaluation: Evaluation,
    simulation: Simulation,
    scenarios: Scenarios | None,
) -> list[ConvoRow]:
    """Build per-conversation rows by joining evaluation, simulation, and scenario data.

    Args:
        evaluation: Evaluation result object.
        simulation: Simulation output object.
        scenarios: Scenario definitions (may be None).

    Returns:
        List of ConvoRow, one per conversation.
    """
    sim_lookup = {c.conversation_id: c for c in simulation.conversations}
    scenario_lookup = (
        {s.scenario_id: s for s in scenarios.scenarios} if scenarios else {}
    )

    rows = []
    for conv in evaluation.conversations:
        sim_conv = sim_lookup.get(conv.conversation_id)
        scenario = (
            scenario_lookup.get(sim_conv.scenario_id)
            if sim_conv and sim_conv.scenario_id
            else None
        )
        rows.append(
            ConvoRow(
                chat_id=conv.conversation_id,
                scenario_id=sim_conv.scenario_id if sim_conv else "",
                goal=scenario.goal if scenario else "",
                user_profile=scenario.user_profile if scenario else "",
                goal_completion_score=conv.goal_completion_score,
                final_score=conv.overall_agent_score,
                status=conv.evaluation_status,
                knowledge=[k.content for k in scenario.knowledge]
                if scenario and scenario.knowledge
                else [],
                goal_completion_reason=conv.goal_completion_reason,
            )
        )
    return rows


def _build_turn_rows(evaluation: Evaluation) -> list[TurnRow]:
    """Build per-turn rows from evaluation data.

    Args:
        evaluation: Evaluation result object.

    Returns:
        List of TurnRow, one per turn.
    """
    rows = []
    for conv in evaluation.conversations:
        for turn in conv.turn_scores:
            rows.append(
                TurnRow(
                    chat_id=conv.conversation_id,
                    turn_id=turn.turn_id,
                    agent_behavior_failure_label=turn.turn_behavior_failure,
                    agent_behavior_failure_reason=turn.turn_behavior_failure_reason,
                    scores=[s.model_dump() for s in turn.scores],
                    qual_scores=[q.model_dump() for q in turn.qual_scores],
                )
            )
    return rows


def _build_error_rows(
    evaluation: Evaluation,
    simulation: Simulation,
    chat_id_to_label: dict[str, str],
) -> list[ErrorRow]:
    """Build per-unique-error rows, joining with simulation for occurrence snippets.

    Args:
        evaluation: Evaluation result object.
        simulation: Simulation output (used to extract turn message snippets).
        chat_id_to_label: Mapping of conversation IDs to human-readable labels.

    Returns:
        List of ErrorRow, one per unique error.
    """
    sim_lookup = {c.conversation_id: c for c in simulation.conversations}

    rows = []
    for error in evaluation.unique_errors:
        occurrences_dict: dict[str, list[str]] = {}
        for occ in error.occurrences:
            if occ.conversation_id not in occurrences_dict:
                occurrences_dict[occ.conversation_id] = []
            occurrences_dict[occ.conversation_id].append(f"turn_{occ.turn_id}")

        snippets: list[OccurrenceSnippet] = []
        for occ in error.occurrences:
            sim_conv = sim_lookup.get(occ.conversation_id)
            if sim_conv is None:
                continue
            agent_turn_index = 0
            user_msg = ""
            agent_msg = ""
            for msg in sim_conv.conversation_history:
                is_user = msg.role in ("simulated_user", "user")
                if is_user:
                    if agent_turn_index == occ.turn_id:
                        user_msg = msg.content
                else:
                    if agent_turn_index == occ.turn_id:
                        agent_msg = msg.content
                    agent_turn_index += 1
                    if agent_turn_index > occ.turn_id:
                        break
            label = chat_id_to_label.get(occ.conversation_id, occ.conversation_id)
            snippets.append(
                OccurrenceSnippet(
                    conversation_label=label,
                    turn_number=occ.turn_id,
                    user_message=user_msg,
                    assistant_message=agent_msg,
                )
            )

        rows.append(
            ErrorRow(
                unique_error=error.unique_error_description,
                agent_behavior_failure_category=error.behavior_failure_category,
                severity=error.severity,
                occurrences=occurrences_dict,
                suggested_fix=getattr(error, "suggested_fix", ""),
                best_module_fix_reasoning=getattr(
                    error, "best_module_fix_reasoning", ""
                ),
                occurrence_snippets=snippets,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _safe_json(obj: object) -> str:
    """Serialise obj to a JSON string safe to embed in an HTML <script> block.

    Handles Pydantic models and lists of Pydantic models via model_dump().
    Escapes </script> to prevent early tag termination.

    Args:
        obj: Python object (or Pydantic model / list of models) to serialise.

    Returns:
        JSON string.
    """
    if isinstance(obj, BaseModel):
        data: Any = obj.model_dump()
    elif isinstance(obj, list) and obj and isinstance(obj[0], BaseModel):
        data = [item.model_dump() for item in obj]
    else:
        data = obj
    return json.dumps(data, indent=2, ensure_ascii=False).replace(
        "</script>", r"<\/script>"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_html_report(params: HtmlReportParams) -> Path:
    """Generate HTML report directly from Python objects.

    Args:
        params: HtmlReportParams containing all report generation parameters.

    Returns:
        Path to generated HTML file.
    """
    final_report_data = _build_final_report_data(
        params.evaluation,
        metric_ranges=params.metric_ranges,
        evaluation_model=params.evaluation_model,
        evaluation_provider=params.evaluation_provider,
    )
    convo_rows = _build_convo_rows(
        params.evaluation, params.simulation, params.scenarios
    )
    if not convo_rows:
        raise ValueError("conversations is required (should be in evaluation)")

    turn_rows = _build_turn_rows(params.evaluation)
    error_rows = _build_error_rows(
        params.evaluation, params.simulation, params.chat_id_to_label
    )
    simulate_data_dicts = [
        conv.model_dump() for conv in params.simulation.conversations
    ]

    template_path = Path(__file__).parent / "report_template.html"
    output_path = Path(params.output_path)

    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    html = template
    html = html.replace("{{FINAL_REPORT_DATA}}", _safe_json(final_report_data))
    html = html.replace("{{SIMULATE_DATA}}", _safe_json(simulate_data_dicts))
    html = html.replace("{{CONVOS_DATA}}", _safe_json(convo_rows))
    html = html.replace("{{TURNS_DATA}}", _safe_json(turn_rows))
    html = html.replace(
        "{{UNIQUE_ERRORS_DATA}}", _safe_json(error_rows) if error_rows else "[]"
    )
    html = html.replace("{{CHAT_ID_TO_LABEL}}", _safe_json(params.chat_id_to_label))
    all_metric_descriptions = {
        **_BUILTIN_METRIC_DESCRIPTIONS,
        **params.metric_descriptions,
    }
    html = html.replace("{{METRIC_DESCRIPTIONS}}", _safe_json(all_metric_descriptions))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
