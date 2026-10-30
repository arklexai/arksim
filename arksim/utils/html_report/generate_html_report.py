# SPDX-License-Identifier: Apache-2.0
"""
Generate a standalone HTML report by embedding data into the template.
This allows the HTML to be opened directly without needing an HTTP server.

This module is called from the evaluator with data objects.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal  # noqa: UP035

from pydantic import BaseModel, Field

from arksim.evaluator.builtin_metrics import AgentBehaviorFailureMetric
from arksim.evaluator.entities import Evaluation
from arksim.evaluator.utils.constants import score_label
from arksim.evaluator.utils.enums import AgentMetrics, EvaluationOutcomes
from arksim.scenario import Scenarios
from arksim.simulation_engine import Simulation

# ---------------------------------------------------------------------------
# Built-in metric descriptions (merged with user-supplied metric_descriptions)
# ---------------------------------------------------------------------------

_BUILTIN_METRIC_DESCRIPTIONS: dict[str, str] = {
    "helpfulness": "How effectively the response addresses the user's needs.",
    "coherence": "Logical flow and consistency of the response.",
    "verbosity": "How concise the response is. Higher is more concise.",
    "relevance": "How well the response addresses the user's query.",
    "faithfulness": "How accurately the response reflects the provided knowledge.",
    "goal_completion": (
        "Whether the simulated user got what they asked for. A low score is "
        "expected when the agent correctly declines a request."
    ),
    AgentMetrics.AGENT_BEHAVIOR_FAILURE.value: AgentBehaviorFailureMetric.DESCRIPTION,
}

# ---------------------------------------------------------------------------
# Built-in label colors (merged with user-supplied qual_label_colors)
# ---------------------------------------------------------------------------

_BUILTIN_QUAL_LABEL_COLORS: dict[str, dict[str, str]] = {
    AgentMetrics.AGENT_BEHAVIOR_FAILURE.value: AgentBehaviorFailureMetric.DEFAULT_LABEL_COLORS,
}

_GOAL_COMPLETION = "goal_completion"
_LOWEST_SHOWN = 3

# Behavior failure labels that are not a verdict on the agent's reply.
_ABF_EXCLUDED = {
    EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
    EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
    EvaluationOutcomes.AGENT_API_ERROR.value,
}

# ---------------------------------------------------------------------------
# JSON data models — define the exact shape passed to the HTML template
# ---------------------------------------------------------------------------


class MetricLowest(BaseModel):
    """One of the lowest-scoring conversations on a metric."""

    chat_id: str
    value: float
    #: Turn metrics only: the conversation's lowest turn.
    min_turn_id: int | None = None
    min_value: float | None = None


class MetricSummary(BaseModel):
    """Run-wide summary of one quantitative metric."""

    key: str
    name: str
    scope: Literal["conversation", "turn"]
    score_min: float
    score_max: float
    value: float  # mean over conversations
    median: float
    band: str
    lowest: list[MetricLowest] = Field(default_factory=list)


class QualLabelConversation(BaseModel):
    """A conversation that got a label, with the turns that got it."""

    chat_id: str
    turn_ids: list[int] = Field(default_factory=list)


class QualLabelSummary(BaseModel):
    """How often one label of a qualitative metric was given."""

    label: str
    count: int
    percentage: float
    color: str | None = None
    conversations: list[QualLabelConversation] = Field(default_factory=list)


class QualMetricSummary(BaseModel):
    """Run-wide label distribution of one qualitative metric."""

    key: str
    name: str
    scope: Literal["conversation", "turn"]
    labels: list[QualLabelSummary]


class EvaluationPromptSummary(BaseModel):
    """Compact representation of a prompt category for the report."""

    category: str
    description: str
    prompts: list[dict[str, str]]


class TokenUsageSummary(BaseModel):
    """Token usage data for the report."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0


class ReportSummary(BaseModel):
    """Top-level summary object embedded as FINAL_REPORT_DATA in the template."""

    total_conversations: int
    total_turns: int
    average_turns_per_conversation: float
    metrics: list[MetricSummary] = Field(default_factory=list)
    qual_metrics: list[QualMetricSummary] = Field(default_factory=list)
    evaluation_model: str | None = None
    evaluation_provider: str | None = None
    generated_at: str | None = None
    evaluation_prompts: list[EvaluationPromptSummary] | None = None
    simulation_token_usage: TokenUsageSummary | None = None
    evaluation_token_usage: TokenUsageSummary | None = None


class ConvoRow(BaseModel):
    """One row in the CONVOS_DATA array — one entry per conversation."""

    chat_id: str
    scenario_id: str
    goal: str
    user_profile: str = ""
    goal_completion_score: float
    knowledge: list[str] = Field(default_factory=list)
    goal_completion_reason: str
    scores: list[dict[str, Any]] = Field(default_factory=list)
    qual_scores: list[dict[str, Any]] = Field(default_factory=list)
    result: str = "no_assertions"
    assertion_results: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = 0
    checks_passed: int = 0
    checks_total: int = 0
    #: One cell per metric for the Conversations table, keyed by metric key.
    metric_results: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TurnRow(BaseModel):
    """One row in the TURNS_DATA array — one entry per turn."""

    chat_id: str
    turn_id: int
    agent_behavior_failure_label: str
    agent_behavior_failure_reason: str
    scores: list[dict[str, Any]] = Field(default_factory=list)
    qual_scores: list[dict[str, Any]] = Field(default_factory=list)


class ErrorRow(BaseModel):
    """One row in the UNIQUE_ERRORS_DATA array — one failure pattern."""

    unique_error: str
    agent_behavior_failure_category: str
    severity: str
    source: str = "behavior"
    occurrences: dict[str, list[str]]  # {conv_id: ["turn_0", "turn_1", ...]}
    scenario_ids: list[str] = Field(default_factory=list)


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
    qual_label_colors: dict[str, dict[str, str]] = Field(default_factory=dict)
    evaluation_model: str | None = None
    evaluation_provider: str | None = None


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _metric_name(key: str) -> str:
    if key == _GOAL_COMPLETION:
        return "User Goal Completion"
    return key.replace("_", " ").title()


def _band(value: float, score_min: float, score_max: float) -> str:
    span = score_max - score_min
    return score_label(1 + 4 * (value - score_min) / span if span > 0 else 1)


def _collect_quant(
    evaluation: Evaluation,
) -> dict[str, tuple[str, dict[str, list[tuple[int | None, float]]]]]:
    """Per metric key: its scope and, per conversation, its (turn_id, value) scores."""
    by_metric: dict[str, tuple[str, dict[str, list[tuple[int | None, float]]]]] = {}

    def add(key: str, scope: str, chat_id: str, turn: int | None, v: float) -> None:
        if v < 0:  # not computed
            return
        _, convs = by_metric.setdefault(key, (scope, defaultdict(list)))
        convs[chat_id].append((turn, v))

    for conv in evaluation.conversations:
        cid = conv.conversation_id
        add(_GOAL_COMPLETION, "conversation", cid, None, conv.goal_completion_score)
        for s in conv.convo_scores:
            add(s.name, "conversation", cid, None, s.value)
        for turn in conv.turn_scores:
            for s in turn.scores:
                add(s.name, "turn", cid, turn.turn_id, s.value)
    return by_metric


def _conv_cell(scores: list[tuple[int | None, float]]) -> dict[str, Any]:
    """A conversation's value on a metric: the mean, with its lowest turn."""
    cell: dict[str, Any] = {
        "type": "quantitative",
        "value": sum(v for _, v in scores) / len(scores),
    }
    turns = [(t, v) for t, v in scores if t is not None]
    if turns:
        t, v = min(turns, key=lambda tv: tv[1])
        cell["min_turn_id"], cell["min_value"] = t, v
    return cell


def _build_metric_summaries(
    quant: dict[str, tuple[str, dict[str, list[tuple[int | None, float]]]]],
    ranges: dict[str, tuple[float, float]],
) -> list[MetricSummary]:
    summaries = []
    for key, (scope, convs) in quant.items():
        if not convs:
            continue
        score_min, score_max = (
            (0.0, 1.0) if key == _GOAL_COMPLETION else ranges.get(key, (1.0, 5.0))
        )
        cells = {cid: _conv_cell(scores) for cid, scores in convs.items()}
        values = [c["value"] for c in cells.values()]
        mean = sum(values) / len(values)
        lowest = sorted(cells.items(), key=lambda kv: kv[1]["value"])[:_LOWEST_SHOWN]
        summaries.append(
            MetricSummary(
                key=key,
                name=_metric_name(key),
                scope=scope,
                score_min=score_min,
                score_max=score_max,
                value=mean,
                median=statistics.median(values),
                band=_band(mean, score_min, score_max),
                lowest=[
                    MetricLowest(
                        chat_id=cid,
                        value=c["value"],
                        min_turn_id=c.get("min_turn_id"),
                        min_value=c.get("min_value"),
                    )
                    for cid, c in lowest
                ],
            )
        )
    return summaries


def _collect_qual(
    evaluation: Evaluation,
) -> dict[str, tuple[str, dict[str, dict[str, list[int]]]]]:
    """Per qualitative metric key: its scope and, per label, the turns per conversation."""
    by_metric: dict[str, tuple[str, dict[str, dict[str, list[int]]]]] = {}

    def add(key: str, scope: str, label: str, chat_id: str, turn: int | None) -> None:
        _, labels = by_metric.setdefault(key, (scope, {}))
        turns = labels.setdefault(label, {}).setdefault(chat_id, [])
        if turn is not None:
            turns.append(turn)

    abf = AgentMetrics.AGENT_BEHAVIOR_FAILURE.value
    for conv in evaluation.conversations:
        cid = conv.conversation_id
        for turn in conv.turn_scores:
            if turn.turn_behavior_failure and turn.turn_behavior_failure not in (
                _ABF_EXCLUDED
            ):
                add(abf, "turn", turn.turn_behavior_failure, cid, turn.turn_id)
            for q in turn.qual_scores:
                add(q.name, "turn", q.value, cid, turn.turn_id)
        for q in conv.convo_qual_scores:
            add(q.name, "conversation", q.value, cid, None)
    return by_metric


def _build_qual_summaries(
    qual: dict[str, tuple[str, dict[str, dict[str, list[int]]]]],
    colors: dict[str, dict[str, str]],
) -> list[QualMetricSummary]:
    summaries = []
    for key, (scope, labels) in qual.items():
        counts = {
            label: sum(max(len(turns), 1) for turns in convs.values())
            for label, convs in labels.items()
        }
        total = sum(counts.values())
        summaries.append(
            QualMetricSummary(
                key=key,
                name=_metric_name(key),
                scope=scope,
                labels=[
                    QualLabelSummary(
                        label=label,
                        count=counts[label],
                        percentage=round(counts[label] / total * 100, 1),
                        color=colors.get(key, {}).get(label),
                        conversations=[
                            QualLabelConversation(chat_id=cid, turn_ids=turns)
                            for cid, turns in labels[label].items()
                        ],
                    )
                    for label in sorted(counts, key=lambda lb: -counts[lb])
                ],
            )
        )
    return summaries


def _usage_summary(usage: Any) -> TokenUsageSummary | None:  # noqa: ANN401
    if not usage:
        return None
    return TokenUsageSummary(
        total_input_tokens=usage.total_input_tokens,
        total_output_tokens=usage.total_output_tokens,
        total_cache_read_tokens=usage.total_cache_read_tokens,
        total_cache_creation_tokens=usage.total_cache_creation_tokens,
        total_reasoning_tokens=usage.total_reasoning_tokens,
        total_tokens=usage.total_tokens,
    )


def _build_final_report_data(
    evaluation: Evaluation,
    simulation: Simulation | None = None,
    metric_ranges: dict[str, tuple[float, float]] | None = None,
    qual_label_colors: dict[str, dict[str, str]] | None = None,
    evaluation_model: str | None = None,
    evaluation_provider: str | None = None,
) -> ReportSummary:
    """Build the summary report from an Evaluation.

    Args:
        evaluation: Evaluation result object.
        simulation: Optional Simulation object (for token usage).
        metric_ranges: Score range per custom metric; built-ins are 1-5.
        qual_label_colors: Bar color per label of each qualitative metric.
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
        metrics=_build_metric_summaries(
            _collect_quant(evaluation), metric_ranges or {}
        ),
        qual_metrics=_build_qual_summaries(
            _collect_qual(evaluation),
            {**_BUILTIN_QUAL_LABEL_COLORS, **(qual_label_colors or {})},
        ),
        evaluation_model=evaluation_model,
        evaluation_provider=evaluation_provider,
        generated_at=datetime.now(timezone.utc).isoformat(),
        evaluation_prompts=eval_prompts,
        simulation_token_usage=_usage_summary(simulation.usage if simulation else None),
        evaluation_token_usage=_usage_summary(evaluation.usage),
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
    quant = _collect_quant(evaluation)
    qual = _collect_qual(evaluation)

    rows = []
    for conv in evaluation.conversations:
        cid = conv.conversation_id
        sim_conv = sim_lookup.get(cid)
        scenario = (
            scenario_lookup.get(sim_conv.scenario_id)
            if sim_conv and sim_conv.scenario_id
            else None
        )
        metric_results = {
            key: _conv_cell(convs[cid])
            for key, (_, convs) in quant.items()
            if cid in convs
        }
        for key, (_, labels) in qual.items():
            got = [
                {"value": label, "turn_ids": convs[cid]}
                for label, convs in labels.items()
                if cid in convs
            ]
            if got:
                metric_results[key] = {"type": "qualitative", "labels": got}
        rows.append(
            ConvoRow(
                chat_id=cid,
                scenario_id=sim_conv.scenario_id if sim_conv else "",
                goal=scenario.goal if scenario else "",
                user_profile=scenario.user_profile if scenario else "",
                goal_completion_score=conv.goal_completion_score,
                knowledge=[k.content for k in scenario.knowledge]
                if scenario and scenario.knowledge
                else [],
                goal_completion_reason=conv.goal_completion_reason,
                scores=[s.model_dump() for s in conv.convo_scores],
                qual_scores=[q.model_dump() for q in conv.convo_qual_scores],
                result=conv.result,
                assertion_results=[r.model_dump() for r in conv.assertion_results],
                turn_count=len(conv.turn_scores),
                checks_passed=sum(r.status == "pass" for r in conv.assertion_results),
                checks_total=len(conv.assertion_results),
                metric_results=metric_results,
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


def _build_error_rows(evaluation: Evaluation) -> list[ErrorRow]:
    """Build one row per failure pattern (unique error).

    Args:
        evaluation: Evaluation result object.

    Returns:
        List of ErrorRow, one per unique error.
    """
    # Use error_scenario_mappings as single source for scenario IDs
    mapping_by_error = {
        m.unique_error_id: m.scenario_ids for m in evaluation.error_scenario_mappings
    }

    rows = []
    for error in evaluation.unique_errors:
        occurrences_dict: dict[str, list[str]] = {}
        for occ in error.occurrences:
            turns = occurrences_dict.setdefault(occ.conversation_id, [])
            if occ.turn_id is not None:
                turns.append(f"turn_{occ.turn_id}")

        rows.append(
            ErrorRow(
                unique_error=error.unique_error_description,
                agent_behavior_failure_category=error.behavior_failure_category,
                severity=error.severity,
                source=error.source,
                occurrences=occurrences_dict,
                scenario_ids=mapping_by_error.get(error.unique_error_id, []),
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
        simulation=params.simulation,
        metric_ranges=params.metric_ranges,
        qual_label_colors=params.qual_label_colors,
        evaluation_model=params.evaluation_model,
        evaluation_provider=params.evaluation_provider,
    )
    convo_rows = _build_convo_rows(
        params.evaluation, params.simulation, params.scenarios
    )
    if not convo_rows:
        raise ValueError("conversations is required (should be in evaluation)")

    turn_rows = _build_turn_rows(params.evaluation)
    error_rows = _build_error_rows(params.evaluation)
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
    all_qual_label_colors = {
        **_BUILTIN_QUAL_LABEL_COLORS,
        **params.qual_label_colors,
    }
    html = html.replace("{{QUAL_LABEL_COLORS}}", _safe_json(all_qual_label_colors))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
