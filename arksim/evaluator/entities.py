# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import sys

from arksim.simulation_engine.tool_types import ToolCall

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from arksim.config.utils import resolve_model_paths
from arksim.constants import DEFAULT_MODEL, DEFAULT_PROVIDER
from arksim.utils.concurrency import validate_num_workers

from .base_metric import (
    ChatMessage,
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
)

_entities_logger = logging.getLogger(__name__)

_DEFAULT_METRICS_TO_RUN = [
    "faithfulness",
    "helpfulness",
    "coherence",
    "verbosity",
    "relevance",
    "goal_completion",
    "agent_behavior_failure",
]


class EvaluationInput(BaseModel):
    """Input configuration for the evaluator module."""

    scenario_file_path: str | None = Field(
        default=None,
        description="Path to the scenario file",
    )
    simulation_file_path: str | None = Field(
        default=None,
        description="Path to the simulation output file",
    )
    output_dir: str | None = Field(
        default="./results/evaluation",
        description="Output directory for evaluation results",
    )
    model: str = Field(default=DEFAULT_MODEL, description="LLM model for evaluation")
    provider: str | None = Field(default=DEFAULT_PROVIDER, description="LLM provider")
    num_workers: int | str = Field(
        default=50,
        description="Number of parallel workers (use 'auto' to default to 4)",
    )
    custom_metrics_file_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Paths to .py files defining custom QuantitativeMetric "
            "or QualitativeMetric subclasses"
        ),
    )
    metrics_to_run: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_METRICS_TO_RUN),
        description="Metrics to run during evaluation",
    )
    generate_html_report: bool = Field(
        default=True, description="Whether to generate an HTML report"
    )
    score_threshold: float | None = Field(
        default=None,
        description=(
            "Deprecated. Use numeric_thresholds with key 'overall_score' instead. "
            "Threshold for per-conversation final scores (0.0–1.0). "
            "If any score < threshold, exit with non-zero code."
        ),
    )
    numeric_thresholds: dict[str, float] | None = Field(
        default=None,
        description=(
            "Per-metric pass/fail thresholds on each metric's native scale. "
            "Keys are metric names (e.g. 'faithfulness', 'helpfulness'). "
            "Built-in turn-level metrics use a 1–5 scale; the mean across all "
            "turns per conversation is compared against the threshold. "
            "'goal_completion' is stored as 0–1 and compared directly. "
            "'overall_score' checks the per-conversation overall_agent_score (0–1)."
        ),
    )
    qualitative_failure_labels: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Hard-gate failure labels for qualitative metrics. "
            "Keys are metric names, values are lists of labels that trigger failure "
            "(e.g. 'prohibited_statements': ['violated']). "
            "Any evaluated turn whose label appears in the list fails the run; "
            "turns where the metric did not run are skipped."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_score_threshold(cls, data: object) -> object:
        """Migrate deprecated score_threshold into numeric_thresholds['overall_score']."""
        if not isinstance(data, dict):
            return data
        score_threshold = data.get("score_threshold")
        if score_threshold is not None:
            _entities_logger.warning(
                "'score_threshold' is deprecated. "
                "Use 'numeric_thresholds: {overall_score: <value>}' instead."
            )
            numeric_thresholds = dict(data.get("numeric_thresholds") or {})
            if "overall_score" not in numeric_thresholds:
                numeric_thresholds["overall_score"] = score_threshold
            data = {
                **data,
                "numeric_thresholds": numeric_thresholds,
                "score_threshold": None,
            }
        return data

    @model_validator(mode="after")
    def validate_evaluation_input(self, info: ValidationInfo) -> Self:
        """Validate evaluation input fields."""
        validate_num_workers(self.num_workers)

        # Paths from config.yaml are resolved relative to the config file's
        # directory. Paths set via CLI are left as-is (cwd-relative).
        config_path = info.context and info.context.get("config_path")
        if config_path:
            resolve_model_paths(
                self,
                path_attrs=(
                    "scenario_file_path",
                    "simulation_file_path",
                    "output_dir",
                ),
                list_path_attrs=("custom_metrics_file_paths",),
                config_dir=os.path.dirname(config_path),
                cli_overrides=(info.context and info.context.get("cli_overrides"))
                or set(),
            )

        return self


class EvaluationParams(BaseModel):
    """Parameters for the evaluation engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    output_dir: str
    agent_name: str = "Agent"
    code_file_path: str | None = None
    entry_function: str | None = None
    num_workers: int | str = Field(default=50)
    custom_metrics: list[QuantitativeMetric] = Field(default_factory=list)
    custom_qualitative_metrics: list[QualitativeMetric] = Field(default_factory=list)
    metrics_to_run: list[str] | None = None


class TurnItem(BaseModel):
    chat_id: str
    turn_id: int
    current_turn: list[ChatMessage]  # user + assistant pair for this turn
    conversation_history: list[ChatMessage]  # full conversation up to this turn
    system_prompt: str
    knowledge: list[str]
    profile: str
    user_goal: str
    tool_calls: list[ToolCall] | None = None


class ConvoItem(BaseModel):
    chat_id: str
    chat_history: list[ChatMessage]
    system_prompt: str
    knowledge: list[str]
    profile: str
    user_goal: str
    turns: int


class TurnEvaluation(BaseModel):
    """Per-turn evaluation result."""

    turn_id: int
    scores: list[QuantResult]  # only metrics that ran
    turn_score: float  # mean of score.value on 1–5; -1 if no metrics ran
    turn_behavior_failure: str
    turn_behavior_failure_reason: str
    qual_scores: list[QualResult] = Field(default_factory=list)
    unique_error_ids: list[str] = Field(default_factory=list)


class Occurrence(BaseModel):
    """Location of a unique error occurrence."""

    conversation_id: str
    turn_id: int


class UniqueError(BaseModel):
    """A unique agent behaviour error detected across conversations."""

    unique_error_id: str  # UUID4 string
    behavior_failure_category: str
    unique_error_description: str
    severity: str = "medium"
    occurrences: list[Occurrence]


class ConversationEvaluation(BaseModel):
    """Per-conversation evaluation result."""

    conversation_id: str
    goal_completion_score: float  # 0–1 (normalized from 1–5)
    goal_completion_reason: str
    turn_success_ratio: float  # 0–1
    overall_agent_score: float  # 0–1
    evaluation_status: str
    turn_scores: list[TurnEvaluation]


class ErrorScenarioGroup(BaseModel):
    """Maps a unique error to the scenarios that triggered it."""

    error_index: int
    unique_error_id: str
    error_description: str
    severity: str
    scenario_ids: list[str]


class Evaluation(BaseModel):
    """Top-level evaluation output file."""

    schema_version: str
    generated_at: str  # ISO 8601 UTC
    evaluator_version: str
    evaluation_id: str
    simulation_id: str
    conversations: list[ConversationEvaluation]
    unique_errors: list[UniqueError]
    error_scenario_groups: list[ErrorScenarioGroup] = Field(default_factory=list)
