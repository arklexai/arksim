# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from arksim.config.utils import resolve_config_relative_path
from arksim.constants import DEFAULT_MODEL, DEFAULT_PROVIDER
from arksim.utils.concurrency import validate_num_workers

from .base_metric import (
    ChatMessage,
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
)

logger = logging.getLogger(__name__)

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
        default="./evaluation",
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
            "Threshold for per-conversation final scores. "
            "If any score < threshold, exit with non-zero code."
        ),
    )

    @model_validator(mode="after")
    def validate_evaluation_input(self) -> Self:
        """Validate evaluation input fields."""
        validate_num_workers(self.num_workers)

        # Paths from config.yaml are resolved relative to the config file's
        # directory. Paths set via CLI are left as-is (cwd-relative).
        config_path = info.context and info.context.get("config_path")
        cli_overrides = (info.context and info.context.get("cli_overrides")) or set()
        if config_path:
            config_dir = os.path.dirname(config_path)
            for attr in ("scenario_file_path", "simulation_file_path", "output_dir"):
                path = getattr(self, attr)
                if path:
                    resolved = resolve_config_relative_path(
                        path, config_dir, cli_overrides, attr
                    )
                    if resolved is not None and resolved != path:
                        logger.debug("%s resolved to: %s", attr, resolved)
                        setattr(self, attr, resolved)

            if self.custom_metrics_file_paths:
                resolved_list = []
                for p in self.custom_metrics_file_paths:
                    resolved = resolve_config_relative_path(
                        p, config_dir, cli_overrides, "custom_metrics_file_paths"
                    )
                    resolved_list.append(resolved if resolved is not None else p)
                if resolved_list != self.custom_metrics_file_paths:
                    logger.debug(
                        "custom_metrics_file_paths resolved to: %s", resolved_list
                    )
                    self.custom_metrics_file_paths = resolved_list

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


class Evaluation(BaseModel):
    """Top-level evaluation output file."""

    schema_version: str
    generated_at: str  # ISO 8601 UTC
    evaluator_version: str
    evaluation_id: str
    simulation_id: str
    conversations: list[ConversationEvaluation]
    unique_errors: list[UniqueError]
