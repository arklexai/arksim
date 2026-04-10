# SPDX-License-Identifier: Apache-2.0
"""
Evaluator package for agent and user evaluation.
"""

from __future__ import annotations

from arksim.scenario.entities import AssertionType, ToolCallsAssertion

from .base_metric import (
    ChatMessage,
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
    ScoreInput,
    format_chat_history,
)
from .entities import (
    ErrorScenarioMapping,
    Evaluation,
    EvaluationInput,
    EvaluationParams,
)
from .error_scenarios import build_error_scenario_data
from .evaluator import Evaluator, run_evaluation
from .thresholds import (
    check_numeric_thresholds,
    check_qualitative_failure_labels,
)
from .trajectory_matching import TrajectoryResult, match_trajectory

__all__ = [
    "ChatMessage",
    "Evaluation",
    "EvaluationInput",
    "QuantitativeMetric",
    "QualitativeMetric",
    "QualResult",
    "Evaluator",
    "EvaluationParams",
    "ErrorScenarioMapping",
    "ScoreInput",
    "QuantResult",
    "build_error_scenario_data",
    "check_numeric_thresholds",
    "check_qualitative_failure_labels",
    "format_chat_history",
    "run_evaluation",
    "TrajectoryResult",
    "match_trajectory",
    "AssertionType",
    "ToolCallsAssertion",
]
