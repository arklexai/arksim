# SPDX-License-Identifier: Apache-2.0
"""⛵️ ArkSim: know how your agent performs before it goes live."""

from __future__ import annotations

import importlib

try:
    from arksim._version import __version__
except ImportError:  # editable install or no build
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "A2AConfig",
    "A2AToolCaptureExtension",
    "AgentConfig",
    "AgentResponse",
    "ChatCompletionsConfig",
    "EvaluationInput",
    "EvaluationParams",
    "Evaluator",
    "QualitativeMetric",
    "QuantitativeMetric",
    "run_evaluation",
    "run_simulation",
    "Scenario",
    "Scenarios",
    "Simulator",
    "SimulationInput",
    "SimulationParams",
    "ToolCall",
    "ToolCallSource",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "A2AConfig": (".config", "A2AConfig"),
    "A2AToolCaptureExtension": (".simulation_engine", "A2AToolCaptureExtension"),
    "AgentConfig": (".config", "AgentConfig"),
    "AgentResponse": (".simulation_engine", "AgentResponse"),
    "ChatCompletionsConfig": (".config", "ChatCompletionsConfig"),
    "Evaluator": (".evaluator", "Evaluator"),
    "EvaluationInput": (".evaluator", "EvaluationInput"),
    "EvaluationParams": (".evaluator", "EvaluationParams"),
    "QualitativeMetric": (".evaluator", "QualitativeMetric"),
    "QuantitativeMetric": (".evaluator", "QuantitativeMetric"),
    "run_evaluation": (".evaluator", "run_evaluation"),
    "run_simulation": (".simulation_engine", "run_simulation"),
    "Scenario": (".scenario", "Scenario"),
    "Scenarios": (".scenario", "Scenarios"),
    "Simulator": (".simulation_engine", "Simulator"),
    "SimulationInput": (".simulation_engine", "SimulationInput"),
    "SimulationParams": (".simulation_engine", "SimulationParams"),
    "ToolCall": (".simulation_engine", "ToolCall"),
    "ToolCallSource": (".simulation_engine", "ToolCallSource"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
