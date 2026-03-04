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
    "AgentConfig",
    "ChatCompletionsConfig",
    "A2AConfig",
    "Evaluator",
    "EvaluationInput",
    "EvaluationParams",
    "QuantitativeMetric",
    "QualitativeMetric",
    "run_evaluation",
    "Scenario",
    "Scenarios",
    "Simulator",
    "SimulationInput",
    "SimulationParams",
    "run_simulation",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AgentConfig": (".config", "AgentConfig"),
    "ChatCompletionsConfig": (".config", "ChatCompletionsConfig"),
    "A2AConfig": (".config", "A2AConfig"),
    "Evaluator": (".evaluator", "Evaluator"),
    "EvaluationInput": (".evaluator", "EvaluationInput"),
    "EvaluationParams": (".evaluator", "EvaluationParams"),
    "QuantitativeMetric": (".evaluator", "QuantitativeMetric"),
    "QualitativeMetric": (".evaluator", "QualitativeMetric"),
    "run_evaluation": (".evaluator", "run_evaluation"),
    "Scenario": (".scenario", "Scenario"),
    "Scenarios": (".scenario", "Scenarios"),
    "Simulator": (".simulation_engine", "Simulator"),
    "SimulationInput": (".simulation_engine", "SimulationInput"),
    "SimulationParams": (".simulation_engine", "SimulationParams"),
    "run_simulation": (".simulation_engine", "run_simulation"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
