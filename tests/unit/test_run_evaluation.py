# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluator.run_evaluation."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arksim.evaluator.entities import (
    ConversationEvaluation,
    Evaluation,
    EvaluationInput,
)
from arksim.evaluator.evaluator import run_evaluation
from arksim.scenario.entities import KnowledgeItem, Scenario, Scenarios
from arksim.simulation_engine.entities import (
    Conversation,
    Message,
    SimulatedUserPrompt,
    Simulation,
)


def _simulation() -> Simulation:
    return Simulation(
        schema_version="v1.1",
        simulator_version="v1",
        simulation_id="sim-1",
        conversations=[
            Conversation(
                conversation_id="conv-1",
                scenario_id="sc-1",
                conversation_history=[
                    Message(turn_id=0, role="simulated_user", content="Hi"),
                    Message(turn_id=0, role="assistant", content="Hello!"),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="tmpl",
                    variables={
                        "scenario.goal": "help",
                        "scenario.knowledge": [],
                    },
                ),
            )
        ],
    )


def _scenarios() -> Scenarios:
    return Scenarios(
        schema_version="v1",
        scenarios=[
            Scenario(
                scenario_id="sc-1",
                user_id="u-1",
                goal="help",
                agent_context="assistant",
                knowledge=[KnowledgeItem(content="kb")],
                user_profile="profile",
                origin={},
            )
        ],
    )


def _evaluation_output() -> Evaluation:
    return Evaluation(
        schema_version="v1",
        generated_at="2026-01-01T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="eval-1",
        simulation_id="sim-1",
        conversations=[
            ConversationEvaluation(
                conversation_id="conv-1",
                goal_completion_score=1.0,
                goal_completion_reason="done",
                turn_success_ratio=1.0,
                overall_agent_score=1.0,
                evaluation_status="Done",
                turn_scores=[],
            )
        ],
        unique_errors=[],
    )


def test_run_evaluation_uses_in_memory_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = EvaluationInput(
        simulation_file_path=None,
        output_dir="/tmp/eval_test",
        generate_html_report=False,
    )
    simulation = _simulation()
    evaluation_output = _evaluation_output()

    monkeypatch.setattr(
        "arksim.evaluator.evaluator.load_json_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("should not read file")),
    )
    monkeypatch.setattr("arksim.evaluator.evaluator.LLM", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(
        "arksim.evaluator.evaluator._load_custom_metrics", lambda _paths: ([], [])
    )

    class FakeEvaluator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.chat_id_to_label = {"conv-1": "Conversation 1"}

        def evaluate(
            self, simulation_obj: Simulation, on_progress: object = None
        ) -> Evaluation:
            assert simulation_obj is simulation
            return evaluation_output

        def display_evaluation_summary(self) -> None:
            return None

        def save_results(self) -> None:
            return None

    monkeypatch.setattr("arksim.evaluator.evaluator.Evaluator", FakeEvaluator)

    result = run_evaluation(settings, simulation=simulation)
    assert result == evaluation_output


def test_run_evaluation_requires_simulation_input() -> None:
    settings = EvaluationInput(
        simulation_file_path=None,
        output_dir="/tmp/eval_test",
        generate_html_report=False,
    )
    with pytest.raises(ValueError, match="Either pass Simulation object"):
        run_evaluation(settings)


def test_run_evaluation_passes_in_memory_scenarios_to_html_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = EvaluationInput(
        simulation_file_path=None,
        scenario_file_path="./scenarios.json",
        output_dir="/tmp/eval_test",
        generate_html_report=True,
    )
    simulation = _simulation()
    scenarios = _scenarios()
    evaluation_output = _evaluation_output()
    html_calls: list[object] = []

    monkeypatch.setattr("arksim.evaluator.evaluator.LLM", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(
        "arksim.evaluator.evaluator._load_custom_metrics", lambda _paths: ([], [])
    )
    monkeypatch.setattr(
        "arksim.scenario.Scenarios.load",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("Scenarios.load should not be called")
        ),
    )

    class FakeEvaluator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.chat_id_to_label = {"conv-1": "Conversation 1"}

        def evaluate(
            self, simulation_obj: Simulation, on_progress: object = None
        ) -> Evaluation:
            assert simulation_obj is simulation
            return evaluation_output

        def display_evaluation_summary(self) -> None:
            return None

        def save_results(self) -> None:
            return None

    monkeypatch.setattr("arksim.evaluator.evaluator.Evaluator", FakeEvaluator)
    report_module = importlib.import_module(
        "arksim.utils.html_report.generate_html_report"
    )
    sentinel_tag = "patched-html-params"

    def fake_html_report_params(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(_sentinel=sentinel_tag, **kwargs)

    monkeypatch.setattr(
        report_module,
        "HtmlReportParams",
        fake_html_report_params,
    )
    monkeypatch.setattr(
        report_module,
        "generate_html_report",
        lambda params: html_calls.append(params),
    )

    run_evaluation(settings, simulation=simulation, scenarios=scenarios)

    assert len(html_calls) == 1
    assert getattr(html_calls[0], "_sentinel", None) == sentinel_tag
    assert html_calls[0].scenarios is scenarios
