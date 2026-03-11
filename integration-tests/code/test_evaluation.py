# SPDX-License-Identifier: Apache-2.0

"""Integration tests for arksim evaluation with real API calls."""

from __future__ import annotations

import asyncio
import os

import pytest

from .conftest import requires_openai

pytestmark = [pytest.mark.integration, requires_openai]


def _run_simulation(
    minimal_scenarios: str, agent_config_openai: dict, output_dir: str
) -> object:
    """Helper: run simulation and return Simulation object."""
    from arksim.simulation_engine import SimulationInput, run_simulation

    sim_output_path = os.path.join(output_dir, "simulation", "simulation.json")
    simulation_input = SimulationInput.model_validate(
        {
            "agent_config": agent_config_openai,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": "openai",
            "model": "gpt-5.1",
            "output_file_path": sim_output_path,
        }
    )
    return asyncio.run(run_simulation(simulation_input))


class TestEvaluateAllBuiltinMetrics:
    """Test evaluation with all built-in metrics."""

    def test_evaluate_all_metrics(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.evaluator import (
            Evaluation,
            EvaluationParams,
            Evaluator,
        )
        from arksim.llms.chat import LLM

        simulation = _run_simulation(
            minimal_scenarios, agent_config_openai, tmp_output_dir
        )

        llm = LLM(model="gpt-5.1", provider="openai")
        eval_params = EvaluationParams(
            output_dir=os.path.join(tmp_output_dir, "evaluation"),
            num_workers=20,
            metrics_to_run=[
                "faithfulness",
                "helpfulness",
                "coherence",
                "verbosity",
                "relevance",
                "goal_completion",
                "agent_behavior_failure",
            ],
        )

        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)

        assert isinstance(evaluation, Evaluation)
        assert len(evaluation.conversations) == 3

        for convo in evaluation.conversations:
            assert 0.0 <= convo.overall_agent_score <= 1.0, (
                f"overall_agent_score {convo.overall_agent_score} out of range"
            )
            assert 0.0 <= convo.goal_completion_score <= 1.0
            assert 0.0 <= convo.turn_success_ratio <= 1.0
            assert len(convo.turn_scores) >= 1

            for turn in convo.turn_scores:
                assert turn.turn_score >= 0
                assert len(turn.scores) >= 1
                metric_names = {s.name for s in turn.scores}
                assert "helpfulness" in metric_names
                assert "coherence" in metric_names


class TestEvaluateSubsetMetrics:
    """Test evaluation with a subset of metrics."""

    def test_subset_metrics(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.evaluator import EvaluationParams, Evaluator
        from arksim.llms.chat import LLM

        simulation = _run_simulation(
            minimal_scenarios, agent_config_openai, tmp_output_dir
        )

        llm = LLM(model="gpt-5.1", provider="openai")
        eval_params = EvaluationParams(
            output_dir=os.path.join(tmp_output_dir, "evaluation"),
            num_workers=20,
            metrics_to_run=["helpfulness", "coherence"],
        )

        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)

        for convo in evaluation.conversations:
            for turn in convo.turn_scores:
                metric_names = {s.name for s in turn.scores}
                assert "helpfulness" in metric_names
                assert "coherence" in metric_names
                # These should not be present
                assert "verbosity" not in metric_names
                assert "relevance" not in metric_names


class TestEvaluateWithCustomMetrics:
    """Test evaluation with a custom quantitative metric."""

    def test_custom_metric(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.evaluator import (
            EvaluationParams,
            Evaluator,
            QuantitativeMetric,
            QuantResult,
            ScoreInput,
        )
        from arksim.llms.chat import LLM

        class ResponseLengthMetric(QuantitativeMetric):
            """Simple metric that scores based on response
            length.
            """

            def __init__(self) -> None:
                super().__init__(
                    name="response_length",
                    score_range=(1, 5),
                    description="Scores response length",
                )

            def score(self, score_input: ScoreInput) -> QuantResult:
                assistant_msgs = [
                    m for m in score_input.current_turn if m.role == "assistant"
                ]
                if not assistant_msgs:
                    return QuantResult(name=self.name, value=1.0)
                length = len(assistant_msgs[0].content)
                # Score: longer is better, up to 500 chars
                value = min(5.0, max(1.0, length / 100))
                return QuantResult(
                    name=self.name,
                    value=value,
                    reason=f"Response length: {length} chars",
                )

        simulation = _run_simulation(
            minimal_scenarios, agent_config_openai, tmp_output_dir
        )

        llm = LLM(model="gpt-5.1", provider="openai")
        eval_params = EvaluationParams(
            output_dir=os.path.join(tmp_output_dir, "evaluation"),
            num_workers=20,
            metrics_to_run=["helpfulness"],
            custom_metrics=[ResponseLengthMetric()],
        )

        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)

        for convo in evaluation.conversations:
            for turn in convo.turn_scores:
                metric_names = {s.name for s in turn.scores}
                assert "response_length" in metric_names


class TestEvaluationSavesResults:
    """Test that evaluation saves output files."""

    def test_saves_evaluation_json(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.evaluator import EvaluationParams, Evaluator
        from arksim.llms.chat import LLM

        simulation = _run_simulation(
            minimal_scenarios, agent_config_openai, tmp_output_dir
        )

        eval_dir = os.path.join(tmp_output_dir, "evaluation")
        llm = LLM(model="gpt-5.1", provider="openai")
        eval_params = EvaluationParams(
            output_dir=eval_dir,
            num_workers=20,
            metrics_to_run=["helpfulness", "coherence"],
        )

        evaluator = Evaluator(eval_params, llm=llm)
        evaluator.evaluate(simulation)
        evaluator.save_results()

        eval_file = os.path.join(eval_dir, "evaluation.json")
        assert os.path.isfile(eval_file)
        assert os.path.getsize(eval_file) > 0


class TestHtmlReportGeneration:
    """Test HTML report generation."""

    def test_generates_html_report(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.evaluator import EvaluationParams, Evaluator
        from arksim.llms.chat import LLM
        from arksim.scenario import Scenarios
        from arksim.utils.html_report.generate_html_report import (
            HtmlReportParams,
            generate_html_report,
        )

        simulation = _run_simulation(
            minimal_scenarios, agent_config_openai, tmp_output_dir
        )
        scenarios = Scenarios.load(minimal_scenarios)

        eval_dir = os.path.join(tmp_output_dir, "evaluation")
        llm = LLM(model="gpt-5.1", provider="openai")
        eval_params = EvaluationParams(
            output_dir=eval_dir,
            num_workers=20,
            metrics_to_run=["helpfulness", "coherence"],
        )

        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)
        evaluator.save_results()

        html_path = os.path.join(eval_dir, "final_report.html")
        report_params = HtmlReportParams(
            simulation=simulation,
            evaluation=evaluation,
            scenarios=scenarios,
            output_path=html_path,
            chat_id_to_label=evaluator.chat_id_to_label,
        )
        generate_html_report(report_params)

        assert os.path.isfile(html_path)
        assert os.path.getsize(html_path) > 0
