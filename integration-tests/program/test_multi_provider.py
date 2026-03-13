# SPDX-License-Identifier: Apache-2.0

"""Integration tests for arksim with multiple LLM providers."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest
import yaml

from .conftest import (
    ANTHROPIC_MODEL,
    GOOGLE_MODEL,
    OPENAI_MODEL,
    requires_anthropic,
    requires_google,
    requires_openai,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


def _run_simulation_with_provider(
    minimal_scenarios: str,
    agent_config: dict,
    output_dir: str,
    provider: str,
    model: str,
) -> object:
    """Helper: run simulation with a specific LLM provider."""
    from arksim.simulation_engine import SimulationInput, run_simulation

    sim_output = os.path.join(output_dir, "simulation", "simulation.json")
    simulation_input = SimulationInput.model_validate(
        {
            "agent_config": agent_config,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": provider,
            "model": model,
            "output_file_path": sim_output,
        }
    )
    return asyncio.run(run_simulation(simulation_input))


class TestSimulateWithAnthropic:
    """Test simulation with Anthropic as the simulated user LLM."""

    @requires_anthropic
    @requires_openai
    def test_simulate_anthropic(
        self, minimal_scenarios: str, agent_config_openai: dict, tmp_output_dir: str
    ) -> None:
        simulation = _run_simulation_with_provider(
            minimal_scenarios,
            agent_config_openai,
            tmp_output_dir,
            provider="anthropic",
            model=ANTHROPIC_MODEL,
        )
        assert len(simulation.conversations) == 3
        for convo in simulation.conversations:
            assert len(convo.conversation_history) >= 2
            for msg in convo.conversation_history:
                assert msg.content


class TestSimulateWithGoogle:
    """Test simulation with Google as the simulated user LLM."""

    @requires_google
    @requires_openai
    def test_simulate_google(
        self, minimal_scenarios: str, agent_config_openai: dict, tmp_output_dir: str
    ) -> None:
        simulation = _run_simulation_with_provider(
            minimal_scenarios,
            agent_config_openai,
            tmp_output_dir,
            provider="google",
            model=GOOGLE_MODEL,
        )
        assert len(simulation.conversations) == 3
        for convo in simulation.conversations:
            assert len(convo.conversation_history) >= 2
            for msg in convo.conversation_history:
                assert msg.content


class TestEvaluateWithAnthropic:
    """Test evaluation with Anthropic LLM."""

    @requires_anthropic
    @requires_openai
    def test_evaluate_anthropic(
        self, minimal_scenarios: str, agent_config_openai: dict, tmp_output_dir: str
    ) -> None:
        from arksim.evaluator import EvaluationParams, Evaluator
        from arksim.llms.chat import LLM

        # Simulate with OpenAI first
        simulation = _run_simulation_with_provider(
            minimal_scenarios,
            agent_config_openai,
            tmp_output_dir,
            provider="openai",
            model=OPENAI_MODEL,
        )

        # Evaluate with Anthropic
        llm = LLM(
            model=ANTHROPIC_MODEL,
            provider="anthropic",
        )
        eval_params = EvaluationParams(
            output_dir=os.path.join(tmp_output_dir, "evaluation"),
            num_workers=20,
            metrics_to_run=["helpfulness", "coherence"],
        )
        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)

        assert len(evaluation.conversations) == 3
        for convo in evaluation.conversations:
            assert 0.0 <= convo.overall_agent_score <= 1.0


class TestEvaluateWithGoogle:
    """Test evaluation with Google LLM."""

    @requires_google
    @requires_openai
    def test_evaluate_google(
        self, minimal_scenarios: str, agent_config_openai: dict, tmp_output_dir: str
    ) -> None:
        from arksim.evaluator import EvaluationParams, Evaluator
        from arksim.llms.chat import LLM

        simulation = _run_simulation_with_provider(
            minimal_scenarios,
            agent_config_openai,
            tmp_output_dir,
            provider="openai",
            model=OPENAI_MODEL,
        )

        llm = LLM(model=GOOGLE_MODEL, provider="google")
        eval_params = EvaluationParams(
            output_dir=os.path.join(tmp_output_dir, "evaluation"),
            num_workers=20,
            metrics_to_run=["helpfulness", "coherence"],
        )
        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)

        assert len(evaluation.conversations) == 3
        for convo in evaluation.conversations:
            assert 0.0 <= convo.overall_agent_score <= 1.0


class TestCLISimulateEvaluateAnthropic:
    """Test CLI simulate-evaluate with Anthropic provider."""

    @requires_anthropic
    @requires_openai
    def test_cli_anthropic(
        self, minimal_scenarios: str, agent_config_openai: dict, tmp_output_dir: str
    ) -> None:
        sim_output = os.path.join(tmp_output_dir, "simulation", "simulation.json")
        eval_dir = os.path.join(tmp_output_dir, "evaluation")
        config = {
            "agent_config": agent_config_openai,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": "anthropic",
            "model": ANTHROPIC_MODEL,
            "output_file_path": sim_output,
            "output_dir": eval_dir,
            "metrics_to_run": ["helpfulness", "coherence"],
            "generate_html_report": False,
        }
        config_path = os.path.join(tmp_output_dir, "config_anthropic.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arksim.cli",
                "simulate-evaluate",
                config_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )
        assert result.returncode == 0, f"CLI anthropic failed: {result.stderr}"


class TestCLISimulateEvaluateGoogle:
    """Test CLI simulate-evaluate with Google provider."""

    @requires_google
    @requires_openai
    def test_cli_google(
        self, minimal_scenarios: str, agent_config_openai: dict, tmp_output_dir: str
    ) -> None:
        sim_output = os.path.join(tmp_output_dir, "simulation", "simulation.json")
        eval_dir = os.path.join(tmp_output_dir, "evaluation")
        config = {
            "agent_config": agent_config_openai,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": "google",
            "model": GOOGLE_MODEL,
            "output_file_path": sim_output,
            "output_dir": eval_dir,
            "metrics_to_run": ["helpfulness", "coherence"],
            "generate_html_report": False,
        }
        config_path = os.path.join(tmp_output_dir, "config_google.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arksim.cli",
                "simulate-evaluate",
                config_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )
        assert result.returncode == 0, f"CLI google failed: {result.stderr}"
