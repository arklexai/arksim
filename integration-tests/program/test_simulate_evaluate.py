# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the full simulate -> evaluate pipeline
via Python API and CLI commands.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest
import yaml

from .conftest import requires_openai

pytestmark = [pytest.mark.integration, requires_openai]


class TestFullPipelinePythonAPI:
    """Test full simulate -> evaluate via Python API."""

    def test_simulate_then_evaluate(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.config import AgentConfig
        from arksim.evaluator import (
            Evaluation,
            EvaluationParams,
            Evaluator,
        )
        from arksim.llms.chat import LLM
        from arksim.scenario import Scenarios
        from arksim.simulation_engine import (
            Simulation,
            SimulationParams,
            Simulator,
        )

        # Step 1: Simulate
        scenarios = Scenarios.load(minimal_scenarios)
        agent_config = AgentConfig.model_validate(agent_config_openai)
        llm = LLM(model="gpt-5.1", provider="openai")

        sim_params = SimulationParams(
            num_convos_per_scenario=1,
            max_turns=5,
            num_workers=20,
            output_file_path=os.path.join(
                tmp_output_dir, "simulation", "simulation.json"
            ),
        )
        simulator = Simulator(
            agent_config=agent_config,
            simulator_params=sim_params,
            llm=llm,
        )
        simulation = asyncio.run(simulator.simulate(scenarios))
        asyncio.run(simulator.save())

        assert isinstance(simulation, Simulation)
        assert len(simulation.conversations) == 3

        # Step 2: Evaluate
        eval_params = EvaluationParams(
            output_dir=os.path.join(tmp_output_dir, "evaluation"),
            num_workers=20,
            metrics_to_run=[
                "helpfulness",
                "coherence",
                "goal_completion",
                "agent_behavior_failure",
            ],
        )
        evaluator = Evaluator(eval_params, llm=llm)
        evaluation = evaluator.evaluate(simulation)
        evaluator.save_results()

        assert isinstance(evaluation, Evaluation)
        assert len(evaluation.conversations) == 3
        assert evaluation.simulation_id == simulation.simulation_id

        for convo in evaluation.conversations:
            assert 0.0 <= convo.overall_agent_score <= 1.0


class TestCLISimulateCommand:
    """Test arksim simulate CLI command."""

    def test_cli_simulate(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        sim_output = os.path.join(tmp_output_dir, "simulation", "simulation.json")
        config = {
            "agent_config": agent_config_openai,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": "openai",
            "model": "gpt-5.1",
            "output_file_path": sim_output,
        }
        config_path = os.path.join(tmp_output_dir, "config_simulate.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        result = subprocess.run(
            [sys.executable, "-m", "arksim.cli", "simulate", config_path],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )
        assert result.returncode == 0, (
            f"CLI simulate failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert os.path.isfile(sim_output)

        with open(sim_output) as f:
            data = json.load(f)
        assert len(data["conversations"]) == 3


class TestCLIEvaluateCommand:
    """Test arksim evaluate CLI command."""

    def test_cli_evaluate(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        # First simulate to produce simulation.json
        sim_output = os.path.join(tmp_output_dir, "simulation", "simulation.json")
        sim_config = {
            "agent_config": agent_config_openai,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": "openai",
            "model": "gpt-5.1",
            "output_file_path": sim_output,
        }
        sim_config_path = os.path.join(tmp_output_dir, "config_simulate.yaml")
        with open(sim_config_path, "w") as f:
            yaml.dump(sim_config, f, default_flow_style=False)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arksim.cli",
                "simulate",
                sim_config_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )
        assert result.returncode == 0, f"Simulate step failed: {result.stderr}"

        # Now evaluate
        eval_dir = os.path.join(tmp_output_dir, "evaluation")
        eval_config = {
            "simulation_file_path": sim_output,
            "output_dir": eval_dir,
            "provider": "openai",
            "model": "gpt-5.1",
            "num_workers": 20,
            "metrics_to_run": ["helpfulness", "coherence"],
            "generate_html_report": True,
        }
        eval_config_path = os.path.join(tmp_output_dir, "config_evaluate.yaml")
        with open(eval_config_path, "w") as f:
            yaml.dump(eval_config, f, default_flow_style=False)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arksim.cli",
                "evaluate",
                eval_config_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )
        assert result.returncode == 0, (
            f"CLI evaluate failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        eval_file = os.path.join(eval_dir, "evaluation.json")
        assert os.path.isfile(eval_file)

        with open(eval_file) as f:
            data = json.load(f)
        assert len(data["conversations"]) == 3


class TestCLISimulateEvaluateCommand:
    """Test arksim simulate-evaluate CLI command."""

    def test_cli_simulate_evaluate(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        sim_output = os.path.join(tmp_output_dir, "simulation", "simulation.json")
        eval_dir = os.path.join(tmp_output_dir, "evaluation")
        config = {
            "agent_config": agent_config_openai,
            "scenario_file_path": minimal_scenarios,
            "num_conversations_per_scenario": 1,
            "max_turns": 5,
            "num_workers": 20,
            "provider": "openai",
            "model": "gpt-5.1",
            "output_file_path": sim_output,
            "output_dir": eval_dir,
            "metrics_to_run": [
                "helpfulness",
                "coherence",
                "goal_completion",
                "agent_behavior_failure",
            ],
            "generate_html_report": True,
        }
        config_path = os.path.join(tmp_output_dir, "config.yaml")
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
        assert result.returncode == 0, (
            f"CLI simulate-evaluate failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        assert os.path.isfile(sim_output)
        eval_file = os.path.join(eval_dir, "evaluation.json")
        assert os.path.isfile(eval_file)

        with open(eval_file) as f:
            data = json.load(f)
        assert len(data["conversations"]) == 3
        for convo in data["conversations"]:
            assert 0.0 <= convo["overall_agent_score"] <= 1.0
