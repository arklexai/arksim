# SPDX-License-Identifier: Apache-2.0
"""ArkSim quality gate pytest test.

Copy to tests/test_agent_quality.py in your repo.

Install:
    pip install arksim pytest pytest-asyncio
"""

import asyncio
import os

import pytest

# TODO: import your agent class — it must subclass BaseAgent.
#
# Option A — subclass BaseAgent directly in your agent module:
#
#   from arksim.simulation_engine.agent.base import BaseAgent
#
#   class MyAgent(BaseAgent):
#       async def get_chat_id(self) -> str:
#           return str(id(self))
#
#       async def execute(self, user_query: str, **kwargs) -> str:
#           # Your agent logic here
#           ...
#
# Option B — thin adapter around an existing agent:
#
#   class MyAgentAdapter(BaseAgent):
#       def __init__(self, agent_config):
#           super().__init__(agent_config)
#           self.agent = MyExistingAgent()
#
#       async def get_chat_id(self) -> str:
#           return str(id(self))
#
#       async def execute(self, user_query: str, **kwargs) -> str:
#           return await self.agent.process(user_query)
#
from my_agent import MyAgent  # TODO: replace with your agent class

# TODO: import your custom metrics — optional, remove if not needed.
#
# Numeric metrics must subclass NumericMetric:
#
#   from arksim.evaluator.metrics.base import NumericMetric
#
#   class MyNumericMetric(NumericMetric):
#       name = "my_numeric_metric"
#       ...
#
# Qualitative metrics must subclass QualitativeMetric:
#
#   from arksim.evaluator.metrics.base import QualitativeMetric
#
#   class MyQualitativeMetric(QualitativeMetric):
#       name = "my_qualitative_metric"
#       ...
#
from my_metrics import (  # TODO: replace with your metric classes
    MyNumericMetric,
    MyQualitativeMetric,
)

from arksim.config import AgentConfig, CustomConfig
from arksim.evaluator import (
    EvaluationParams,
    Evaluator,
    check_numeric_thresholds,
    check_qualitative_failure_labels,
    check_score_threshold,
)
from arksim.llms.chat import LLM
from arksim.scenario import Scenarios
from arksim.simulation_engine import SimulationParams, Simulator
from arksim.utils.html_report.generate_html_report import (
    HtmlReportParams,
    generate_html_report,
)

# ── Quality gate thresholds ────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.7

NUMERIC_THRESHOLDS = {
    "goal_completion": 0.8,
    "faithfulness": 3.5,
}

QUALITATIVE_FAILURE_LABELS = {
    "agent_behavior_failure": ["false information", "disobey user request"],
}


# ── Test ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_quality() -> None:
    results_dir = "./tests/arksim/results"
    simulation_output_file = os.path.join(results_dir, "simulation", "simulation.json")
    evaluation_output_dir = os.path.join(results_dir, "evaluation")

    model = "gpt-4o"
    provider = "openai"

    # ── Simulate ──────────────────────────────────────────────────────────────
    scenario_output = Scenarios.load("./tests/arksim/scenarios.json")
    agent_config = AgentConfig(
        agent_type="custom",
        agent_name=MyAgent.__name__,
        custom_config=CustomConfig(agent_class=MyAgent),
    )
    llm = LLM(model=model, provider=provider)

    simulation_params = SimulationParams(
        num_convos_per_scenario=3,
        max_turns=5,
        num_workers=10,
        output_file_path=simulation_output_file,
    )

    simulator = Simulator(
        agent_config=agent_config,
        simulator_params=simulation_params,
        llm=llm,
    )
    simulation_output = await simulator.simulate(scenario_output)
    await simulator.save()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator_params = EvaluationParams(
        output_dir=evaluation_output_dir,
        num_workers=10,
        custom_metrics=[
            MyNumericMetric(),  # TODO: replace with your numeric metric instances
        ],
        custom_qualitative_metrics=[
            MyQualitativeMetric(),  # TODO: replace with your qualitative metric instances
        ],
    )

    evaluator = Evaluator(evaluator_params, llm=llm)
    evaluator_output = await asyncio.to_thread(evaluator.evaluate, simulation_output)
    evaluator.display_evaluation_summary()
    evaluator.save_results()

    # ── Generate HTML report ───────────────────────────────────────────────────
    html_output_path = os.path.join(evaluation_output_dir, "final_report.html")
    report_params = HtmlReportParams(
        simulation=simulation_output,
        evaluation=evaluator_output,
        scenarios=scenario_output,
        output_path=html_output_path,
        chat_id_to_label=evaluator.chat_id_to_label,
    )
    await asyncio.to_thread(generate_html_report, report_params)

    # ── Assert quality gates ───────────────────────────────────────────────────
    # Overall score gate
    assert check_score_threshold(
        evaluator_output,
        score_threshold=SCORE_THRESHOLD,
    ), "Score threshold check failed — see arksim/results/evaluation for details"

    # Per-metric gate
    assert check_numeric_thresholds(
        evaluator_output,
        numeric_thresholds=NUMERIC_THRESHOLDS,
    ), "Numeric threshold check failed — see arksim/results/evaluation for details"

    # Qualitative gate
    assert check_qualitative_failure_labels(
        evaluator_output,
        qualitative_failure_labels=QUALITATIVE_FAILURE_LABELS,
    ), (
        "Qualitative failure label check failed — see arksim/results/evaluation for details"
    )
