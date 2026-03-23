# SPDX-License-Identifier: Apache-2.0
"""
Example script that runs the full pipeline (simulation -> evaluation)
programmatically using a custom agent class - no HTTP server required.

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from custom_agent import ToolCallExampleAgent
from custom_metrics import (
    DataPrivacyMetric,
    ToolUsageEfficiencyMetric,
    UnauthorizedActionMetric,
    VerificationComplianceMetric,
)

from arksim.config import AgentConfig, CustomConfig
from arksim.evaluator import (
    EvaluationParams,
    Evaluator,
    check_numeric_thresholds,
    check_qualitative_failure_labels,
)
from arksim.llms.chat import LLM
from arksim.scenario import Scenarios
from arksim.simulation_engine import SimulationParams, Simulator
from arksim.utils.html_report.generate_html_report import (
    HtmlReportParams,
    generate_html_report,
)


async def main() -> None:
    # -- Paths ---------------------------------------------------------------
    agent_setup_dir = os.path.dirname(__file__)
    results_dir = os.path.join(agent_setup_dir, "results")

    scenario_file_path = os.path.join(agent_setup_dir, "scenarios.json")
    simulation_output_file = os.path.join(results_dir, "simulation", "simulation.json")
    evaluation_output_dir = os.path.join(results_dir, "evaluation")

    model = "gpt-4.1-mini"
    provider = "openai"
    max_turns = 3

    # -- Step 1: Run Simulation ----------------------------------------------
    print("=" * 60)
    print("Step 1/2: Running Simulation...")
    print("=" * 60)

    scenario_output = Scenarios.load(scenario_file_path)
    agent_config = AgentConfig(
        agent_type="custom",
        agent_name=ToolCallExampleAgent.__name__,
        custom_config=CustomConfig(agent_class=ToolCallExampleAgent),
    )
    llm = LLM(model=model, provider=provider)

    simulation_params = SimulationParams(
        num_convos_per_scenario=2,
        max_turns=max_turns,
        num_workers=50,
        output_file_path=simulation_output_file,
    )

    simulator = Simulator(
        agent_config=agent_config,
        simulator_params=simulation_params,
        llm=llm,
    )
    simulation_output = await simulator.simulate(scenario_output)
    await simulator.save()

    # -- Step 2: Evaluate (with custom metrics) ------------------------------
    print("\n" + "=" * 60)
    print("Step 2/2: Evaluating Results...")
    print("=" * 60)

    evaluator_params = EvaluationParams(
        output_dir=evaluation_output_dir,
        num_workers=50,
        metrics_to_run=[
            "helpfulness",
            "agent_behavior_failure",
            "tool_call_behavior_failure",
        ],
        custom_metrics=[
            VerificationComplianceMetric(),
            ToolUsageEfficiencyMetric(),
        ],
        custom_qualitative_metrics=[
            UnauthorizedActionMetric(),
            DataPrivacyMetric(),
        ],
    )

    evaluator = Evaluator(evaluator_params, llm=llm, scenarios=scenario_output)
    evaluator_output = await asyncio.to_thread(evaluator.evaluate, simulation_output)
    evaluator.display_evaluation_summary()
    evaluator.save_results()

    # -- Step 3: Check thresholds --------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3/3: Checking Thresholds...")
    print("=" * 60)

    numeric_thresholds = {
        "overall_score": 0.6,
        "verification_compliance": 3.0,
    }
    qualitative_failure_labels = {
        "unauthorized_action": ["violated"],
    }

    numeric_ok = check_numeric_thresholds(evaluator_output, numeric_thresholds)
    qual_ok = check_qualitative_failure_labels(
        evaluator_output, qualitative_failure_labels
    )

    if not numeric_ok or not qual_ok:
        print("\nEvaluation failed: one or more threshold gates did not pass.")
        sys.exit(1)

    # Generate HTML report
    html_output_path = os.path.join(evaluation_output_dir, "final_report.html")
    report_params = HtmlReportParams(
        simulation=simulation_output,
        evaluation=evaluator_output,
        scenarios=scenario_output,
        output_path=html_output_path,
        chat_id_to_label=evaluator.chat_id_to_label,
    )
    await asyncio.to_thread(generate_html_report, report_params)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to: {results_dir}")
    print(f"HTML report: {html_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
