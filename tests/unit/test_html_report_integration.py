# SPDX-License-Identifier: Apache-2.0
"""Integration test for HTML report generation.

Uses importlib to load modules directly from file paths, bypassing
the arksim.__init__.py chains that pull in heavy deps
(langchain, azure, etc.) that may not be installed in the test env.

All sys.modules patching is done inside a module-scoped fixture so that
monkeypatch restores the original entries automatically after the test
module finishes — no manual bookkeeping required.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path constant and raw loader (no sys.modules side-effects at import time)
# ---------------------------------------------------------------------------
_ARKSIM_ROOT = Path(__file__).resolve().parents[2] / "arksim"


def _load_module(name: str, filepath: Path) -> types.ModuleType:
    """Load a Python module from filepath into sys.modules[name]."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Minimal Pydantic stubs used as arksim.scenario replacements
# (pure class definitions — no sys.modules manipulation)
# ---------------------------------------------------------------------------
class _KnowledgeItem(BaseModel):
    content: str


class _Scenario(BaseModel):
    scenario_id: str
    goal: str = ""
    user_profile: str = ""
    knowledge: list[_KnowledgeItem] = Field(default_factory=list)
    origin: dict[str, Any] = Field(default_factory=dict)


class _Scenarios(BaseModel):
    schema_version: str = "1.0"
    scenarios: list[_Scenario] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _module_mp() -> Generator[MonkeyPatch, None, None]:
    """Module-scoped MonkeyPatch that restores sys.modules after all tests."""
    with MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def html_report_env(_module_mp: MonkeyPatch) -> SimpleNamespace:
    """Load the HTML report stack with isolated sys.modules via monkeypatch.

    monkeypatch.setitem records the original value (or absence) and restores
    it automatically when the module scope exits — no manual bookkeeping.
    """
    mp = _module_mp

    def load(name: str, rel: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, _ARKSIM_ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    # 1. Lightweight deps
    load("arksim.evaluator.utils.enums", "evaluator/utils/enums.py")
    load("arksim.evaluator.base_metric", "evaluator/base_metric.py")

    # 2. Entities
    entities_mod = load("arksim.evaluator.entities", "evaluator/entities.py")

    # 3. Simulation engine entities
    sim_entities_mod = load(
        "arksim.simulation_engine.entities", "simulation_engine/entities.py"
    )

    # 4. Stub arksim.simulation_engine package
    sim_pkg = types.ModuleType("arksim.simulation_engine")
    sim_pkg.Conversation = sim_entities_mod.Conversation
    sim_pkg.Simulation = sim_entities_mod.Simulation
    sim_pkg.combine_knowledge = MagicMock()
    mp.setitem(sys.modules, "arksim.simulation_engine", sim_pkg)

    # 5. Stub arksim.scenario
    scenario_mod = types.ModuleType("arksim.scenario")
    scenario_mod.Scenario = _Scenario
    scenario_mod.Scenarios = _Scenarios
    mp.setitem(sys.modules, "arksim.scenario", scenario_mod)

    # 6. Stub heavy deps
    mp.setitem(sys.modules, "arksim.llms", MagicMock())
    mp.setitem(sys.modules, "arksim.llms.chat", MagicMock())

    # 7. Load generate_html_report
    gen_report_mod = load(
        "arksim.utils.html_report.generate_html_report",
        "utils/html_report/generate_html_report.py",
    )

    return SimpleNamespace(
        Evaluation=entities_mod.Evaluation,
        ConversationEvaluation=entities_mod.ConversationEvaluation,
        TurnEvaluation=entities_mod.TurnEvaluation,
        QuantResult=entities_mod.QuantResult,
        UniqueError=entities_mod.UniqueError,
        Occurrence=entities_mod.Occurrence,
        Conversation=sim_entities_mod.Conversation,
        Simulation=sim_entities_mod.Simulation,
        Message=sim_entities_mod.Message,
        SimulatedUserPrompt=sim_entities_mod.SimulatedUserPrompt,
        HtmlReportParams=gen_report_mod.HtmlReportParams,
        generate_html_report=gen_report_mod.generate_html_report,
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _build_test_data(env: SimpleNamespace) -> tuple[Any, Any, dict[str, str]]:
    """Build minimal test data for HTML report generation."""
    conversations = env.Simulation(
        schema_version="1.0",
        simulator_version="test",
        conversations=[
            env.Conversation(
                conversation_id="conv-uuid-1",
                scenario_id="sc-1",
                conversation_history=[
                    env.Message(
                        turn_id=0, role="simulated_user", content="What is the weather?"
                    ),
                    env.Message(
                        turn_id=0, role="assistant", content="It is sunny today."
                    ),
                    env.Message(turn_id=1, role="simulated_user", content="Thanks!"),
                    env.Message(turn_id=1, role="assistant", content="You're welcome!"),
                ],
                simulated_user_prompt=env.SimulatedUserPrompt(
                    simulated_user_prompt_template="You are a user.",
                    variables={"goal": "Get weather info"},
                ),
            ),
            env.Conversation(
                conversation_id="conv-uuid-2",
                scenario_id="sc-2",
                conversation_history=[
                    env.Message(
                        turn_id=0,
                        role="simulated_user",
                        content="Tell me about insurance",
                    ),
                    env.Message(
                        turn_id=0, role="assistant", content="Sure, let me explain."
                    ),
                ],
                simulated_user_prompt=env.SimulatedUserPrompt(
                    simulated_user_prompt_template="You are a user.",
                    variables={"goal": "Learn about insurance"},
                ),
            ),
        ],
    )

    evaluation_results = env.Evaluation(
        schema_version="1.0",
        generated_at="2025-01-01T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="eval-uuid-1",
        simulation_id="sim-uuid-1",
        conversations=[
            env.ConversationEvaluation(
                conversation_id="conv-uuid-1",
                goal_completion_score=1.0,
                goal_completion_reason="Completed",
                turn_success_ratio=1.0,
                overall_agent_score=1.0,
                evaluation_status="Done",
                turn_scores=[
                    env.TurnEvaluation(
                        turn_id=0,
                        scores=[
                            env.QuantResult(
                                name="helpfulness", value=4.0, reason="Good"
                            ),
                            env.QuantResult(name="coherence", value=4.0, reason="Good"),
                        ],
                        turn_score=4.0,
                        turn_behavior_failure="no failure",
                        turn_behavior_failure_reason="All good",
                    ),
                    env.TurnEvaluation(
                        turn_id=1,
                        scores=[
                            env.QuantResult(
                                name="helpfulness", value=4.5, reason="Great"
                            ),
                            env.QuantResult(
                                name="coherence", value=4.5, reason="Great"
                            ),
                        ],
                        turn_score=4.5,
                        turn_behavior_failure="no failure",
                        turn_behavior_failure_reason="All good",
                    ),
                ],
            ),
            env.ConversationEvaluation(
                conversation_id="conv-uuid-2",
                goal_completion_score=0.3,
                goal_completion_reason="Incomplete",
                turn_success_ratio=0.5,
                overall_agent_score=0.45,
                evaluation_status="Failed",
                turn_scores=[
                    env.TurnEvaluation(
                        turn_id=0,
                        scores=[
                            env.QuantResult(
                                name="helpfulness", value=2.0, reason="Lacks detail"
                            ),
                        ],
                        turn_score=2.0,
                        turn_behavior_failure="lack of specific information",
                        turn_behavior_failure_reason="Did not provide details",
                    ),
                ],
            ),
        ],
        unique_errors=[
            env.UniqueError(
                unique_error_id="uid-1",
                behavior_failure_category="lack of specific information",
                unique_error_description="Agent fails to provide detailed insurance information",
                severity="medium",
                occurrences=[
                    env.Occurrence(conversation_id="conv-uuid-2", turn_id=0),
                ],
            ),
        ],
    )

    chat_id_to_label = {
        "conv-uuid-1": "Conversation 1",
        "conv-uuid-2": "Conversation 2",
    }

    return conversations, evaluation_results, chat_id_to_label


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHtmlReportIntegration:
    """Integration tests for generate_html_report with new evaluation schema."""

    def test_html_contains_chat_id_to_label(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Generated HTML embeds CHAT_ID_TO_LABEL JS constant."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL" in html
        assert '"conv-uuid-1": "Conversation 1"' in html
        assert '"conv-uuid-2": "Conversation 2"' in html

    def test_html_contains_occurrences_data(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Unique errors data includes occurrence dict keyed by conversation_id."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "conv-uuid-2" in html
        assert "lack of specific information" in html

    def test_html_contains_chat_label_function(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Template includes chatLabel() and toggleSnippet() JS helper functions."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "function chatLabel(chatId)" in html
        assert "function toggleSnippet(" in html

    def test_empty_label_map_produces_valid_html(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Empty chat_id_to_label still produces valid HTML."""
        convos, eval_results, _ = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label={},
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL = {}" in html

    def test_no_unique_errors_produces_valid_html(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Report with no unique errors still generates valid HTML."""
        convos, eval_results, label_map = _build_test_data(html_report_env)
        eval_results.unique_errors = []

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL" in html
        assert "chatLabel" in html

    def test_html_contains_methodology_section(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Report includes collapsible scoring methodology section."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "How Scores Are Computed" in html
        assert "toggleMethodology" in html
        assert "methodologyContent" in html
        assert "Helpfulness" in html
        assert "Coherence" in html
        assert "Verbosity" in html
        assert "Relevance" in html
        assert "Faithfulness" in html
        assert "Poor" in html
        assert "Needs Improvement" in html
        assert "Excellent" in html
        assert "turn_success_ratio" in html
        assert "goal_completion" in html
        assert "0.75" in html
        assert "0.25" in html
        assert "arxiv.org" in html

    def test_html_contains_severity_badge(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Error cards include severity badge and sorting logic."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "severity-badge" in html
        assert "severity-critical" in html
        assert "severity-high" in html
        assert "severity-medium" in html
        assert "severity-low" in html
        assert "severityOrder" in html
        assert "module-level heuristics" in html
