"""Integration test for HTML report generation.

Uses importlib to load modules directly from file paths, bypassing
the arksim.__init__.py chains that pull in heavy deps
(langchain, azure, etc.) that may not be installed in the test env.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Direct-import helpers: load .py files without triggering __init__.py chains
# ---------------------------------------------------------------------------
_ARKSIM_ROOT = Path(__file__).resolve().parents[2] / "arksim"


def _load_module(name: str, filepath: Path) -> types.ModuleType:
    """Load a Python module from filepath into sys.modules[name]."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1. Load lightweight dependency modules first
_load_module(
    "arksim.evaluator.utils.enums",
    _ARKSIM_ROOT / "evaluator" / "utils" / "enums.py",
)
_load_module(
    "arksim.evaluator.base_metric",
    _ARKSIM_ROOT / "evaluator" / "base_metric.py",
)

# 2. Load entities
_entities_mod = _load_module(
    "arksim.evaluator.entities",
    _ARKSIM_ROOT / "evaluator" / "entities.py",
)

# 3. Load simulation_engine.entities directly (no arksim deps)
_sim_entities_mod = _load_module(
    "arksim.simulation_engine.entities",
    _ARKSIM_ROOT / "simulation_engine" / "entities.py",
)

# 4. Stub arksim.simulation_engine package to export Conversation and Simulation
_sim_pkg = types.ModuleType("arksim.simulation_engine")
_sim_pkg.Conversation = _sim_entities_mod.Conversation
_sim_pkg.Simulation = _sim_entities_mod.Simulation
_sim_pkg.combine_knowledge = MagicMock()
sys.modules["arksim.simulation_engine"] = _sim_pkg


# 5. Stub arksim.scenario with minimal Pydantic Scenario and Scenarios classes
class _KnowledgeItem(BaseModel):
    content: str


class _Scenario(BaseModel):
    scenario_id: str
    goal: str = ""
    user_attributes: dict[str, Any] = Field(default_factory=dict)
    knowledge: list[_KnowledgeItem] = Field(default_factory=list)


class _Scenarios(BaseModel):
    schema_version: str = "1.0"
    scenarios: list[_Scenario] = Field(default_factory=list)


_scenario_mod = types.ModuleType("arksim.scenario")
_scenario_mod.Scenario = _Scenario
_scenario_mod.Scenarios = _Scenarios
sys.modules["arksim.scenario"] = _scenario_mod

# 6. Stub heavy deps not needed by generate_html_report but safe to stub
_orig_llms = sys.modules.get("arksim.llms")
_orig_llms_chat = sys.modules.get("arksim.llms.chat")
sys.modules["arksim.llms"] = MagicMock()
sys.modules["arksim.llms.chat"] = MagicMock()

# 7. Load generate_html_report (imports pandas, entities — all available now)
_gen_report_mod = _load_module(
    "arksim.utils.html_report.generate_html_report",
    _ARKSIM_ROOT / "utils" / "html_report" / "generate_html_report.py",
)

# Restore original sys.modules to avoid polluting other test files
if _orig_llms is None:
    sys.modules.pop("arksim.llms", None)
else:
    sys.modules["arksim.llms"] = _orig_llms
if _orig_llms_chat is None:
    sys.modules.pop("arksim.llms.chat", None)
else:
    sys.modules["arksim.llms.chat"] = _orig_llms_chat

# Pull references
Evaluation = _entities_mod.Evaluation
ConversationEvaluation = _entities_mod.ConversationEvaluation
TurnEvaluation = _entities_mod.TurnEvaluation
QuantResult = _entities_mod.QuantResult
UniqueError = _entities_mod.UniqueError
Occurrence = _entities_mod.Occurrence
Conversation = _sim_entities_mod.Conversation
Simulation = _sim_entities_mod.Simulation
Message = _sim_entities_mod.Message
SimulatedUserPrompt = _sim_entities_mod.SimulatedUserPrompt
HtmlReportParams = _gen_report_mod.HtmlReportParams
generate_html_report = _gen_report_mod.generate_html_report


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _build_test_data() -> tuple[Simulation, Evaluation, dict[str, str]]:
    """Build minimal test data for HTML report generation."""
    conversations = Simulation(
        schema_version="1.0",
        simulator_version="test",
        conversations=[
            Conversation(
                conversation_id="conv-uuid-1",
                scenario_id="sc-1",
                conversation_history=[
                    Message(
                        turn_id=0, role="simulated_user", content="What is the weather?"
                    ),
                    Message(turn_id=0, role="assistant", content="It is sunny today."),
                    Message(turn_id=1, role="simulated_user", content="Thanks!"),
                    Message(turn_id=1, role="assistant", content="You're welcome!"),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="You are a user.",
                    variables={"goal": "Get weather info"},
                ),
            ),
            Conversation(
                conversation_id="conv-uuid-2",
                scenario_id="sc-2",
                conversation_history=[
                    Message(
                        turn_id=0,
                        role="simulated_user",
                        content="Tell me about insurance",
                    ),
                    Message(
                        turn_id=0, role="assistant", content="Sure, let me explain."
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="You are a user.",
                    variables={"goal": "Learn about insurance"},
                ),
            ),
        ],
    )

    evaluation_results = Evaluation(
        schema_version="1.0",
        generated_at="2025-01-01T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="eval-uuid-1",
        simulation_id="sim-uuid-1",
        conversations=[
            ConversationEvaluation(
                conversation_id="conv-uuid-1",
                goal_completion_score=1.0,
                goal_completion_reason="Completed",
                turn_success_ratio=1.0,
                overall_agent_score=1.0,
                evaluation_status="Done",
                turn_scores=[
                    TurnEvaluation(
                        turn_id=0,
                        scores=[
                            QuantResult(name="helpfulness", value=4.0, reason="Good"),
                            QuantResult(name="coherence", value=4.0, reason="Good"),
                        ],
                        turn_score=4.0,
                        turn_behavior_failure="no failure",
                        turn_behavior_failure_reason="All good",
                    ),
                    TurnEvaluation(
                        turn_id=1,
                        scores=[
                            QuantResult(name="helpfulness", value=4.5, reason="Great"),
                            QuantResult(name="coherence", value=4.5, reason="Great"),
                        ],
                        turn_score=4.5,
                        turn_behavior_failure="no failure",
                        turn_behavior_failure_reason="All good",
                    ),
                ],
            ),
            ConversationEvaluation(
                conversation_id="conv-uuid-2",
                goal_completion_score=0.3,
                goal_completion_reason="Incomplete",
                turn_success_ratio=0.5,
                overall_agent_score=0.45,
                evaluation_status="Failed",
                turn_scores=[
                    TurnEvaluation(
                        turn_id=0,
                        scores=[
                            QuantResult(
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
            UniqueError(
                unique_error_id="uid-1",
                behavior_failure_category="lack of specific information",
                unique_error_description="Agent fails to provide detailed insurance information",
                severity="medium",
                occurrences=[
                    Occurrence(conversation_id="conv-uuid-2", turn_id=0),
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

    def test_html_contains_chat_id_to_label(self) -> None:
        """Generated HTML embeds CHAT_ID_TO_LABEL JS constant."""
        convos, eval_results, label_map = _build_test_data()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL" in html
        assert '"conv-uuid-1": "Conversation 1"' in html
        assert '"conv-uuid-2": "Conversation 2"' in html

    def test_html_contains_occurrences_data(self) -> None:
        """Unique errors data includes occurrence dict keyed by conversation_id."""
        convos, eval_results, label_map = _build_test_data()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = generate_html_report(params)
        html = Path(result_path).read_text()

        # The unique errors row should contain the occurrence conversation id
        assert "conv-uuid-2" in html
        assert "lack of specific information" in html

    def test_html_contains_chat_label_function(self) -> None:
        """Template includes chatLabel() and toggleSnippet() JS helper functions."""
        convos, eval_results, label_map = _build_test_data()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = generate_html_report(params)
        html = Path(result_path).read_text()

        assert "function chatLabel(chatId)" in html
        assert "function toggleSnippet(" in html

    def test_empty_label_map_produces_valid_html(self) -> None:
        """Empty chat_id_to_label still produces valid HTML."""
        convos, eval_results, _ = _build_test_data()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label={},
        )
        result_path = generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL = {}" in html

    def test_no_unique_errors_produces_valid_html(self) -> None:
        """Report with no unique errors still generates valid HTML."""
        convos, eval_results, label_map = _build_test_data()
        eval_results.unique_errors = []

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL" in html
        assert "chatLabel" in html

    def test_html_contains_methodology_section(self) -> None:
        """Report includes collapsible scoring methodology section."""
        convos, eval_results, label_map = _build_test_data()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = generate_html_report(params)
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

    def test_html_contains_severity_badge(self) -> None:
        """Error cards include severity badge and sorting logic."""
        convos, eval_results, label_map = _build_test_data()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = generate_html_report(params)
        html = Path(result_path).read_text()

        assert "severity-badge" in html
        assert "severity-critical" in html
        assert "severity-high" in html
        assert "severity-medium" in html
        assert "severity-low" in html
        assert "severityOrder" in html
        assert "module-level heuristics" in html
