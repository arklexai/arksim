"""Tests for error_detection: collect_agent_behavior_failure_reasoning.

Uses importlib to load modules directly from file paths, bypassing
the arksim.evaluator.__init__.py which pulls in heavy deps
(langchain, azure, etc.) that may not be installed in the test env.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Direct-import helpers: load .py files without triggering __init__.py chains
# ---------------------------------------------------------------------------
_ARKSIM_ROOT = Path(__file__).resolve().parents[2] / "arksim"


def _load_module(name: str, filepath: Path):
    """Load a Python module from filepath into sys.modules[name]."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 1. Load lightweight dependency modules first
_enums_mod = _load_module(
    "arksim.evaluator.utils.enums",
    _ARKSIM_ROOT / "evaluator" / "utils" / "enums.py",
)

_base_metric_mod = _load_module(
    "arksim.evaluator.base_metric",
    _ARKSIM_ROOT / "evaluator" / "base_metric.py",
)

# 2. Load entities.py directly
_entities_mod = _load_module(
    "arksim.evaluator.entities",
    _ARKSIM_ROOT / "evaluator" / "entities.py",
)

# 3. Stub LLM deps before loading error_detection
_orig_llms = sys.modules.get("arksim.llms")
_orig_llms_chat = sys.modules.get("arksim.llms.chat")
sys.modules["arksim.llms"] = MagicMock()
sys.modules["arksim.llms.chat"] = MagicMock()

_prompts_mod = _load_module(
    "arksim.evaluator.utils.prompts",
    _ARKSIM_ROOT / "evaluator" / "utils" / "prompts.py",
)
_schema_mod = _load_module(
    "arksim.evaluator.utils.schema",
    _ARKSIM_ROOT / "evaluator" / "utils" / "schema.py",
)

_error_detection_mod = _load_module(
    "arksim.evaluator.error_detection",
    _ARKSIM_ROOT / "evaluator" / "error_detection.py",
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


# Pull out the classes/functions we need
ConversationEvaluation = _entities_mod.ConversationEvaluation
TurnEvaluation = _entities_mod.TurnEvaluation
QuantResult = _entities_mod.QuantResult
EvaluationOutcomes = _enums_mod.EvaluationOutcomes
collect_agent_behavior_failure_reasoning = (
    _error_detection_mod.collect_agent_behavior_failure_reasoning
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_turn(turn_id: int, failure_label: str, failure_reason: str) -> TurnEvaluation:
    """Build a TurnEvaluation with minimal fields."""
    return TurnEvaluation(
        turn_id=turn_id,
        scores=[],
        turn_score=-1,
        turn_behavior_failure=failure_label,
        turn_behavior_failure_reason=failure_reason,
    )


def _make_conv_eval(
    conversation_id: str, turns: list
) -> ConversationEvaluation:
    """Build a ConversationEvaluation with minimal fields."""
    return ConversationEvaluation(
        conversation_id=conversation_id,
        goal_completion_score=0.5,
        goal_completion_reason="OK",
        turn_success_ratio=0.5,
        overall_agent_score=0.5,
        evaluation_status="Done",
        turn_scores=turns,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCollectAgentBehaviorFailureReasoning:
    """Tests for collect_agent_behavior_failure_reasoning."""

    def test_collects_failure_turns(self):
        """Turns with a matching failure label are collected."""
        turn = _make_turn(0, "lack of specific information", "Agent missed key details")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning(
            [conv], ["lack of specific information"]
        )
        assert len(result) == 1
        assert "conv-1_0" in result[0]
        assert "lack of specific information" in result[0]
        assert "Agent missed key details" in result[0]

    def test_skips_no_failure(self):
        """Turns with 'no failure' label are always skipped."""
        turn = _make_turn(0, "no failure", "All good")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning(
            [conv], ["lack of specific information"]
        )
        assert len(result) == 0

    def test_skips_non_matching_categories(self):
        """Turns whose label is not in failure_categories are skipped."""
        turn = _make_turn(0, "repetition", "Repeated same response")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning(
            [conv], ["lack of specific information"]
        )
        assert len(result) == 0

    def test_empty_conversations(self):
        """Empty conversation list returns empty result."""
        result = collect_agent_behavior_failure_reasoning([], ["repetition"])
        assert result == []

    def test_empty_categories(self):
        """Empty categories list matches nothing."""
        turn = _make_turn(0, "repetition", "Repeated")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning([conv], [])
        assert len(result) == 0

    def test_multiple_turns_multiple_convos(self):
        """Collects from multiple turns across multiple conversations."""
        turn1 = _make_turn(0, "repetition", "Repeated response")
        turn2 = _make_turn(1, "false information", "Wrong facts provided")
        conv1 = _make_conv_eval("conv-1", [turn1, turn2])
        turn3 = _make_turn(0, "no failure", "All good")
        conv2 = _make_conv_eval("conv-2", [turn3])
        categories = ["repetition", "false information"]
        result = collect_agent_behavior_failure_reasoning([conv1, conv2], categories)
        assert len(result) == 2

    def test_skips_special_outcomes(self):
        """skipped_good_performance, evaluation_run_failed, agent_api_error are skipped."""
        skip_labels = [
            EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
            EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
            EvaluationOutcomes.AGENT_API_ERROR.value,
        ]
        turns = [_make_turn(i, label, "reason") for i, label in enumerate(skip_labels)]
        conv = _make_conv_eval("conv-1", turns)
        result = collect_agent_behavior_failure_reasoning([conv], skip_labels)
        assert len(result) == 0

    def test_result_format(self):
        """Each result string encodes conv_id, turn_id, label and reason."""
        turn = _make_turn(3, "repetition", "Said the same thing twice")
        conv = _make_conv_eval("abc-xyz", [turn])
        result = collect_agent_behavior_failure_reasoning([conv], ["repetition"])
        assert len(result) == 1
        assert "abc-xyz_3" in result[0]
        assert "repetition" in result[0]
        assert "Said the same thing twice" in result[0]

    def test_only_failure_turns_collected(self):
        """Only failing turns within a conversation are collected, not successful ones."""
        turn_fail = _make_turn(0, "repetition", "Repeated")
        turn_ok = _make_turn(1, "no failure", "Fine")
        conv = _make_conv_eval("conv-1", [turn_fail, turn_ok])
        result = collect_agent_behavior_failure_reasoning([conv], ["repetition"])
        assert len(result) == 1
        assert "conv-1_0" in result[0]
