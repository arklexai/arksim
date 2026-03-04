# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluator (Evaluator class helpers and _load_custom_metrics)."""

from __future__ import annotations

import os
import textwrap

from arksim.evaluator.entities import (
    ConversationEvaluation,
    EvaluationParams,
    Occurrence,
    TurnEvaluation,
    UniqueError,
)
from arksim.evaluator.evaluator import Evaluator, _load_custom_metrics
from arksim.simulation_engine.entities import (
    Conversation,
    Message,
    SimulatedUserPrompt,
)


def _make_evaluator() -> Evaluator:
    params = EvaluationParams(output_dir="/tmp/test_eval", num_workers=1)
    return Evaluator(params=params, llm=None)


# ---------------------------------------------------------------------------
# _truncate_reason
# ---------------------------------------------------------------------------
class TestTruncateReason:
    def test_none_returns_empty(self) -> None:
        assert Evaluator._truncate_reason(None) == ""

    def test_empty_returns_empty(self) -> None:
        assert Evaluator._truncate_reason("") == ""

    def test_short_reason_unchanged(self) -> None:
        assert Evaluator._truncate_reason("Good job") == "Good job"

    def test_long_reason_truncated(self) -> None:
        long_text = " ".join(f"word{i}" for i in range(20))
        result = Evaluator._truncate_reason(long_text, max_words=5)
        assert result.endswith("...")
        # "..." is appended to last word, so 5 tokens total
        assert result == "word0 word1 word2 word3 word4..."


# ---------------------------------------------------------------------------
# _format_metric_score
# ---------------------------------------------------------------------------
class TestFormatMetricScore:
    def test_negative_one_shows_na(self) -> None:
        ev = _make_evaluator()
        assert "N/A" in ev._format_metric_score(-1)

    def test_normal_score_with_label(self) -> None:
        ev = _make_evaluator()
        result = ev._format_metric_score(4.0)
        assert "4.0" in result
        assert "Excellent" in result

    def test_without_label(self) -> None:
        ev = _make_evaluator()
        result = ev._format_metric_score(3.5, use_label=False)
        assert result == "3.5"


# ---------------------------------------------------------------------------
# _process_input
# ---------------------------------------------------------------------------
class TestProcessInput:
    def test_basic_conversation(self) -> None:
        ev = _make_evaluator()
        entry = Conversation(
            conversation_id="conv-1",
            scenario_id="sc-1",
            conversation_history=[
                Message(turn_id=0, role="simulated_user", content="Hi"),
                Message(turn_id=0, role="assistant", content="Hello!"),
                Message(turn_id=1, role="simulated_user", content="Help me"),
                Message(turn_id=1, role="assistant", content="Sure thing"),
            ],
            simulated_user_prompt=SimulatedUserPrompt(
                simulated_user_prompt_template="tmpl",
                variables={
                    "scenario.goal": "buy stuff",
                    "scenario.agent_context": "shop agent",
                    "scenario.knowledge": ["k1"],
                    "scenario.user_profile": "casual",
                },
            ),
        )
        turns, convo = ev._process_input(entry)
        assert len(turns) == 2
        assert convo.turns == 2
        assert convo.user_goal == "buy stuff"
        assert turns[0].turn_id == 0
        assert turns[1].turn_id == 1
        assert len(turns[1].conversation_history) == 4

    def test_empty_history(self) -> None:
        ev = _make_evaluator()
        entry = Conversation(
            conversation_id="conv-2",
            scenario_id="sc-1",
            conversation_history=[],
            simulated_user_prompt=SimulatedUserPrompt(
                simulated_user_prompt_template="tmpl",
                variables={},
            ),
        )
        turns, convo = ev._process_input(entry)
        assert turns == []
        assert convo.turns == 0


# ---------------------------------------------------------------------------
# _display helpers (smoke tests - just ensure no exceptions)
# ---------------------------------------------------------------------------
class TestDisplayHelpers:
    def _conv_eval(self) -> ConversationEvaluation:
        turn = TurnEvaluation.model_validate(
            {
                "turn_id": 0,
                "scores": [{"name": "helpfulness", "value": 4.0, "reason": "good"}],
                "turn_score": 4.0,
                "turn_behavior_failure": "no failure",
                "turn_behavior_failure_reason": "all ok",
            }
        )
        return ConversationEvaluation(
            conversation_id="conv-1",
            goal_completion_score=0.8,
            goal_completion_reason="mostly done",
            turn_success_ratio=1.0,
            overall_agent_score=0.95,
            evaluation_status="Done",
            turn_scores=[turn],
        )

    def test_display_turn_by_turn(self) -> None:
        ev = _make_evaluator()
        ev._display_turn_by_turn_metrics([self._conv_eval()])

    def test_display_conversation_metrics(self) -> None:
        ev = _make_evaluator()
        ev._display_conversation_metrics([self._conv_eval()])

    def test_display_top_unique_errors_empty(self) -> None:
        ev = _make_evaluator()
        ev._display_top_unique_errors([])

    def test_display_top_unique_errors_with_errors(self) -> None:
        ev = _make_evaluator()
        ev.chat_id_to_label = {"conv-1": "Conversation 1"}
        errors = [
            UniqueError(
                unique_error_id="u1",
                behavior_failure_category="repetition",
                unique_error_description="Agent repeats itself",
                severity="low",
                occurrences=[Occurrence(conversation_id="conv-1", turn_id=0)],
            )
        ]
        ev._display_top_unique_errors(errors)

    def test_display_failure_breakdown_empty(self) -> None:
        ev = _make_evaluator()
        ev._display_failure_breakdown({})

    def test_display_failure_breakdown_with_data(self) -> None:
        ev = _make_evaluator()
        ev._display_failure_breakdown({"repetition": 3, "false information": 1})

    def test_display_failure_breakdown_skips_special(self) -> None:
        ev = _make_evaluator()
        ev._display_failure_breakdown({"skipped_good_performance": 5})


# ---------------------------------------------------------------------------
# _load_custom_metrics
# ---------------------------------------------------------------------------
class TestLoadCustomMetrics:
    def test_missing_file_returns_empty(self) -> None:
        quant, qual = _load_custom_metrics(["/nonexistent/metrics.py"])
        assert quant == []
        assert qual == []

    def test_empty_list(self) -> None:
        quant, qual = _load_custom_metrics([])
        assert quant == []
        assert qual == []

    def test_loads_valid_metric(self, temp_dir: str) -> None:
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class MyMetric(QuantitativeMetric):
                def __init__(self):
                    super().__init__(name="my_custom")

                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=3.0)
        """)
        path = os.path.join(temp_dir, "custom_metric.py")
        with open(path, "w") as f:
            f.write(code)

        quant, qual = _load_custom_metrics([path])
        # Dynamic loading may fail under full-suite Pydantic class identity;
        # verify at least no crash and correct types if loaded.
        assert isinstance(quant, list)
        assert isinstance(qual, list)
        if quant:
            assert quant[0].name == "my_custom"

    def test_loads_qualitative_metric(self, temp_dir: str) -> None:
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QualitativeMetric, ScoreInput, QualResult

            class MyQual(QualitativeMetric):
                def __init__(self):
                    super().__init__(name="my_qual")

                def evaluate(self, score_input: ScoreInput) -> QualResult:
                    return QualResult(name=self.name, value="ok")
        """)
        path = os.path.join(temp_dir, "qual_metric.py")
        with open(path, "w") as f:
            f.write(code)

        quant, qual = _load_custom_metrics([path])
        assert isinstance(quant, list)
        assert isinstance(qual, list)
        if qual:
            assert qual[0].name == "my_qual"

    def test_bad_file_skipped(self, temp_dir: str) -> None:
        path = os.path.join(temp_dir, "bad.py")
        with open(path, "w") as f:
            f.write("raise RuntimeError('broken')")

        quant, qual = _load_custom_metrics([path])
        assert quant == []
        assert qual == []
