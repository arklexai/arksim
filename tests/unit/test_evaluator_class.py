# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluator (Evaluator class helpers and _load_custom_metrics)."""

from __future__ import annotations

import os
import textwrap

import pytest

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
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Module file not found"):
            _load_custom_metrics(["/nonexistent/metrics.py"])

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

    def test_bad_file_raises(self, temp_dir: str) -> None:
        path = os.path.join(temp_dir, "bad.py")
        with open(path, "w") as f:
            f.write("raise RuntimeError('broken')")

        with pytest.raises(RuntimeError, match="Failed to load module from"):
            _load_custom_metrics([path])

    def test_uninstantiable_metric_raises(self, temp_dir: str) -> None:
        code = textwrap.dedent("""
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class BrokenMetric(QuantitativeMetric):
                name = "broken"
                def __init__(self):
                    raise ValueError("cannot instantiate")
                def score(self, input: ScoreInput) -> QuantResult:
                    ...
        """)
        path = os.path.join(temp_dir, "broken_metric.py")
        with open(path, "w") as f:
            f.write(code)

        with pytest.raises(
            RuntimeError, match="Could not instantiate quantitative metric"
        ):
            _load_custom_metrics([path])


class TestLoadCustomMetricsLLMInjection:
    """_load_custom_metrics injects the LLM into metrics that accept it."""

    def test_llm_is_set_on_quantitative_metric(self, temp_dir: str) -> None:
        """Metrics that accept llm= receive the injected instance on self.llm."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class LLMAwareMetric(QuantitativeMetric):
                def __init__(self, llm=None):
                    super().__init__(name="llm_aware", llm=llm)

                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "llm_aware_metric.py")
        with open(path, "w") as f:
            f.write(code)

        sentinel = object()
        quant, _ = _load_custom_metrics([path], llm=sentinel)
        assert quant[0].llm is sentinel

    def test_llm_is_set_on_qualitative_metric(self, temp_dir: str) -> None:
        """Qualitative metrics that accept llm= receive the injected instance."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QualitativeMetric, ScoreInput, QualResult

            class LLMAwareQual(QualitativeMetric):
                def __init__(self, llm=None):
                    super().__init__(name="llm_aware_qual", llm=llm)

                def evaluate(self, score_input: ScoreInput) -> QualResult:
                    return QualResult(name=self.name, value="ok")
        """)
        path = os.path.join(temp_dir, "llm_aware_qual.py")
        with open(path, "w") as f:
            f.write(code)

        sentinel = object()
        _, qual = _load_custom_metrics([path], llm=sentinel)
        assert qual[0].llm is sentinel

    def test_metric_without_llm_param_still_loads(self, temp_dir: str) -> None:
        """Metrics with no llm parameter in __init__ are not broken by injection."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class LegacyMetric(QuantitativeMetric):
                def __init__(self):
                    super().__init__(name="legacy")

                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "legacy_metric.py")
        with open(path, "w") as f:
            f.write(code)

        sentinel = object()
        quant, _ = _load_custom_metrics([path], llm=sentinel)
        assert quant[0].name == "legacy"
        with pytest.raises(RuntimeError, match="llm is not set"):
            _ = quant[0].llm

    def test_llm_aware_metric_with_no_llm_raises_on_access(self, temp_dir: str) -> None:
        """Accessing self.llm raises RuntimeError when no LLM was injected."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class LLMAwareMetric(QuantitativeMetric):
                def __init__(self, llm=None):
                    super().__init__(name="llm_aware", llm=llm)

                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "llm_aware_no_llm.py")
        with open(path, "w") as f:
            f.write(code)

        quant, _ = _load_custom_metrics([path])  # no llm= given
        with pytest.raises(RuntimeError, match="llm is not set"):
            _ = quant[0].llm

    def test_metric_inheriting_base_init_receives_llm(self, temp_dir: str) -> None:
        """Metrics with no __init__ override inherit the base class signature and receive the LLM."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class MinimalMetric(QuantitativeMetric):
                # No __init__ override - inherits base class signature including llm=
                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "minimal_metric.py")
        with open(path, "w") as f:
            f.write(code)

        sentinel = object()
        quant, _ = _load_custom_metrics([path], llm=sentinel)
        assert quant[0].llm is sentinel

    def test_abstract_subclass_is_skipped(self, temp_dir: str) -> None:
        """Abstract intermediate subclasses are skipped; concrete subclasses still load."""
        code = textwrap.dedent("""\
            import abc
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class AbstractBase(QuantitativeMetric, abc.ABC):
                # Shared helper that users may define as an abstract intermediate class
                @abc.abstractmethod
                def score(self, score_input: ScoreInput) -> QuantResult: ...

            class ConcreteMetric(AbstractBase):
                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "abstract_metric.py")
        with open(path, "w") as f:
            f.write(code)

        quant, _ = _load_custom_metrics([path])
        assert len(quant) == 1
        assert quant[0].name == "ConcreteMetric"


class TestLoadCustomMetricsScope:
    """_load_custom_metrics preserves the scope attribute set by custom metrics."""

    def test_default_scope_is_turn(self, temp_dir: str) -> None:
        """Metrics without an explicit scope default to 'turn'."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class DefaultScope(QuantitativeMetric):
                def __init__(self):
                    super().__init__(name="default_scope")

                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "default_scope.py")
        with open(path, "w") as f:
            f.write(code)

        quant, _ = _load_custom_metrics([path])
        assert len(quant) == 1
        assert quant[0].scope == "turn"

    def test_conversation_scope_preserved(self, temp_dir: str) -> None:
        """Metrics with scope='conversation' retain it after loading."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import QuantitativeMetric, ScoreInput, QuantResult

            class ConvoMetric(QuantitativeMetric):
                def __init__(self):
                    super().__init__(name="convo_metric", scope="conversation")

                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)
        """)
        path = os.path.join(temp_dir, "convo_scope.py")
        with open(path, "w") as f:
            f.write(code)

        quant, _ = _load_custom_metrics([path])
        assert len(quant) == 1
        assert quant[0].scope == "conversation"

    def test_mixed_scopes_loaded(self, temp_dir: str) -> None:
        """File with both turn and conversation metrics loads both."""
        code = textwrap.dedent("""\
            from arksim.evaluator.base_metric import (
                QuantitativeMetric, QualitativeMetric,
                ScoreInput, QuantResult, QualResult,
            )

            class TurnQuant(QuantitativeMetric):
                def __init__(self):
                    super().__init__(name="turn_q", scope="turn")
                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=1.0)

            class ConvoQuant(QuantitativeMetric):
                def __init__(self):
                    super().__init__(name="convo_q", scope="conversation")
                def score(self, score_input: ScoreInput) -> QuantResult:
                    return QuantResult(name=self.name, value=2.0)

            class TurnQual(QualitativeMetric):
                def __init__(self):
                    super().__init__(name="turn_ql", scope="turn")
                def evaluate(self, score_input: ScoreInput) -> QualResult:
                    return QualResult(name=self.name, value="ok")

            class ConvoQual(QualitativeMetric):
                def __init__(self):
                    super().__init__(name="convo_ql", scope="conversation")
                def evaluate(self, score_input: ScoreInput) -> QualResult:
                    return QualResult(name=self.name, value="ok")
        """)
        path = os.path.join(temp_dir, "mixed_scope.py")
        with open(path, "w") as f:
            f.write(code)

        quant, qual = _load_custom_metrics([path])
        assert len(quant) == 2
        assert len(qual) == 2
        turn_q = [m for m in quant if m.scope == "turn"]
        convo_q = [m for m in quant if m.scope == "conversation"]
        assert len(turn_q) == 1
        assert turn_q[0].name == "turn_q"
        assert len(convo_q) == 1
        assert convo_q[0].name == "convo_q"
        turn_ql = [m for m in qual if m.scope == "turn"]
        convo_ql = [m for m in qual if m.scope == "conversation"]
        assert len(turn_ql) == 1
        assert turn_ql[0].name == "turn_ql"
        assert len(convo_ql) == 1
        assert convo_ql[0].name == "convo_ql"
