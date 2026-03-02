"""Tests for evaluator entities and enums."""

import pytest
from pydantic import ValidationError

from arksim.evaluator import ChatMessage, QuantResult
from arksim.evaluator.entities import (
    ConversationEvaluation,
    ConvoItem,
    EvaluationParams,
    Occurrence,
    TurnEvaluation,
    TurnItem,
    UniqueError,
)
from arksim.evaluator.utils.enums import (
    AgentBehaviorFailureType,
    AgentMetrics,
    EvaluationStatus,
)


class TestEvaluationStatus:
    """Tests for EvaluationStatus enum."""

    def test_evaluation_failed(self) -> None:
        """Test EVALUATION_FAILED status."""
        assert EvaluationStatus.EVALUATION_FAILED.value == "Evaluation Failed"

    def test_done(self) -> None:
        """Test DONE status."""
        assert EvaluationStatus.DONE.value == "Done"

    def test_partial_failure(self) -> None:
        """Test PARTIAL_FAILURE status."""
        assert EvaluationStatus.PARTIAL_FAILURE.value == "Partial Failure"

    def test_failed(self) -> None:
        """Test FAILED status."""
        assert EvaluationStatus.FAILED.value == "Failed"

    def test_all_statuses_defined(self) -> None:
        """Test all expected statuses are defined."""
        statuses = [s.value for s in EvaluationStatus]
        assert len(statuses) == 4


class TestAgentMetrics:
    """Tests for AgentMetrics enum."""

    def test_helpfulness(self) -> None:
        """Test HELPFULNESS metric."""
        assert AgentMetrics.HELPFULNESS.value == "helpfulness"

    def test_coherence(self) -> None:
        """Test COHERENCE metric."""
        assert AgentMetrics.COHERENCE.value == "coherence"

    def test_all_metrics_defined(self) -> None:
        """Test all expected metrics are defined."""
        metrics = [m.value for m in AgentMetrics]
        expected = [
            "helpfulness",
            "coherence",
            "verbosity",
            "relevance",
            "faithfulness",
            "goal_completion",
            "agent_behavior_failure",
            "unique_bugs",
        ]
        assert set(metrics) == set(expected)


class TestAgentBehaviorFailureType:
    """Tests for AgentBehaviorFailureType enum."""

    def test_lack_of_specific_information(self) -> None:
        """Test LACK_OF_SPECIFIC_INFORMATION type."""
        assert (
            AgentBehaviorFailureType.LACK_OF_SPECIFIC_INFORMATION.value
            == "lack of specific information"
        )

    def test_no_failure(self) -> None:
        """Test NO_FAILURE type."""
        assert AgentBehaviorFailureType.NO_FAILURE.value == "no failure"

    def test_all_failure_types_defined(self) -> None:
        """Test all expected failure types are defined."""
        types = [t.value for t in AgentBehaviorFailureType]
        assert len(types) == 6


class TestEvaluationParams:
    """Tests for EvaluationParams model."""

    def test_valid_config(self) -> None:
        """Test creating valid EvaluationParams."""
        config = EvaluationParams(
            output_dir="/output",
            agent_name="TestAgent",
            num_workers=4,
        )

        assert config.output_dir == "/output"
        assert config.agent_name == "TestAgent"
        assert config.num_workers == 4

    def test_default_agent_name(self) -> None:
        """Test default agent_name is 'Agent'."""
        config = EvaluationParams(output_dir="/output", num_workers=1)

        assert config.agent_name == "Agent"

    def test_optional_code_file_path(self) -> None:
        """Test code_file_path is optional."""
        config = EvaluationParams(
            output_dir="/output",
            num_workers=1,
            code_file_path="/path/to/code.py",
        )

        assert config.code_file_path == "/path/to/code.py"

    def test_requires_output_dir(self) -> None:
        """Test output_dir is required."""
        with pytest.raises(ValidationError):
            EvaluationParams(num_workers=1)


class TestTurnItem:
    """Tests for TurnItem model."""

    def test_valid_turn_item(self) -> None:
        """Test creating valid TurnItem."""
        turn_msgs = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi!"),
        ]
        item = TurnItem(
            chat_id="chat-123",
            turn_id=1,
            current_turn=turn_msgs,
            conversation_history=turn_msgs,
            system_prompt="You are helpful.",
            knowledge=["Knowledge base"],
            profile="User profile",
            user_goal="Get information",
        )

        assert item.chat_id == "chat-123"
        assert item.turn_id == 1


class TestConvoItem:
    """Tests for ConvoItem model."""

    def test_valid_convo_item(self) -> None:
        """Test creating valid ConvoItem."""
        item = ConvoItem(
            chat_id="chat-123",
            chat_history=[ChatMessage(role="user", content="Full history")],
            system_prompt="prompt",
            knowledge=["knowledge"],
            profile="profile",
            user_goal="goal",
            turns=5,
        )

        assert item.chat_id == "chat-123"
        assert item.turns == 5


class TestScore:
    """Tests for QuantResult model (stored quantitative result)."""

    def test_valid_score(self) -> None:
        """Test creating a valid QuantResult."""
        s = QuantResult(name="helpfulness", value=4.0, reason="Very helpful")
        assert s.name == "helpfulness"
        assert s.value == 4.0
        assert s.reason == "Very helpful"

    def test_various_metrics(self) -> None:
        """Test QuantResult works with all standard metric names."""
        for metric in (
            "helpfulness",
            "coherence",
            "verbosity",
            "relevance",
            "faithfulness",
        ):
            s = QuantResult(name=metric, value=3.0, reason="OK")
            assert s.name == metric


class TestTurnEvaluation:
    """Tests for TurnEvaluation model."""

    def test_valid_turn_evaluation(self) -> None:
        """Test creating a valid TurnEvaluation."""
        item = TurnEvaluation(
            turn_id=1,
            scores=[QuantResult(name="helpfulness", value=4.0, reason="Good")],
            turn_score=4.0,
            turn_behavior_failure="no failure",
            turn_behavior_failure_reason="All good",
        )

        assert item.turn_id == 1
        assert item.turn_score == 4.0
        assert len(item.scores) == 1

    def test_default_unique_error_ids(self) -> None:
        """Test unique_error_ids defaults to empty list."""
        item = TurnEvaluation(
            turn_id=0,
            scores=[],
            turn_score=-1,
            turn_behavior_failure="no failure",
            turn_behavior_failure_reason="",
        )
        assert item.unique_error_ids == []

    def test_multiple_scores(self) -> None:
        """Test TurnEvaluation holds multiple metric scores."""
        scores = [
            QuantResult(name="helpfulness", value=4.0, reason="Good"),
            QuantResult(name="coherence", value=3.5, reason="OK"),
        ]
        item = TurnEvaluation(
            turn_id=2,
            scores=scores,
            turn_score=3.75,
            turn_behavior_failure="no failure",
            turn_behavior_failure_reason="",
        )
        assert len(item.scores) == 2


class TestConversationEvaluation:
    """Tests for ConversationEvaluation model."""

    def test_valid_conversation_evaluation(self) -> None:
        """Test creating a valid ConversationEvaluation."""
        item = ConversationEvaluation(
            conversation_id="conv-1",
            goal_completion_score=0.9,
            goal_completion_reason="Goal achieved",
            turn_success_ratio=0.8,
            overall_agent_score=0.87,
            evaluation_status="Done",
            turn_scores=[],
        )

        assert item.conversation_id == "conv-1"
        assert item.overall_agent_score == 0.87
        assert item.evaluation_status == "Done"

    def test_with_turn_scores(self) -> None:
        """Test ConversationEvaluation holds nested TurnEvaluation list."""
        turn = TurnEvaluation(
            turn_id=0,
            scores=[QuantResult(name="helpfulness", value=4.0, reason="Good")],
            turn_score=4.0,
            turn_behavior_failure="no failure",
            turn_behavior_failure_reason="",
        )
        item = ConversationEvaluation(
            conversation_id="conv-2",
            goal_completion_score=0.5,
            goal_completion_reason="Partial",
            turn_success_ratio=0.5,
            overall_agent_score=0.5,
            evaluation_status="Partial Failure",
            turn_scores=[turn],
        )
        assert len(item.turn_scores) == 1


class TestOccurrence:
    """Tests for Occurrence model."""

    def test_valid_occurrence(self) -> None:
        """Test creating a valid Occurrence."""
        occ = Occurrence(conversation_id="conv-1", turn_id=2)
        assert occ.conversation_id == "conv-1"
        assert occ.turn_id == 2


class TestUniqueError:
    """Tests for UniqueError model."""

    def test_valid_unique_error(self) -> None:
        """Test creating a valid UniqueError."""
        item = UniqueError(
            unique_error_id="uid-1",
            behavior_failure_category="repetition",
            unique_error_description="Agent repeats same response",
            occurrences=[Occurrence(conversation_id="c1", turn_id=0)],
        )

        assert item.behavior_failure_category == "repetition"
        assert len(item.occurrences) == 1

    def test_no_fix_fields(self) -> None:
        """Test that UniqueError has no fix fields (pro-only)."""
        item = UniqueError(
            unique_error_id="uid-1",
            behavior_failure_category="test",
            unique_error_description="desc",
            occurrences=[],
        )

        assert not hasattr(item, "module_fixes")
        assert not hasattr(item, "best_module_fix")
        assert not hasattr(item, "code_fixes")

    def test_default_severity(self) -> None:
        """Test default severity is 'medium'."""
        item = UniqueError(
            unique_error_id="uid-1",
            behavior_failure_category="test",
            unique_error_description="desc",
            occurrences=[],
        )
        assert item.severity == "medium"

    def test_custom_severity(self) -> None:
        """Test custom severity values."""
        for sev in ("critical", "high", "medium", "low"):
            item = UniqueError(
                unique_error_id="uid-1",
                behavior_failure_category="test",
                unique_error_description="desc",
                severity=sev,
                occurrences=[],
            )
            assert item.severity == sev


