# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.trajectory_matching."""

from __future__ import annotations

from arksim.evaluator.trajectory_matching import match_trajectory
from arksim.scenario.entities import ExpectedToolCall
from arksim.simulation_engine.tool_types import ToolCall


def _tc(name: str, **kwargs: object) -> ToolCall:
    """Helper to build a ToolCall with sensible defaults."""
    return ToolCall(id="id", name=name, arguments=dict(kwargs))


def _etc(
    name: str,
    arg_match_mode: str = "ignore",
    **kwargs: object,
) -> ExpectedToolCall:
    """Helper to build an ExpectedToolCall."""
    return ExpectedToolCall(
        name=name, arguments=dict(kwargs), arg_match_mode=arg_match_mode
    )


# ── Strict mode ──


class TestStrictMode:
    def test_exact_order_match(self) -> None:
        actual = [_tc("get_order"), _tc("cancel_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is True

    def test_order_mismatch(self) -> None:
        actual = [_tc("cancel_order"), _tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.failure_label == "disobey user request"
        assert result.ordering_issues

    def test_count_mismatch_extra_same_name(self) -> None:
        """Extra call with a name in the expected set -> repetition."""
        actual = [_tc("get_order"), _tc("cancel_order"), _tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.extra_calls == ["get_order"]
        assert result.failure_label == "repetition"

    def test_count_mismatch_extra_different_name(self) -> None:
        """Extra call with a name NOT in expected set -> disobey user request."""
        actual = [_tc("get_order"), _tc("cancel_order"), _tc("search_products")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.extra_calls == ["search_products"]
        assert result.failure_label == "disobey user request"

    def test_substitution_not_ordering(self) -> None:
        """Same length but a tool not in expected set is a substitution, not ordering."""
        actual = [_tc("get_order"), _tc("search_products")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.failure_label == "disobey user request"
        # Should report "Wrong tool called", not "ordering mismatch"
        assert "Wrong tool called" in result.reason
        assert not result.ordering_issues

    def test_ordering_not_lost_on_arg_mismatch(self) -> None:
        """Ordering issues at earlier positions aren't dropped by a later arg mismatch."""
        actual = [
            _tc("cancel_order"),
            _tc("get_order", order_id="wrong"),
            _tc("search_products", query="laptop"),
        ]
        expected = [
            _etc("get_order"),
            _etc("cancel_order"),
            _etc("search_products", arg_match_mode="exact", query="shoes"),
        ]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        # Both the ordering issues AND the arg mismatch should be reported
        assert result.ordering_issues
        assert "argument mismatch" in result.reason
        # Ordering takes precedence for the failure label
        assert result.failure_label == "disobey user request"

    def test_count_mismatch_missing(self) -> None:
        actual = [_tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.missing_calls == ["cancel_order"]
        assert result.failure_label == "disobey user request"


# ── Unordered mode ──


class TestUnorderedMode:
    def test_same_set_any_order(self) -> None:
        actual = [_tc("cancel_order"), _tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is True

    def test_missing_call(self) -> None:
        actual = [_tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False
        assert result.failure_label == "disobey user request"
        assert "cancel_order" in result.missing_calls

    def test_extra_call_same_name(self) -> None:
        """Redundant duplicate of an expected call -> repetition."""
        actual = [_tc("get_order"), _tc("cancel_order"), _tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False
        assert result.failure_label == "repetition"
        assert "get_order" in result.extra_calls

    def test_extra_call_different_name(self) -> None:
        """Unrelated extra call -> disobey user request."""
        actual = [_tc("get_order"), _tc("cancel_order"), _tc("search_products")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False
        assert result.failure_label == "disobey user request"
        assert "search_products" in result.extra_calls

    def test_missing_and_extra(self) -> None:
        actual = [_tc("search_products")]
        expected = [_etc("get_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False
        assert result.failure_label == "disobey user request"
        assert "get_order" in result.missing_calls
        assert "search_products" in result.extra_calls


# ── contains mode ──


class TestContainsMode:
    def test_expected_subset_of_actual(self) -> None:
        actual = [_tc("get_customer"), _tc("get_order"), _tc("cancel_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "contains")
        assert result.matched is True

    def test_missing_from_contains(self) -> None:
        actual = [_tc("get_customer")]
        expected = [_etc("get_order")]
        result = match_trajectory(actual, expected, "contains")
        assert result.matched is False
        assert "get_order" in result.missing_calls


# ── within mode ──


class TestWithinMode:
    def test_actual_subset_of_expected(self) -> None:
        actual = [_tc("get_order")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "within")
        assert result.matched is True

    def test_unexpected_call_unrelated(self) -> None:
        """Tool not in expected set -> disobey user request."""
        actual = [_tc("get_order"), _tc("search_products")]
        expected = [_etc("get_order"), _etc("cancel_order")]
        result = match_trajectory(actual, expected, "within")
        assert result.matched is False
        assert result.failure_label == "disobey user request"
        assert "search_products" in result.extra_calls

    def test_duplicate_of_expected_call_allowed(self) -> None:
        """Calling an expected tool twice is OK in within mode (still within the set)."""
        actual = [_tc("get_order"), _tc("get_order")]
        expected = [_etc("get_order")]
        result = match_trajectory(actual, expected, "within")
        assert result.matched is True


# ── Argument matching ──


class TestArgumentMatching:
    def test_exact_args_match(self) -> None:
        actual = [_tc("get_order", order_id="123")]
        expected = [_etc("get_order", arg_match_mode="exact", order_id="123")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is True

    def test_exact_args_mismatch(self) -> None:
        actual = [_tc("get_order", order_id="456")]
        expected = [_etc("get_order", arg_match_mode="exact", order_id="123")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False

    def test_ignore_args(self) -> None:
        actual = [_tc("get_order", order_id="anything")]
        expected = [_etc("get_order", arg_match_mode="ignore")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is True

    def test_partial_args_match(self) -> None:
        actual = [_tc("get_order", order_id="123", verbose=True)]
        expected = [_etc("get_order", arg_match_mode="partial", order_id="123")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is True

    def test_partial_args_missing_key(self) -> None:
        actual = [_tc("get_order", verbose=True)]
        expected = [_etc("get_order", arg_match_mode="partial", order_id="123")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False

    def test_strict_mode_arg_mismatch(self) -> None:
        actual = [_tc("get_order", order_id="wrong")]
        expected = [_etc("get_order", arg_match_mode="exact", order_id="123")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.failure_label == "false information"

    def test_strict_mode_partial_arg_mismatch(self) -> None:
        actual = [_tc("get_order")]
        expected = [_etc("get_order", arg_match_mode="partial", order_id="123")]
        result = match_trajectory(actual, expected, "strict")
        assert result.matched is False
        assert result.failure_label == "lack of specific information"


# ── Failure label mapping ──


class TestFailureLabels:
    def test_missing_maps_to_disobey(self) -> None:
        result = match_trajectory(
            [_tc("get_order")],
            [_etc("get_order"), _etc("cancel_order")],
            "unordered",
        )
        assert result.failure_label == "disobey user request"

    def test_extra_same_name_maps_to_repetition(self) -> None:
        result = match_trajectory(
            [_tc("get_order"), _tc("get_order")],
            [_etc("get_order")],
            "unordered",
        )
        assert result.failure_label == "repetition"

    def test_extra_different_name_maps_to_disobey(self) -> None:
        result = match_trajectory(
            [_tc("get_order"), _tc("search_products")],
            [_etc("get_order")],
            "unordered",
        )
        assert result.failure_label == "disobey user request"

    def test_wrong_order_maps_to_disobey(self) -> None:
        result = match_trajectory(
            [_tc("cancel_order"), _tc("get_order")],
            [_etc("get_order"), _etc("cancel_order")],
            "strict",
        )
        assert result.failure_label == "disobey user request"


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_expected_passes(self) -> None:
        result = match_trajectory([_tc("get_order")], [], "strict")
        assert result.matched is True

    def test_empty_actual_fails(self) -> None:
        result = match_trajectory([], [_etc("get_order")], "strict")
        assert result.matched is False
        assert result.failure_label == "disobey user request"

    def test_empty_actual_within_mode_passes(self) -> None:
        """within mode allows skipping all tools, so zero calls is valid."""
        result = match_trajectory([], [_etc("get_order")], "within")
        assert result.matched is True

    def test_empty_actual_contains_mode_fails(self) -> None:
        """contains mode requires at least the expected tools."""
        result = match_trajectory([], [_etc("get_order")], "contains")
        assert result.matched is False
        assert result.failure_label == "disobey user request"

    def test_duplicate_expected_calls(self) -> None:
        actual = [_tc("get_order"), _tc("get_order")]
        expected = [_etc("get_order"), _etc("get_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is True

    def test_duplicate_expected_but_only_one_actual(self) -> None:
        actual = [_tc("get_order")]
        expected = [_etc("get_order"), _etc("get_order")]
        result = match_trajectory(actual, expected, "unordered")
        assert result.matched is False
        assert "get_order" in result.missing_calls


# ── Conversation-level integration (evaluator) ──


class TestConversationLevelTrajectory:
    """Test that trajectory matching runs at the conversation level in the evaluator."""

    def test_cross_turn_tool_calls_aggregated(self) -> None:
        """Tool calls spread across turns are aggregated and match correctly."""
        from arksim.evaluator.entities import EvaluationParams, TurnEvaluation
        from arksim.evaluator.evaluator import Evaluator
        from arksim.evaluator.utils.enums import EvaluationOutcomes
        from arksim.scenario.entities import Scenario, Scenarios
        from arksim.simulation_engine.entities import (
            Conversation,
            Message,
            SimulatedUserPrompt,
        )

        scenarios = Scenarios(
            schema_version="v1",
            scenarios=[
                Scenario(
                    scenario_id="cancel_order_test",
                    user_id="u1",
                    goal="Cancel order",
                    agent_context="ctx",
                    user_profile="profile",
                    expected_tool_calls=[
                        _etc("get_order"),
                        _etc("cancel_order"),
                    ],
                    match_mode="strict",
                ),
            ],
        )

        evaluator = Evaluator(
            params=EvaluationParams(output_dir="/tmp"),
            scenarios=scenarios,
        )

        conversations = [
            Conversation(
                conversation_id="c1",
                scenario_id="cancel_order_test",
                conversation_history=[
                    Message(
                        turn_id=0, role="simulated_user", content="cancel my order"
                    ),
                    Message(
                        turn_id=0,
                        role="assistant",
                        content="let me check",
                        tool_calls=[_tc("get_order")],
                    ),
                    Message(turn_id=1, role="simulated_user", content="yes cancel it"),
                    Message(
                        turn_id=1,
                        role="assistant",
                        content="done",
                        tool_calls=[_tc("cancel_order")],
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="",
                    variables={},
                ),
            ),
        ]

        # Build TurnItems via _process_input
        convo_list, convo_item = evaluator._process_input(conversations[0])
        processed_entries = [(convo_list, convo_item)]

        # Create fake turn evaluations (no failures)
        turn_results = {
            "c1": [
                TurnEvaluation(
                    turn_id=0,
                    scores=[],
                    turn_score=-1,
                    turn_behavior_failure=EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
                    turn_behavior_failure_reason="",
                ),
                TurnEvaluation(
                    turn_id=1,
                    scores=[],
                    turn_score=-1,
                    turn_behavior_failure=EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
                    turn_behavior_failure_reason="",
                ),
            ],
        }

        evaluator._apply_trajectory_matching(
            conversations, processed_entries, turn_results
        )

        # Should pass: aggregated [get_order, cancel_order] matches strict order
        for turn in turn_results["c1"]:
            traj_quals = [
                q
                for q in turn.qual_scores
                if q.reason and q.reason.startswith("[Trajectory]")
            ]
            assert len(traj_quals) == 0

    def test_cross_turn_wrong_order_detected(self) -> None:
        """Wrong order across turns is caught by conversation-level matching."""
        from arksim.evaluator.entities import EvaluationParams, TurnEvaluation
        from arksim.evaluator.evaluator import Evaluator
        from arksim.evaluator.utils.enums import EvaluationOutcomes
        from arksim.scenario.entities import Scenario, Scenarios
        from arksim.simulation_engine.entities import (
            Conversation,
            Message,
            SimulatedUserPrompt,
        )

        scenarios = Scenarios(
            schema_version="v1",
            scenarios=[
                Scenario(
                    scenario_id="cancel_order_test",
                    user_id="u1",
                    goal="Cancel order",
                    agent_context="ctx",
                    user_profile="profile",
                    expected_tool_calls=[
                        _etc("get_order"),
                        _etc("cancel_order"),
                    ],
                    match_mode="strict",
                ),
            ],
        )

        evaluator = Evaluator(
            params=EvaluationParams(output_dir="/tmp"),
            scenarios=scenarios,
        )

        # Agent calls cancel_order BEFORE get_order (wrong order)
        conversations = [
            Conversation(
                conversation_id="c1",
                scenario_id="cancel_order_test",
                conversation_history=[
                    Message(
                        turn_id=0, role="simulated_user", content="cancel my order"
                    ),
                    Message(
                        turn_id=0,
                        role="assistant",
                        content="cancelled",
                        tool_calls=[_tc("cancel_order")],
                    ),
                    Message(turn_id=1, role="simulated_user", content="wait what"),
                    Message(
                        turn_id=1,
                        role="assistant",
                        content="let me check",
                        tool_calls=[_tc("get_order")],
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="",
                    variables={},
                ),
            ),
        ]

        convo_list, convo_item = evaluator._process_input(conversations[0])
        processed_entries = [(convo_list, convo_item)]

        turn_results = {
            "c1": [
                TurnEvaluation(
                    turn_id=0,
                    scores=[],
                    turn_score=-1,
                    turn_behavior_failure=EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
                    turn_behavior_failure_reason="",
                ),
                TurnEvaluation(
                    turn_id=1,
                    scores=[],
                    turn_score=-1,
                    turn_behavior_failure=EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
                    turn_behavior_failure_reason="",
                ),
            ],
        }

        evaluator._apply_trajectory_matching(
            conversations, processed_entries, turn_results
        )

        # Failure attributed to last turn with tool calls (turn 1)
        last_turn = turn_results["c1"][1]
        assert last_turn.turn_behavior_failure == "disobey user request"
        assert "[Trajectory]" in last_turn.turn_behavior_failure_reason

        # Turn 0 should be unaffected
        first_turn = turn_results["c1"][0]
        assert (
            first_turn.turn_behavior_failure
            == EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value
        )

    def test_no_expected_calls_skips_matching(self) -> None:
        """Conversations without expected_tool_calls are not checked."""
        from arksim.evaluator.entities import EvaluationParams, TurnEvaluation
        from arksim.evaluator.evaluator import Evaluator
        from arksim.evaluator.utils.enums import EvaluationOutcomes
        from arksim.scenario.entities import Scenario, Scenarios
        from arksim.simulation_engine.entities import (
            Conversation,
            Message,
            SimulatedUserPrompt,
        )

        scenarios = Scenarios(
            schema_version="v1",
            scenarios=[
                Scenario(
                    scenario_id="no_expectations",
                    user_id="u1",
                    goal="Check order",
                    agent_context="ctx",
                    user_profile="profile",
                    # No expected_tool_calls
                ),
            ],
        )

        evaluator = Evaluator(
            params=EvaluationParams(output_dir="/tmp"),
            scenarios=scenarios,
        )

        conversations = [
            Conversation(
                conversation_id="c1",
                scenario_id="no_expectations",
                conversation_history=[
                    Message(turn_id=0, role="simulated_user", content="hi"),
                    Message(
                        turn_id=0,
                        role="assistant",
                        content="hello",
                        tool_calls=[_tc("get_order"), _tc("random_tool")],
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="",
                    variables={},
                ),
            ),
        ]

        convo_list, convo_item = evaluator._process_input(conversations[0])
        processed_entries = [(convo_list, convo_item)]

        turn_results = {
            "c1": [
                TurnEvaluation(
                    turn_id=0,
                    scores=[],
                    turn_score=-1,
                    turn_behavior_failure=EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
                    turn_behavior_failure_reason="",
                ),
            ],
        }

        evaluator._apply_trajectory_matching(
            conversations, processed_entries, turn_results
        )

        # No trajectory failures since no expected_tool_calls defined
        turn = turn_results["c1"][0]
        assert (
            turn.turn_behavior_failure
            == EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value
        )

    def test_no_tool_calls_still_fails(self) -> None:
        """Agent makes no tool calls but expected_tool_calls is defined."""
        from arksim.evaluator.entities import EvaluationParams, TurnEvaluation
        from arksim.evaluator.evaluator import Evaluator
        from arksim.evaluator.utils.enums import EvaluationOutcomes
        from arksim.scenario.entities import Scenario, Scenarios
        from arksim.simulation_engine.entities import (
            Conversation,
            Message,
            SimulatedUserPrompt,
        )

        scenarios = Scenarios(
            schema_version="v1",
            scenarios=[
                Scenario(
                    scenario_id="cancel_order_test",
                    user_id="u1",
                    goal="Cancel order",
                    agent_context="ctx",
                    user_profile="profile",
                    expected_tool_calls=[
                        _etc("get_order"),
                        _etc("cancel_order"),
                    ],
                    match_mode="unordered",
                ),
            ],
        )

        evaluator = Evaluator(
            params=EvaluationParams(output_dir="/tmp"),
            scenarios=scenarios,
        )

        # Agent responds without making any tool calls
        conversations = [
            Conversation(
                conversation_id="c1",
                scenario_id="cancel_order_test",
                conversation_history=[
                    Message(
                        turn_id=0, role="simulated_user", content="cancel my order"
                    ),
                    Message(
                        turn_id=0,
                        role="assistant",
                        content="Sorry, I cannot do that.",
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="",
                    variables={},
                ),
            ),
        ]

        convo_list, convo_item = evaluator._process_input(conversations[0])
        processed_entries = [(convo_list, convo_item)]

        turn_results = {
            "c1": [
                TurnEvaluation(
                    turn_id=0,
                    scores=[],
                    turn_score=-1,
                    turn_behavior_failure=EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
                    turn_behavior_failure_reason="",
                ),
            ],
        }

        evaluator._apply_trajectory_matching(
            conversations, processed_entries, turn_results
        )

        # Failure should be attributed to the last (only) turn
        turn = turn_results["c1"][0]
        assert turn.turn_behavior_failure == "disobey user request"
        assert "[Trajectory]" in turn.turn_behavior_failure_reason
