# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from arksim.evaluator.assertions import (
    AssertionResult,
    ToolEvidence,
    _BehaviorVerdict,
    _BehaviorVerdicts,
    assertion_label,
    evaluate_assertions,
    evaluate_text_match,
    evaluate_tool_calls,
    expected_behavior_text,
    overall_result,
    summarize_results,
)
from arksim.evaluator.base_metric import ChatMessage
from arksim.evaluator.entities import TurnItem
from arksim.scenario.entities import (
    Assertion,
    ExpectedToolCall,
    LLMJudgeAssertion,
    Scenario,
    TextMatchAssertion,
    ToolCallsAssertion,
)
from arksim.simulation_engine.tool_types import ToolCall, ToolCallCaptureQuality

_parse = TypeAdapter(Assertion).validate_python

CHAT = [
    ChatMessage(role="user", content="Cancel ORD-1001 please"),
    ChatMessage(role="assistant", content="ORD-1001 has shipped, tracking 1ZORD1001."),
    ChatMessage(role="user", content="Fine."),
    ChatMessage(role="assistant", content="Anything else?"),
]
CALLS = [
    ToolCall(id="1", name="get_order", arguments={"order_id": "ORD-1001"}),
    ToolCall(id="2", name="get_delivery_status", arguments={"order_id": "ORD-1001"}),
]


def _turns(calls: list[ToolCall] | tuple = ()) -> list[TurnItem]:
    """CHAT as two turns, with ``calls`` made in the first."""

    def turn(i: int, calls: list[ToolCall] | None) -> TurnItem:
        return TurnItem(
            chat_id="c",
            turn_id=i,
            current_turn=CHAT[2 * i : 2 * i + 2],
            conversation_history=CHAT[: 2 * i + 2],
            system_prompt="",
            knowledge=[],
            profile="",
            user_goal="",
            tool_calls=calls,
        )

    return [turn(0, list(calls)), turn(1, None)]


def _scenario(assertions: list[dict]) -> Scenario:
    return Scenario(
        scenario_id="s",
        user_id="u",
        goal="g",
        agent_context="a",
        user_profile="p",
        assertions=assertions,
    )


# ── Schema ──


def test_union_parses_every_type() -> None:
    assert isinstance(
        _parse({"type": "llm_judge", "expected": "Declines"}),
        LLMJudgeAssertion,
    )
    assert isinstance(
        _parse({"type": "text_match", "op": "contains", "value": "x"}),
        TextMatchAssertion,
    )
    tc = _parse({"type": "tool_calls", "forbidden": [{"name": "cancel_order"}]})
    assert isinstance(tc, ToolCallsAssertion)
    assert tc.expected == []


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "llm_judge", "expected": ""},
        {"type": "text_match", "op": "contains", "value": ""},
        {"type": "text_match", "op": "equals", "value": "x"},
        {"type": "text_match", "op": "matches", "value": "("},
        {"type": "unknown"},
    ],
)
def test_invalid_assertions_rejected(raw: dict) -> None:
    with pytest.raises(ValidationError):
        _parse(raw)


def test_at_most_one_tool_calls_assertion() -> None:
    with pytest.raises(ValidationError):
        _scenario(
            [
                {"type": "tool_calls", "expected": [{"name": "a"}]},
                {"type": "tool_calls", "expected": [{"name": "b"}]},
            ]
        )


def test_existing_tool_calls_scenario_still_loads() -> None:
    s = _scenario([{"type": "tool_calls", "expected": [{"name": "get_order"}]}])
    assert s.find_assertion("tool_calls").expected[0].name == "get_order"


# ── Reply text ──


@pytest.mark.parametrize(
    ("op", "value", "status"),
    [
        ("contains", "1zord1001", "pass"),  # case is ignored
        ("contains", "1ZORD1001", "pass"),
        ("not_contains", "123456", "pass"),
        ("not_contains", "SHIPPED", "fail"),
        ("matches", r"1z[a-z]+\d+", "pass"),
        ("matches", r"refund of \$\d+", "fail"),
        ("contains", "a.b", "fail"),  # literal, not regex
    ],
)
def test_text_match(op: str, value: str, status: str) -> None:
    a = TextMatchAssertion(type="text_match", op=op, value=value)
    replies = [m.content for m in CHAT if m.role == "assistant"]
    assert evaluate_text_match(a, replies).status == status


def test_text_match_matches_in_linear_time() -> None:
    # Exponential under backtracking: seconds at 26 characters with re.
    a = TextMatchAssertion(type="text_match", op="matches", value="(a+)+$")
    assert evaluate_text_match(a, ["a" * 5000 + "b"]).status == "fail"


@pytest.mark.parametrize(
    ("value", "valid"),
    [(r"\p{L}+", True), (r"\Qa.b\E", True), (r"(?=x)", False), (r"(a)\1", False)],
)
def test_text_match_pattern_dialect_is_re2(value: str, valid: bool) -> None:
    # The backend validates with RE2, so the same patterns load here.
    if valid:
        TextMatchAssertion(type="text_match", op="matches", value=value)
    else:
        with pytest.raises(ValidationError):
            TextMatchAssertion(type="text_match", op="matches", value=value)


def test_text_match_ignores_user_messages() -> None:
    a = TextMatchAssertion(type="text_match", op="contains", value="please")
    llm = MagicMock()
    [result] = evaluate_assertions([a], llm=llm, turns=_turns([]))
    assert result.status == "fail"
    llm.call.assert_not_called()


# ── Tool calls ──


def test_tool_calls_expected_and_forbidden() -> None:
    ok = ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name="get_order")],
        match_mode="contains",
        forbidden=[ExpectedToolCall(name="cancel_order")],
    )
    assert evaluate_tool_calls(ok, _turns(CALLS)).status == "pass"

    bad = ok.model_copy(
        update={"forbidden": [ExpectedToolCall(name="get_delivery_status")]}
    )
    result = evaluate_tool_calls(bad, _turns(CALLS))
    assert result.status == "fail"
    assert "get_delivery_status" in result.reason


def test_forbidden_respects_arguments() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        forbidden=[
            ExpectedToolCall(
                name="get_order",
                arguments={"order_id": "ORD-1002"},
                arg_match_mode="partial",
            )
        ],
    )
    assert evaluate_tool_calls(a, _turns(CALLS)).status == "pass"


def test_forbidden_arguments_not_captured_is_inconclusive() -> None:
    # A redacted cancel_order may have been the forbidden one.
    redacted = ToolCall(
        id="3",
        name="cancel_order",
        arguments={},
        capture_quality=ToolCallCaptureQuality(arguments="redacted"),
    )
    a = ToolCallsAssertion(
        type="tool_calls",
        forbidden=[
            ExpectedToolCall(
                name="cancel_order",
                arguments={"order_id": "ORD-1001"},
                arg_match_mode="partial",
            )
        ],
    )
    assert evaluate_tool_calls(a, _turns([*CALLS, redacted])).status == "inconclusive"
    # Without an argument check, the name alone decides.
    by_name = a.model_copy(
        update={"forbidden": [ExpectedToolCall(name="cancel_order")]}
    )
    assert evaluate_tool_calls(by_name, _turns([*CALLS, redacted])).status == "fail"


def test_forbidden_exact_match_on_redacted_arguments_is_inconclusive() -> None:
    # Redacted to {}: equal to an exact {} entry, but not known to be empty.
    redacted = ToolCall(
        id="3",
        name="cancel_order",
        arguments={},
        capture_quality=ToolCallCaptureQuality(arguments="redacted"),
    )
    a = ToolCallsAssertion(
        type="tool_calls",
        forbidden=[
            ExpectedToolCall(name="cancel_order", arguments={}, arg_match_mode="exact")
        ],
    )
    assert evaluate_tool_calls(a, _turns([redacted])).status == "inconclusive"
    # A fully captured matching call still fails.
    seen = redacted.model_copy(update={"capture_quality": None})
    assert evaluate_tool_calls(a, _turns([redacted, seen])).status == "fail"


def test_tool_calls_missing_expected_fails() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name="cancel_order")],
        match_mode="contains",
    )
    assert evaluate_tool_calls(a, _turns(CALLS)).status == "fail"


# ── Behavior (LLM) ──


def _llm(verdicts: list[tuple[int, str]]) -> MagicMock:
    llm = MagicMock()
    llm.call.return_value = _BehaviorVerdicts(
        verdicts=[_BehaviorVerdict(index=i, reason="r", status=s) for i, s in verdicts]
    )
    return llm


# ── Same verdicts as the platform: partial capture is never a pass ──


def _absent_args(call: ToolCall) -> ToolCall:
    return call.model_copy(
        update={
            "arguments": {},
            "capture_quality": ToolCallCaptureQuality(arguments="absent"),
        }
    )


def test_expected_exact_match_on_missing_arguments_is_inconclusive() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[
            ExpectedToolCall(name="get_order", arguments={}, arg_match_mode="exact")
        ],
        match_mode="contains",
    )
    result = evaluate_tool_calls(a, _turns([_absent_args(CALLS[0])]))
    assert result.status == "inconclusive"
    assert "not fully captured" in result.reason


def test_strict_order_on_ambiguous_capture_is_inconclusive() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name=c.name) for c in CALLS],
        match_mode="strict",
    )
    ambiguous = [
        c.model_copy(
            update={"capture_quality": ToolCallCaptureQuality(ordering="ambiguous")}
        )
        for c in CALLS
    ]
    assert evaluate_tool_calls(a, _turns(ambiguous)).status == "inconclusive"
    assert evaluate_tool_calls(a, _turns(CALLS)).status == "pass"


def test_forbidden_without_evidence_is_inconclusive_only_when_told() -> None:
    a = ToolCallsAssertion(
        type="tool_calls", forbidden=[ExpectedToolCall(name="cancel_order")]
    )
    # The agent reported its calls itself: none means none.
    assert evaluate_tool_calls(a, _turns([])).status == "pass"
    nothing = ToolEvidence(observed=False)
    assert evaluate_tool_calls(a, _turns([]), nothing).status == "inconclusive"
    gaps = ToolEvidence(gaps=["unavailable"])
    assert evaluate_tool_calls(a, _turns(CALLS), gaps).status == "inconclusive"


def test_evidence_gaps_give_one_readable_reason() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name="refund")],
        forbidden=[ExpectedToolCall(name="cancel_order")],
        match_mode="contains",
    )
    gaps = ToolEvidence(gaps=["no tool calls captured", "no tool calls captured"])
    result = evaluate_tool_calls(a, _turns(CALLS), gaps)
    assert result.reason == "Some turns could not be checked: no tool calls captured."


_GAPS = ToolEvidence(gaps=["tool calls only partly captured"])


def _expect(names: list[str], mode: str) -> ToolCallsAssertion:
    return ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name=n) for n in names],
        match_mode=mode,
    )


@pytest.mark.parametrize(
    ("names", "mode", "status"),
    [
        # An observed call outside the list fails whatever the gaps hold.
        (["get_order"], "within", "fail"),
        (["get_order"], "unordered", "fail"),
        # Missing calls may be in the gaps.
        (["get_order", "refund"], "contains", "inconclusive"),
        (["get_order", "get_delivery_status", "refund"], "unordered", "inconclusive"),
        (["get_order", "refund", "get_delivery_status"], "strict", "inconclusive"),
        # Unseen calls cannot reorder the observed ones.
        (["get_delivery_status", "refund", "get_order"], "strict", "fail"),
        # Unseen calls cannot undo a contains match.
        (["get_order", "get_delivery_status"], "contains", "pass"),
        # Elsewhere all expected calls present still leaves the gaps unchecked.
        (["get_order", "get_delivery_status"], "within", "inconclusive"),
        (["get_order", "get_delivery_status"], "strict", "inconclusive"),
    ],
)
def test_evidence_gaps_keep_fails_the_observed_calls_prove(
    names: list[str], mode: str, status: str
) -> None:
    result = evaluate_tool_calls(_expect(names, mode), _turns(CALLS), _GAPS)
    assert result.status == status
    if status == "inconclusive":
        assert result.reason.startswith("Some turns could not be checked")


def test_strict_order_fail_under_gaps_says_why() -> None:
    a = _expect(["get_delivery_status", "refund", "get_order"], "strict")
    result = evaluate_tool_calls(a, _turns(CALLS), _GAPS)
    assert result.reason.startswith("Observed calls are out of the expected order.")


def test_disallowed_call_fails_despite_redacted_arguments() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[
            ExpectedToolCall(
                name="get_order",
                arguments={"order_id": "ORD-1001"},
                arg_match_mode="exact",
            )
        ],
        match_mode="within",
    )
    calls = [_absent_args(CALLS[0]), CALLS[1]]
    assert evaluate_tool_calls(a, _turns(calls)).status == "fail"
    assert evaluate_tool_calls(a, _turns(calls[:1])).status == "inconclusive"


def test_captured_wrong_arguments_fail_beside_a_redacted_call() -> None:
    """Review on frontend#1249: a redacted get_order switched argument checks
    off for every get_order call, so a fully captured call to the wrong
    order turned from fail to inconclusive."""
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[
            ExpectedToolCall(
                name="get_order",
                arguments={"order_id": "ORD-1001"},
                arg_match_mode="exact",
            )
        ],
        match_mode="within",
    )
    wrong = CALLS[0].model_copy(update={"arguments": {"order_id": "ORD-2002"}})
    result = evaluate_tool_calls(a, _turns([_absent_args(CALLS[0]), wrong]))
    assert result.status == "fail"


def test_redacted_call_order_does_not_decide_the_verdict() -> None:
    """Regression: greedy pairing spent a redacted get_order on
    the first expected entry, so [redacted, A] failed while [A, redacted] was
    inconclusive."""
    order = ExpectedToolCall(
        name="get_order", arguments={"order_id": "A"}, arg_match_mode="exact"
    )
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[order, order.model_copy(update={"arguments": {"order_id": "B"}})],
        match_mode="unordered",
    )
    seen = CALLS[0].model_copy(update={"arguments": {"order_id": "A"}})
    for calls in ([_absent_args(CALLS[0]), seen], [seen, _absent_args(CALLS[0])]):
        assert evaluate_tool_calls(a, _turns(calls)).status == "inconclusive"


def test_strict_redacted_call_is_inconclusive_not_a_mismatch() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[
            ExpectedToolCall(
                name="get_order",
                arguments={"order_id": "ORD-1001"},
                arg_match_mode="exact",
            )
        ],
        match_mode="strict",
    )
    result = evaluate_tool_calls(a, _turns([_absent_args(CALLS[0])]))
    assert result.status == "inconclusive"


def test_ambiguous_strict_order_with_extra_call_fails() -> None:
    a = _expect(["get_order"], "strict")
    ambiguous = [
        c.model_copy(
            update={"capture_quality": ToolCallCaptureQuality(ordering="ambiguous")}
        )
        for c in CALLS
    ]
    assert evaluate_tool_calls(a, _turns(ambiguous)).status == "fail"
    assert evaluate_tool_calls(a, _turns(ambiguous), _GAPS).status == "fail"


def test_within_reads_as_allowed_tools() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name="get_order")],
        match_mode="within",
    )
    assert assertion_label(a) == "Allowed tools: get_order"


def test_tool_calls_result_carries_its_trajectory() -> None:
    a = ToolCallsAssertion(
        type="tool_calls",
        expected=[ExpectedToolCall(name="get_order"), ExpectedToolCall(name="refund")],
        match_mode="contains",
    )
    result = evaluate_tool_calls(a, _turns(CALLS))
    assert result.status == "fail"
    assert result.trajectory.status == "fail"
    assert [c["turn_id"] for c in result.trajectory.actual] == [0, 0]
    assert result.trajectory.missing == [{"name": "refund"}]


def test_judge_sees_each_tool_call_before_the_reply_it_was_made_for() -> None:
    llm = _llm([(0, "pass")])
    evaluate_assertions(
        [LLMJudgeAssertion(type="llm_judge", expected="Looks up the order first")],
        llm=llm,
        turns=_turns(CALLS),
    )
    prompt = llm.call.call_args.args[0][1]["content"]
    lookup = prompt.index("assistant tool call: get_order")
    assert (
        prompt.index("user: Cancel ORD-1001")
        < lookup
        < prompt.index("assistant: ORD-1001 has shipped")
    )
    assert "assistant tool call" not in prompt[prompt.index("user: Fine.") :]


def test_behavior_checks_share_one_judge_call_and_keep_order() -> None:
    assertions = [
        LLMJudgeAssertion(type="llm_judge", expected="Declines"),
        TextMatchAssertion(type="text_match", op="contains", value="1ZORD1001"),
        LLMJudgeAssertion(type="llm_judge", expected="Explains policy"),
    ]
    llm = _llm([(0, "pass"), (1, "fail")])
    results = evaluate_assertions(
        assertions,
        llm=llm,
        turns=_turns(CALLS),
        agent_context="Store bot",
    )
    assert llm.call.call_count == 1
    assert [r.type for r in results] == [
        "llm_judge",
        "text_match",
        "llm_judge",
    ]
    assert [r.status for r in results] == ["pass", "pass", "fail"]
    prompt = llm.call.call_args.args[0][1]["content"]
    assert "0. Declines" in prompt and "1. Explains policy" in prompt
    assert "get_order" in prompt and "Store bot" in prompt


def test_behavior_missing_verdict_is_inconclusive() -> None:
    a = [LLMJudgeAssertion(type="llm_judge", expected="Declines")]
    [result] = evaluate_assertions(a, llm=_llm([]), turns=_turns([]))
    assert result.status == "inconclusive"


def test_behavior_judge_failure_is_inconclusive() -> None:
    llm = MagicMock()
    llm.call.side_effect = RuntimeError("boom")
    a = [LLMJudgeAssertion(type="llm_judge", expected="Declines")]
    [result] = evaluate_assertions(a, llm=llm, turns=_turns([]))
    assert result.status == "inconclusive"


# ── Result rollup ──


def _r(status: str) -> AssertionResult:
    return AssertionResult(type="text_match", label="x", status=status)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], "no_assertions"),
        (["pass"], "pass"),
        (["pass", "pass"], "pass"),
        (["pass", "inconclusive"], "inconclusive"),
        (["inconclusive", "fail"], "fail"),
        (["pass", "fail"], "fail"),
    ],
)
def test_overall_result(statuses: list[str], expected: str) -> None:
    assert overall_result([_r(s) for s in statuses]) == expected


def test_summarize_results() -> None:
    assert summarize_results(["pass", "fail", "pass", "no_assertions"]) == {
        "passed": 2,
        "failed": 1,
        "inconclusive": 0,
        "no_assertions": 1,
    }


# ── Labels and prompt text ──


def test_labels() -> None:
    assert (
        assertion_label(
            TextMatchAssertion(type="text_match", op="not_contains", value="123456")
        )
        == 'No reply contains "123456"'
    )
    label = assertion_label(
        ToolCallsAssertion(
            type="tool_calls",
            expected=[
                ExpectedToolCall(
                    name="cancel_order",
                    arguments={"order_id": "ORD-1002"},
                    arg_match_mode="partial",
                )
            ],
            match_mode="contains",
            forbidden=[ExpectedToolCall(name="refund")],
        )
    )
    assert label == (
        'Calls cancel_order({"order_id": "ORD-1002"}) (at least these); '
        "Never calls refund"
    )


def test_expected_behavior_text() -> None:
    assertions = [
        LLMJudgeAssertion(type="llm_judge", expected="Declines"),
        TextMatchAssertion(type="text_match", op="contains", value="x"),
    ]
    assert expected_behavior_text(assertions) == "- Declines"
