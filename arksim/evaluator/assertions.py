# SPDX-License-Identifier: Apache-2.0
"""Scenario assertions ("checks"): what the agent is expected to do.

Each assertion is reported on its own and the conversation's result is
derived from them. Only assertions decide the result; metrics such as
goal completion are reported alongside but never feed it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from arksim.llms.chat import LLM
from arksim.scenario.entities import (
    Assertion,
    ExpectedToolCall,
    LLMJudgeAssertion,
    TextMatchAssertion,
    ToolCallsAssertion,
)
from arksim.simulation_engine.tool_types import ToolCall

from .base_metric import ChatMessage
from .trajectory_matching import (
    TrajectoryResult,
    args_known,
    match_trajectory,
    tool_call_matches,
)
from .utils.prompts import (
    llm_judge_check_system_prompt,
    llm_judge_check_user_prompt,
)

logger = logging.getLogger(__name__)

AssertionStatus = Literal["pass", "fail", "inconclusive"]
ScenarioResult = Literal["pass", "fail", "inconclusive", "no_assertions"]

_TOOL_RESULT_CHARS = 500

_MATCH_MODE_LABELS = {
    "strict": "in this exact order",
    "unordered": "in any order, no others",
    "contains": "at least these",
}


class Turn(Protocol):
    """One exchange: its messages and the tool calls the agent made in it."""

    turn_id: int
    current_turn: list[ChatMessage]
    tool_calls: list[ToolCall] | None


class ToolEvidence(BaseModel):
    """How complete a conversation's observed tool calls are.

    Omitted, the calls are taken as complete, as when the agent reports them
    itself. A caller that captures them from traces says where they may not be.
    """

    #: Why some turns' calls are unknown, worded for the reason, e.g.
    #: "tool calls only partly captured".
    gaps: list[str] = Field(default_factory=list)
    #: False when no turn's calls were captured or reported at all.
    observed: bool = True


class ToolCallTrajectory(BaseModel):
    """A tool_calls assertion's evidence and the verdict on its expected calls.

    ``not_applicable`` when the assertion expects no calls.
    """

    status: Literal["pass", "fail", "inconclusive", "not_applicable"]
    match_mode: str | None = None
    expected: list[dict[str, Any]] = Field(default_factory=list)
    #: ``{"name", "arguments", "turn_id"}`` per observed call, in turn order.
    actual: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[dict[str, Any]] = Field(default_factory=list)
    unexpected: list[dict[str, Any]] = Field(default_factory=list)
    reason: str = ""


class AssertionResult(BaseModel):
    """Outcome of one assertion for one conversation."""

    type: str
    label: str
    status: AssertionStatus
    reason: str = ""
    #: tool_calls only: what was expected and observed.
    trajectory: ToolCallTrajectory | None = None


# ── Labels ──


def _call_text(call: ExpectedToolCall) -> str:
    if call.arguments and call.arg_match_mode != "ignore":
        return f"{call.name}({json.dumps(call.arguments, sort_keys=True)})"
    return call.name


def assertion_label(assertion: Assertion) -> str:
    """Short human-readable description of a check."""
    if isinstance(assertion, LLMJudgeAssertion):
        return assertion.expected
    if isinstance(assertion, TextMatchAssertion):
        return {
            "contains": f'A reply contains "{assertion.value}"',
            "not_contains": f'No reply contains "{assertion.value}"',
            "matches": f"A reply matches /{assertion.value}/",
        }[assertion.op]
    parts = []
    if assertion.expected:
        calls = ", ".join(_call_text(c) for c in assertion.expected)
        if assertion.match_mode == "within":
            parts.append(f"Allowed tools: {calls}")
        else:
            parts.append(f"Calls {calls} ({_MATCH_MODE_LABELS[assertion.match_mode]})")
    if assertion.forbidden:
        parts.append(
            "Never calls " + ", ".join(_call_text(c) for c in assertion.forbidden)
        )
    return "; ".join(parts) or "Tool calls"


def expected_behavior_text(assertions: Sequence[Assertion]) -> str:
    """The scenario's behavior checks, one per line, for judge prompts."""
    return "\n".join(
        f"- {a.expected}" for a in assertions if isinstance(a, LLMJudgeAssertion)
    )


# ── Deterministic checks ──


def evaluate_text_match(
    assertion: TextMatchAssertion, replies: Sequence[str]
) -> AssertionResult:
    """Check the agent's replies for a substring or regex."""
    pattern = assertion.compiled()
    hits = [i for i, r in enumerate(replies) if pattern.search(r)]
    label = assertion_label(assertion)
    if assertion.op == "not_contains":
        if hits:
            return AssertionResult(
                type=assertion.type,
                label=label,
                status="fail",
                reason=f"Found in agent reply {hits[0] + 1}.",
            )
        return AssertionResult(
            type=assertion.type, label=label, status="pass", reason="Not found."
        )
    if hits:
        return AssertionResult(
            type=assertion.type,
            label=label,
            status="pass",
            reason=f"Found in agent reply {hits[0] + 1}.",
        )
    return AssertionResult(
        type=assertion.type,
        label=label,
        status="fail",
        reason="No agent reply matched.",
    )


def evaluate_tool_calls(
    assertion: ToolCallsAssertion,
    turns: Sequence[Turn],
    evidence: ToolEvidence | None = None,
) -> AssertionResult:
    """Expected calls via trajectory matching, then the must-not-call list.

    Never a pass on evidence that cannot support one: a forbidden call is
    inconclusive when some turns' calls are unknown or none were observed.
    """
    trajectory = match_expected_calls(assertion, turns, evidence)
    statuses: list[AssertionStatus] = []
    reasons: list[str] = []
    if trajectory.status != "not_applicable":
        statuses.append(trajectory.status)
        reasons.append(trajectory.reason)
    if assertion.forbidden:
        status, reason = check_forbidden(assertion.forbidden, _calls(turns))
        if (
            status != "fail"
            and evidence is not None
            and (evidence.gaps or not evidence.observed)
        ):
            status, reason = "inconclusive", _gaps_reason(evidence)
        statuses.append(status)
        if reason not in reasons:
            reasons.append(reason)
    return AssertionResult(
        type=assertion.type,
        label=assertion_label(assertion),
        status=worst_status(statuses),
        reason=" ".join(r for r in reasons if r),
        trajectory=trajectory,
    )


def _gaps_reason(evidence: ToolEvidence) -> str:
    if not evidence.gaps:
        return "No tool calls were observed."
    return f"Some turns could not be checked: {', '.join(sorted(set(evidence.gaps)))}."


def match_expected_calls(
    assertion: ToolCallsAssertion,
    turns: Sequence[Turn],
    evidence: ToolEvidence | None = None,
) -> ToolCallTrajectory:
    """Match the observed calls, in turn order, against the expected ones.

    A fail stands whenever the observed calls alone rule out a pass.
    Otherwise inconclusive, never pass, when some turns' calls are unknown
    (they can only add unseen calls, so only a ``contains`` match stands), when ``within`` sees no call at all (an
    empty trace cannot satisfy it), when an argument-matched tool's arguments
    were not fully captured, or when ``strict`` ordering rests on calls whose
    captured ordering is ambiguous. The other modes match an empty observed
    list normally, so a non-empty expected list fails.
    """
    if not assertion.expected:
        return ToolCallTrajectory(status="not_applicable")
    mode = assertion.match_mode
    block = ToolCallTrajectory(
        status="inconclusive",
        match_mode=mode,
        expected=[call.model_dump() for call in assertion.expected],
        actual=[
            {"name": c.name, "arguments": c.arguments, "turn_id": t.turn_id}
            for t in turns
            for c in t.tool_calls or []
        ],
    )
    gaps = evidence is not None and bool(evidence.gaps)
    actual = _calls(turns)
    if not actual and mode == "within":
        block.reason = (
            _gaps_reason(evidence) if gaps else "No tool calls were observed."
        )
        return block
    arg_matched = {e.name for e in assertion.expected if e.arg_match_mode != "ignore"}
    uncaptured = sorted(
        {c.name for c in actual if c.name in arg_matched and not args_known(c)}
    )
    # A call with uncaptured arguments matches on its name alone, so any fail
    # left is one its missing arguments could not have cleared.
    expected = assertion.expected
    result = match_trajectory(actual, expected, mode)
    block.missing = [{"name": name} for name in result.missing_calls]
    block.unexpected = [{"name": name} for name in result.extra_calls]
    block.reason = result.reason
    ambiguous = mode == "strict" and any(
        c.capture_quality is not None and c.capture_quality.ordering == "ambiguous"
        for c in actual
    )
    if _definite_fail(result, actual, expected, mode, gaps=gaps, ambiguous=ambiguous):
        block.status = "fail"
        # A short strict list reports only what is missing, which the gaps may
        # hold; the observed calls fail on their order.
        if (
            gaps
            and mode == "strict"
            and len(actual) < len(expected)
            and not result.extra_calls
        ):
            block.reason = "Observed calls are out of the expected order. " + (
                result.reason
            )
        return block
    if gaps and not (mode == "contains" and result.matched and not uncaptured):
        # Unseen calls cannot undo a contains match, so only that pass stands.
        block.reason = _gaps_reason(evidence)
    elif uncaptured:
        block.reason = "Arguments were not fully captured for: " + ", ".join(uncaptured)
    elif ambiguous:
        block.reason = "Captured tool call ordering is ambiguous. " + result.reason
    else:
        block.status = "pass" if result.matched else "fail"
    return block


def _definite_fail(
    result: TrajectoryResult,
    actual: list[ToolCall],
    expected: list[ExpectedToolCall],
    mode: str,
    *,
    gaps: bool,
    ambiguous: bool,
) -> bool:
    """Whether the observed calls fail however the unknowns resolve."""
    if result.matched:
        return False
    if ambiguous:
        # Any order of the observed calls must fail.
        if gaps:
            return bool(match_trajectory(actual, expected, "unordered").extra_calls)
        return not match_trajectory(actual, expected, "unordered").matched
    if not gaps:
        return True
    # Unseen calls can be added anywhere but never remove an observed one.
    if mode == "strict":
        return not _is_subsequence(actual, expected)
    return mode in ("unordered", "within") and bool(result.extra_calls)


def _is_subsequence(actual: list[ToolCall], expected: list[ExpectedToolCall]) -> bool:
    remaining = iter(expected)
    return all(any(tool_call_matches(c, e) for e in remaining) for c in actual)


def _calls(turns: Sequence[Turn]) -> list[ToolCall]:
    return [c for t in turns for c in t.tool_calls or []]


def check_forbidden(
    forbidden: Sequence[ExpectedToolCall], actual: Sequence[ToolCall]
) -> tuple[AssertionStatus, str]:
    """Fail on a forbidden call that was made.

    For an entry that checks arguments, a call captured without its full
    arguments can neither match nor clear it, in either direction: the
    entry is inconclusive.
    """
    called: list[str] = []
    unclear: list[str] = []
    for f in forbidden:
        same_name = [a for a in actual if a.name == f.name]
        known = [a for a in same_name if f.arg_match_mode == "ignore" or args_known(a)]
        if any(tool_call_matches(a, f) for a in known):
            called.append(f.name)
        elif len(known) < len(same_name):
            unclear.append(f.name)
    if called:
        return "fail", "Called forbidden tool(s): " + ", ".join(called) + "."
    if unclear:
        return (
            "inconclusive",
            "Arguments were not fully captured for: "
            + ", ".join(sorted(set(unclear)))
            + ".",
        )
    return "pass", "No forbidden tool was called."


def worst_status(statuses: Sequence[AssertionStatus]) -> AssertionStatus:
    """fail over inconclusive over pass; pass when there is nothing to check."""
    if "fail" in statuses:
        return "fail"
    return "inconclusive" if "inconclusive" in statuses else "pass"


# ── LLM-judged checks ──


class _BehaviorVerdict(BaseModel):
    index: int
    reason: str
    status: AssertionStatus


class _BehaviorVerdicts(BaseModel):
    verdicts: list[_BehaviorVerdict]


def _format_call(tc: ToolCall) -> str:
    result = tc.result if tc.error is None else f"ERROR: {tc.error}"
    result = str(result)[:_TOOL_RESULT_CHARS] if result is not None else ""
    return f"{tc.name}({json.dumps(tc.arguments, sort_keys=True)}) -> {result}"


def _format_turns(turns: Sequence[Turn], evidence: ToolEvidence | None) -> str:
    """The conversation with each turn's tool calls just before its reply."""
    lines = []
    if evidence is not None and (evidence.gaps or not evidence.observed):
        lines.append(
            "(Tool calls were not captured on every turn; a turn that shows none "
            "may still have made calls.)"
        )
    for t in turns:
        for m in t.current_turn:
            if m.role == "assistant":
                lines.extend(
                    f"assistant tool call: {_format_call(c)}"
                    for c in t.tool_calls or []
                )
            lines.append(f"{m.role}: {m.content}")
    return "\n".join(lines)


def evaluate_llm_judges(
    assertions: Sequence[LLMJudgeAssertion],
    *,
    llm: LLM,
    turns: Sequence[Turn],
    evidence: ToolEvidence | None = None,
    agent_context: str = "",
    knowledge: str = "",
) -> list[AssertionResult]:
    """Judge every llm_judge assertion of a conversation in one LLM call."""
    if not assertions:
        return []
    expectations = "\n".join(f"{i}. {a.expected}" for i, a in enumerate(assertions))
    try:
        response = llm.call(
            [
                {"role": "system", "content": llm_judge_check_system_prompt},
                {
                    "role": "user",
                    "content": llm_judge_check_user_prompt.format(
                        agent_context=agent_context or "N/A",
                        knowledge=knowledge or "N/A",
                        conversation=_format_turns(turns, evidence),
                        expectations=expectations,
                    ),
                },
            ],
            schema=_BehaviorVerdicts,
        )
        by_index = {v.index: v for v in response.verdicts}
    except Exception:
        logger.exception("LLM judge call failed")
        by_index = {}
    results = []
    for i, a in enumerate(assertions):
        verdict = by_index.get(i)
        results.append(
            AssertionResult(
                type=a.type,
                label=a.expected,
                status=verdict.status if verdict else "inconclusive",
                reason=verdict.reason if verdict else "The judge returned no verdict.",
            )
        )
    return results


# ── Orchestration ──


def evaluate_assertions(
    assertions: Sequence[Assertion],
    *,
    llm: LLM,
    turns: Sequence[Turn],
    evidence: ToolEvidence | None = None,
    agent_context: str = "",
    knowledge: str = "",
) -> list[AssertionResult]:
    """Evaluate every assertion, preserving the scenario's order.

    ``evidence`` says how complete the turns' tool calls are; omit it when the
    agent reported them itself.
    """
    judged = iter(
        evaluate_llm_judges(
            [a for a in assertions if isinstance(a, LLMJudgeAssertion)],
            llm=llm,
            turns=turns,
            evidence=evidence,
            agent_context=agent_context,
            knowledge=knowledge,
        )
    )
    replies = [
        m.content for t in turns for m in t.current_turn if m.role == "assistant"
    ]
    results = []
    for a in assertions:
        if isinstance(a, LLMJudgeAssertion):
            results.append(next(judged))
        elif isinstance(a, TextMatchAssertion):
            results.append(evaluate_text_match(a, replies))
        else:
            results.append(evaluate_tool_calls(a, turns, evidence))
    return results


def overall_result(results: Sequence[AssertionResult]) -> ScenarioResult:
    """Fail if any check failed, inconclusive if any could not be decided."""
    if not results:
        return "no_assertions"
    return worst_status([r.status for r in results])


_SUMMARY_KEYS: dict[ScenarioResult, str] = {
    "pass": "passed",
    "fail": "failed",
    "inconclusive": "inconclusive",
    "no_assertions": "no_assertions",
}


def summarize_results(results: Sequence[ScenarioResult]) -> dict[str, int]:
    """Count conversations by result, for the run's assertion summary."""
    counts = Counter(results)
    return {key: counts.get(result, 0) for result, key in _SUMMARY_KEYS.items()}
