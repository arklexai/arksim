# SPDX-License-Identifier: Apache-2.0
"""Deterministic trajectory matching for tool call evaluation.

Compares actual tool calls against expected tool calls without LLM cost.
"""

from __future__ import annotations

from pydantic import BaseModel

from arksim.scenario.entities import ExpectedToolCall
from arksim.simulation_engine.tool_types import ToolCall


class TrajectoryResult(BaseModel):
    """Result of trajectory matching."""

    matched: bool
    failure_label: str | None = None
    reason: str = ""
    missing_calls: list[str] = []
    extra_calls: list[str] = []
    ordering_issues: list[str] = []


def _args_match(
    actual_args: dict,
    expected_args: dict,
    arg_match_mode: str,
) -> bool:
    """Check if actual arguments match expected arguments."""
    if arg_match_mode == "ignore":
        return True
    if arg_match_mode == "exact":
        return actual_args == expected_args
    # partial: expected args must appear in actual args
    return all(
        k in actual_args and actual_args[k] == v for k, v in expected_args.items()
    )


def _find_match(
    actual: ToolCall,
    expected: ExpectedToolCall,
) -> bool:
    """Check if an actual tool call matches an expected one."""
    if actual.name != expected.name:
        return False
    return _args_match(actual.arguments, expected.arguments, expected.arg_match_mode)


def match_trajectory(
    actual: list[ToolCall],
    expected: list[ExpectedToolCall],
    match_mode: str = "unordered",
) -> TrajectoryResult:
    """Match actual tool calls against expected tool calls.

    Args:
        actual: Tool calls the agent actually made.
        expected: Tool calls the scenario author expected.
        match_mode: One of "strict", "unordered", "contains", "within".

    Returns:
        TrajectoryResult with match status and diagnostics.
    """
    if not expected:
        return TrajectoryResult(matched=True, reason="No expected tool calls defined.")

    if not actual:
        return TrajectoryResult(
            matched=False,
            failure_label="disobey user request",
            reason="Agent made no tool calls but expected: "
            + ", ".join(e.name for e in expected),
            missing_calls=[e.name for e in expected],
        )

    if match_mode == "strict":
        return _match_strict(actual, expected)
    if match_mode == "unordered":
        return _match_unordered(actual, expected)
    if match_mode == "contains":
        return _match_contains(actual, expected)
    if match_mode == "within":
        return _match_within(actual, expected)

    return TrajectoryResult(
        matched=False,
        failure_label="disobey user request",
        reason=f"Unknown match_mode: {match_mode}",
    )


def _match_strict(
    actual: list[ToolCall],
    expected: list[ExpectedToolCall],
) -> TrajectoryResult:
    """Exact order and count match."""
    ordering_issues: list[str] = []

    if len(actual) != len(expected):
        # Use unordered matching to identify accurate missing/extra lists,
        # then report the count mismatch with correct diagnostics.
        matched_expected = [False] * len(expected)
        matched_actual = [False] * len(actual)
        for i, exp in enumerate(expected):
            for j, act in enumerate(actual):
                if not matched_actual[j] and _find_match(act, exp):
                    matched_expected[i] = True
                    matched_actual[j] = True
                    break
        missing = [
            expected[i].name for i in range(len(expected)) if not matched_expected[i]
        ]
        extra = [actual[j].name for j in range(len(actual)) if not matched_actual[j]]
        if extra and not missing:
            failure_label = _extra_calls_label(extra, expected)
        else:
            failure_label = "disobey user request"
        return TrajectoryResult(
            matched=False,
            failure_label=failure_label,
            reason=f"Expected {len(expected)} tool calls but got {len(actual)}."
            + (f" Missing: {', '.join(missing)}." if missing else "")
            + (f" Extra: {', '.join(extra)}." if extra else ""),
            missing_calls=missing,
            extra_calls=extra,
        )

    substitutions: list[str] = []
    arg_mismatches: list[tuple[int, str, str]] = []  # (position, label, detail)
    for i, (act, exp) in enumerate(zip(actual, expected, strict=False)):
        if act.name != exp.name:
            if any(e.name == act.name for e in expected):
                ordering_issues.append(
                    f"Position {i}: expected {exp.name}, got {act.name}"
                )
            else:
                substitutions.append(
                    f"Position {i}: expected {exp.name}, got {act.name} (not in expected set)"
                )
        elif not _args_match(act.arguments, exp.arguments, exp.arg_match_mode):
            label = (
                "lack of specific information"
                if exp.arg_match_mode == "partial"
                else "false information"
            )
            arg_mismatches.append(
                (
                    i,
                    label,
                    f"Tool {act.name} at position {i}: argument mismatch. "
                    f"Expected {exp.arguments}, got {act.arguments}.",
                )
            )

    # Report all collected issues together so nothing is silently dropped
    if ordering_issues or substitutions or arg_mismatches:
        parts: list[str] = []
        if ordering_issues:
            parts.append("Tool call ordering mismatch. " + "; ".join(ordering_issues))
        if substitutions:
            parts.append("Wrong tool called. " + "; ".join(substitutions))
        if arg_mismatches:
            parts.extend(detail for _, _, detail in arg_mismatches)

        # Pick the most severe failure label: ordering/substitution issues
        # take precedence ("disobey user request") over arg mismatches.
        # Among arg mismatches, "false information" (critical) beats
        # "lack of specific information" (medium).
        if ordering_issues or substitutions:
            failure_label = "disobey user request"
        elif any(label == "false information" for _, label, _ in arg_mismatches):
            failure_label = "false information"
        else:
            failure_label = arg_mismatches[0][1]

        return TrajectoryResult(
            matched=False,
            failure_label=failure_label,
            reason=". ".join(parts),
            ordering_issues=ordering_issues,
        )

    return TrajectoryResult(matched=True, reason="All tool calls match in order.")


def _extra_calls_label(
    extra: list[str],
    expected: list[ExpectedToolCall],
) -> str:
    """Pick failure label for extra tool calls.

    Same-name extras (redundant duplicates) map to "repetition".
    Unrelated extras (tools outside the expected set) map to
    "disobey user request".
    """
    expected_names = {e.name for e in expected}
    if all(name in expected_names for name in extra):
        return "repetition"
    return "disobey user request"


def _match_unordered(
    actual: list[ToolCall],
    expected: list[ExpectedToolCall],
) -> TrajectoryResult:
    """Same set of calls, any order."""
    # Track which expected calls have been matched
    matched_expected: list[bool] = [False] * len(expected)
    matched_actual: list[bool] = [False] * len(actual)

    for i, exp in enumerate(expected):
        for j, act in enumerate(actual):
            if not matched_actual[j] and _find_match(act, exp):
                matched_expected[i] = True
                matched_actual[j] = True
                break

    missing = [
        expected[i].name for i in range(len(expected)) if not matched_expected[i]
    ]
    extra = [actual[j].name for j in range(len(actual)) if not matched_actual[j]]

    if missing and extra:
        return TrajectoryResult(
            matched=False,
            failure_label="disobey user request",
            reason=f"Missing expected calls: {', '.join(missing)}. "
            f"Unexpected calls: {', '.join(extra)}.",
            missing_calls=missing,
            extra_calls=extra,
        )
    if missing:
        return TrajectoryResult(
            matched=False,
            failure_label="disobey user request",
            reason=f"Missing expected tool calls: {', '.join(missing)}.",
            missing_calls=missing,
        )
    if extra:
        return TrajectoryResult(
            matched=False,
            failure_label=_extra_calls_label(extra, expected),
            reason=f"Unexpected extra tool calls: {', '.join(extra)}.",
            extra_calls=extra,
        )

    return TrajectoryResult(matched=True, reason="All expected tool calls found.")


def _match_contains(
    actual: list[ToolCall],
    expected: list[ExpectedToolCall],
) -> TrajectoryResult:
    """Agent must call at least the expected tools (extras are allowed)."""
    matched_expected: list[bool] = [False] * len(expected)
    matched_actual: list[bool] = [False] * len(actual)

    for i, exp in enumerate(expected):
        for j, act in enumerate(actual):
            if not matched_actual[j] and _find_match(act, exp):
                matched_expected[i] = True
                matched_actual[j] = True
                break

    missing = [
        expected[i].name for i in range(len(expected)) if not matched_expected[i]
    ]
    if missing:
        return TrajectoryResult(
            matched=False,
            failure_label="disobey user request",
            reason=f"Missing expected tool calls: {', '.join(missing)}.",
            missing_calls=missing,
        )

    return TrajectoryResult(
        matched=True, reason="All expected tool calls found (contains mode)."
    )


def _match_within(
    actual: list[ToolCall],
    expected: list[ExpectedToolCall],
) -> TrajectoryResult:
    """Agent can only call tools from the expected set (may skip some)."""
    matched_actual: list[bool] = [False] * len(actual)

    for j, act in enumerate(actual):
        for exp in expected:
            if _find_match(act, exp):
                matched_actual[j] = True
                break

    extra = [actual[j].name for j in range(len(actual)) if not matched_actual[j]]
    if extra:
        return TrajectoryResult(
            matched=False,
            failure_label=_extra_calls_label(extra, expected),
            reason=f"Agent called tools not in expected set: {', '.join(extra)}.",
            extra_calls=extra,
        )

    return TrajectoryResult(
        matched=True, reason="All actual calls within expected set (within mode)."
    )
