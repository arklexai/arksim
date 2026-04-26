# SPDX-License-Identifier: Apache-2.0
"""Root cause analysis for evaluation results.

Analyses evaluation statistics and uses an LLM to generate human-readable
hypotheses about why the agent failed, along with concrete recommendations.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM output schema (internal — not exported)
# ---------------------------------------------------------------------------


class _HypothesisItem(BaseModel):
    label: str  # 2–5 words
    problem: str  # one sentence, cite specific numbers or patterns
    fix: str  # one concrete sentence


class _RcaOutput(BaseModel):
    hypotheses: list[_HypothesisItem]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a quality analyst reviewing AI agent evaluation results.
Given a failure summary, return 2–3 root cause items.

Each item has:
- label: 2–5 words naming the root cause (e.g. "80-word brevity limit", \
"Hallucination under pressure", "Knowledge base not used")
- problem: ONE sentence describing the specific problem. \
You MUST cite exact numbers or patterns from the data \
(e.g. "average response is 67 words" not "responses are short", \
"2 out of 4 errors are lack of specific information" not "many errors").
- fix: ONE sentence stating the concrete action to fix it \
(e.g. "Remove the word-count cap in the agent system prompt" not "improve the agent").

Be blunt. No vague language. Cite the data.

Common patterns:
- avg response < 80 words + "lack of specific information" failures \
→ label: "Word limit causing omissions", cite the avg word count
- "false information" where knowledge has no numeric data \
→ label: "Hallucination under pressure"
- "disobey user request" for numbers when knowledge has them \
→ label: "Over-cautious refusal", cite the specific error
- repeated failures on same scenario → knowledge gap for that topic

Return 2–3 items. No duplicates."""

_USER_PROMPT = """\
Conversations: {conversation_count} ({impaired_count} impaired)
Avg response length: {avg_words:.0f} words

Failures:
{failure_distribution}

Unique errors ({unique_error_count}):
{unique_errors}
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_root_cause_analysis(
    evaluation: object,
    simulation: object,
    llm: object,
) -> object:
    """Analyse evaluation results and generate root cause hypotheses.

    Args:
        evaluation: ``Evaluation`` output from the evaluator.
        simulation: ``Simulation`` output (used to compute response lengths).
        llm: ``LLM`` instance to call for hypothesis generation.

    Returns:
        ``RootCauseAnalysis`` with a list of hypotheses and a generated_at
        timestamp.  Returns an empty ``RootCauseAnalysis`` on any error so
        the rest of the pipeline is never blocked.
    """
    # Import here to avoid circular imports at module load time.
    from arksim.evaluator.entities import RootCauseAnalysis, RootCauseHypothesis

    try:
        stats = _compute_stats(evaluation, simulation)
        output: _RcaOutput = llm.call(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _format_prompt(stats)},
            ],
            schema=_RcaOutput,
        )
        hypotheses = [
            RootCauseHypothesis(label=h.label, problem=h.problem, fix=h.fix)
            for h in output.hypotheses
        ]
    except Exception:
        logger.warning("Root cause analysis failed; skipping.", exc_info=True)
        hypotheses = []

    return RootCauseAnalysis(
        hypotheses=hypotheses,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_stats(evaluation: object, simulation: object) -> dict:  # type: ignore[type-arg]
    """Compute statistics needed for the LLM prompt."""
    # Failure category distribution across all turns
    failure_counts: Counter = Counter()
    for conv in evaluation.conversations:
        for turn in conv.turn_scores:
            failure_counts[turn.turn_behavior_failure] += 1

    # Average agent response word count
    word_counts: list[int] = []
    for conv in simulation.conversations:
        for msg in conv.conversation_history:
            if msg.role not in ("simulated_user", "user"):
                word_counts.append(len(msg.content.split()))
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0

    # Impaired conversation count
    impaired_statuses = {"Partial Failure", "Failed", "PARTIAL_FAILURE", "FAILED"}
    impaired_count = sum(
        1
        for conv in evaluation.conversations
        if conv.evaluation_status in impaired_statuses
    )

    return {
        "conversation_count": len(evaluation.conversations),
        "impaired_count": impaired_count,
        "avg_words": avg_words,
        "failure_counts": failure_counts,
        "unique_errors": evaluation.unique_errors,
    }


def _format_prompt(stats: dict) -> str:  # type: ignore[type-arg]
    """Format the user prompt from computed statistics."""
    _skip = {"no failure", "none", "skipped_good_performance", "evaluation_run_failed"}
    failure_lines = "\n".join(
        f"  {category}: {count} turn(s)"
        for category, count in sorted(
            stats["failure_counts"].items(), key=lambda x: -x[1]
        )
        if category.lower() not in _skip and count > 0
    ) or "  (no failures recorded)"

    error_lines = "\n".join(
        f"  [{err.severity.upper()}] {err.behavior_failure_category}: "
        f"{err.unique_error_description}"
        for err in stats["unique_errors"]
    ) or "  (none)"

    return _USER_PROMPT.format(
        conversation_count=stats["conversation_count"],
        impaired_count=stats["impaired_count"],
        avg_words=stats["avg_words"],
        failure_distribution=failure_lines,
        unique_error_count=len(stats["unique_errors"]),
        unique_errors=error_lines,
    )
