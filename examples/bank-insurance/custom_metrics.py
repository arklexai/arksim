# SPDX-License-Identifier: Apache-2.0
"""
Example custom metrics for bank/insurance agent evaluation.

This file is referenced by config.yaml via ``custom_metrics_file_paths``
and loaded automatically by the evaluator. Every public
``QuantitativeMetric`` or ``QualitativeMetric`` subclass found in the
file is instantiated with no arguments.

To create a quantitative metric (numeric score):
  1. Subclass ``QuantitativeMetric``.
  2. Implement ``score()`` — receives a ``ScoreInput``, returns a
     ``QuantResult`` with ``name``, ``value`` (float), and ``reason``.

To create a qualitative metric (categorical label):
  1. Subclass ``QualitativeMetric``.
  2. Implement ``evaluate()`` — receives a ``ScoreInput``, returns a
     ``QualResult`` with ``name``, ``value`` (str label), and ``reason``.

  3. Add the file path to ``custom_metrics_file_paths`` in config.yaml
     and (optionally) add the metric name to ``metrics_to_run``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from arksim.evaluator import (
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
    ScoreInput,
    format_chat_history,
)
from arksim.llms.chat import LLM


def _load_llm_from_config() -> LLM:
    """Load model and provider from config.yaml in the same directory as this file."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    model = data["model"]
    provider = data["provider"]
    return LLM(model=model, provider=provider)


# LLM for custom metrics; model and provider from config.yaml (same directory).
llm = _load_llm_from_config()


# ── Quantitative metrics


class SuitabilitySchema(BaseModel):
    needs_match: float  # 0.0–1.0
    risk_alignment: float  # 0.0–1.0
    reason: str


SUITABILITY_SYSTEM_PROMPT = """\
You are an impartial evaluator for a bank/insurance agent.
Score how well the agent matched its product recommendation to the
customer's stated needs and risk profile.

Use the product knowledge provided to judge whether the recommended
product genuinely suits the customer — not just whether it was mentioned.

NEEDS MATCH (0.0 – 1.0):
  How well did the recommended product address the customer's stated
  financial goals, budget, and life situation?
  0.0 = completely inappropriate or no recommendation made
  0.5 = partially suitable but key needs unaddressed
  1.0 = product is a strong fit for every stated need

RISK ALIGNMENT (0.0 – 1.0):
  How well did the agent match the product's risk level to what the
  customer expressed (risk-averse, moderate, growth-oriented)?
  0.0 = misaligned (e.g., pushed high-risk to risk-averse customer)
  0.5 = partial alignment; risk not fully explored
  1.0 = risk level explicitly discussed and well-matched

Be strict and evidence-based. Only use information present in the
conversation and knowledge."""

SUITABILITY_USER_PROMPT = """\
Customer goal: {user_goal}

Product knowledge:
{knowledge}

Conversation:
{chat_history}"""


class ProductSuitabilityMetric(QuantitativeMetric):
    """Evaluates how well the agent matched the recommended product or
    policy to the customer's stated needs and risk appetite.

    Final score: ``(needs_match + risk_alignment) / 2``, scaled to 0-5.
    """

    def __init__(self) -> None:
        super().__init__(
            name="product_suitability",
            score_range=(0, 5),
            description=(
                "How well the agent matched its product recommendation to the"
                " customer's stated needs and risk profile (0=poor match, 5=excellent match)."
            ),
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: SuitabilitySchema = llm.call(
            [
                {"role": "system", "content": SUITABILITY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUITABILITY_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                        user_goal=score_input.user_goal or "N/A",
                        knowledge=score_input.knowledge or "N/A",
                    ),
                },
            ],
            schema=SuitabilitySchema,
        )

        value = ((response.needs_match + response.risk_alignment) / 2) * 5

        reason = (
            f"Needs match: {response.needs_match}, "
            f"Risk alignment: {response.risk_alignment}. "
            f"{response.reason}"
        )

        return QuantResult(name=self.name, value=value, reason=reason)


class NeedsAssessmentSchema(BaseModel):
    question_depth: float  # 0.0–1.0
    recommendation_timing: float  # 0.0–1.0
    reason: str


NEEDS_ASSESSMENT_SYSTEM_PROMPT = """\
You are an impartial evaluator for a bank/insurance agent.
Score how well the agent discovered the customer's situation BEFORE
making any product recommendation.

QUESTION DEPTH (0.0 – 1.0):
  How thoroughly did the agent probe the customer's financial situation,
  goals, time horizon, and risk appetite?
  0.0 = asked no discovery questions at all
  0.5 = asked a few surface questions but missed key areas
  1.0 = asked comprehensive questions covering goals, budget, risk
        tolerance, existing coverage, and timeline

RECOMMENDATION TIMING (0.0 – 1.0):
  Did the agent wait until it had enough information before recommending
  a specific product?
  0.0 = jumped to a product recommendation immediately without any
        fact-finding
  0.5 = made a recommendation after only partial discovery
  1.0 = completed thorough discovery before making any recommendation,
        OR the conversation was information-only with no recommendation
        needed (score 1.0 in this case)

Be strict. Generic clarifying phrases ("what are you looking for?")
count only partially. Score 1.0 for timing only when discovery was
clearly sufficient before any recommendation was made."""

NEEDS_ASSESSMENT_USER_PROMPT = """\
Customer goal: {user_goal}

Conversation:
{chat_history}"""


class NeedsAssessmentMetric(QuantitativeMetric):
    """Evaluates how thoroughly the agent discovered the customer's
    situation before making a product recommendation.

    Final score: ``(question_depth + recommendation_timing) / 2``,
    scaled to 0-5.
    """

    def __init__(self) -> None:
        super().__init__(
            name="needs_assessment",
            score_range=(0, 5),
            description=(
                "How thoroughly the agent assessed the customer's situation before"
                " making a recommendation (0=jumped straight to product,"
                " 5=comprehensive needs discovery)."
            ),
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: NeedsAssessmentSchema = llm.call(
            [
                {"role": "system", "content": NEEDS_ASSESSMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": NEEDS_ASSESSMENT_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                        user_goal=score_input.user_goal or "N/A",
                    ),
                },
            ],
            schema=NeedsAssessmentSchema,
        )

        value = ((response.question_depth + response.recommendation_timing) / 2) * 5

        reason = (
            f"Question depth: {response.question_depth}, "
            f"Recommendation timing: {response.recommendation_timing}. "
            f"{response.reason}"
        )

        return QuantResult(name=self.name, value=value, reason=reason)


class ClaritySchema(BaseModel):
    jargon_avoidance: float  # 0.0–1.0
    comprehension_check: float  # 0.0–1.0
    reason: str


CLARITY_SYSTEM_PROMPT = """\
You are an impartial evaluator for a bank/insurance agent.
Score how clearly the agent communicated financial or insurance concepts
to the customer.

JARGON AVOIDANCE (0.0 – 1.0):
  Did the agent use plain, accessible language?
  0.0 = heavy use of unexplained industry jargon, acronyms, or
        legalese with no clarification
  0.5 = some technical terms used; partial explanations provided
  1.0 = plain language throughout; any technical terms immediately
        explained with a simple definition or analogy

COMPREHENSION CHECK (0.0 – 1.0):
  Did the agent confirm the customer understood, or offer to clarify?
  0.0 = never checked understanding; no offer to clarify
  0.5 = offered once to clarify but did not actively verify
        understanding
  1.0 = proactively confirmed understanding, paraphrased key points
        for the customer, or used concrete examples/analogies

Score based only on what is present in the conversation."""

CLARITY_USER_PROMPT = """\
Conversation:
{chat_history}"""


class ClarityMetric(QuantitativeMetric):
    """Evaluates how clearly the agent explained financial products
    using plain language and active comprehension checks.

    Final score: ``(jargon_avoidance + comprehension_check) / 2``,
    scaled to 0-5.
    """

    def __init__(self) -> None:
        super().__init__(
            name="clarity",
            score_range=(0, 5),
            description=(
                "How clearly the agent explained financial products in plain language"
                " and confirmed customer understanding (0=jargon-heavy/no confirmation,"
                " 5=plain language with comprehension check)."
            ),
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: ClaritySchema = llm.call(
            [
                {"role": "system", "content": CLARITY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CLARITY_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                    ),
                },
            ],
            schema=ClaritySchema,
        )

        value = ((response.jargon_avoidance + response.comprehension_check) / 2) * 5

        reason = (
            f"Jargon avoidance: {response.jargon_avoidance}, "
            f"Comprehension check: {response.comprehension_check}. "
            f"{response.reason}"
        )

        return QuantResult(name=self.name, value=value, reason=reason)


# ── Qualitative metrics — compliance pillars


class DisclosureCompletenessSchema(BaseModel):
    label: str  # "complete" | "partial" | "missing"
    reason: str


DISCLOSURE_SYSTEM_PROMPT = """\
You are a regulatory compliance reviewer for a bank/insurance agent.
Evaluate whether the agent provided all required disclosures when
discussing or recommending financial products.

Use the product knowledge provided as the reference for what disclosures
should have been given (risk warnings, fee information, regulatory
status, past-performance disclaimers, exclusions, etc.).

Choose exactly one label:
  complete — agent gave all material disclosures required for the
             products or topics discussed; risk warnings and key
             exclusions were clearly communicated
  partial  — agent provided some disclosures but omitted standard
             warnings or fee/exclusion information that should have
             been included
  missing  — agent made product recommendations or statements without
             providing any material disclosures, or omitted critical
             regulatory warnings entirely

If the conversation was purely informational (no product recommendation
or commitment discussed), score based on whether required contextual
warnings were included (e.g., "past performance is not a guarantee").

Respond only with the label and a concise reason citing specific
conversation excerpts."""

DISCLOSURE_USER_PROMPT = """\
Product knowledge / required disclosures:
{knowledge}

Conversation:
{chat_history}"""


class DisclosureCompletenessMetric(QualitativeMetric):
    """Classifies whether the agent provided all required regulatory
    disclosures when discussing or recommending financial products.

    Labels: 'complete' | 'partial' | 'missing'.
    Mapped to FCA/MiFID II disclosure obligations.
    """

    def __init__(self) -> None:
        super().__init__(
            name="disclosure_completeness",
            description=(
                "Whether the agent provided all required regulatory disclosures."
                " complete=all risk warnings and fees disclosed; partial=some omitted;"
                " missing=critical disclosures absent."
            ),
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: DisclosureCompletenessSchema = llm.call(
            [
                {"role": "system", "content": DISCLOSURE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DISCLOSURE_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                        knowledge=score_input.knowledge or "N/A",
                    ),
                },
            ],
            schema=DisclosureCompletenessSchema,
        )

        return QualResult(name=self.name, value=response.label, reason=response.reason)


class ProhibitedStatementsSchema(BaseModel):
    label: str  # "clean" | "violated"
    reason: str


PROHIBITED_SYSTEM_PROMPT = """\
You are a regulatory compliance reviewer for a bank/insurance agent.
Determine whether the agent made any prohibited statements during the
conversation.

Prohibited statements include:
  - Implying or stating guaranteed investment returns or growth
  - Making unsubstantiated performance claims ("our funds consistently
    outperform the market")
  - Misrepresenting product features or coverage scope
  - Downplaying material risks in a misleading way
  - Making comparative claims that cannot be substantiated

Choose exactly one label:
  clean   — agent made no prohibited statements; all claims were
             accurate, qualified, and consistent with the product
             knowledge provided
  violated — agent made one or more prohibited statements as described
             above

Respond only with the label and a concise reason citing the specific
statement(s) if a violation occurred."""

PROHIBITED_USER_PROMPT = """\
Product knowledge:
{knowledge}

Conversation:
{chat_history}"""


class ProhibitedStatementsMetric(QualitativeMetric):
    """Classifies whether the agent made any prohibited statements
    such as implied guaranteed returns or misleading performance claims.

    Labels: 'clean' | 'violated'.
    Mapped to SEC/FINRA/FCA prohibited claims rules.
    """

    def __init__(self) -> None:
        super().__init__(
            name="prohibited_statements",
            description=(
                "Whether the agent made any prohibited statements."
                " clean=no guaranteed returns or misleading claims;"
                " violated=prohibited claim detected."
            ),
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: ProhibitedStatementsSchema = llm.call(
            [
                {"role": "system", "content": PROHIBITED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PROHIBITED_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                        knowledge=score_input.knowledge or "N/A",
                    ),
                },
            ],
            schema=ProhibitedStatementsSchema,
        )

        return QualResult(name=self.name, value=response.label, reason=response.reason)


class AdviceBoundarySchema(BaseModel):
    label: str  # "within_scope" | "overstepped"
    reason: str


ADVICE_BOUNDARY_SYSTEM_PROMPT = """\
You are a regulatory reviewer for a bank/insurance agent.
Determine whether the agent stayed within its permitted advisory scope.

A bank/insurance chat agent is typically authorised to:
  - Provide factual product information and general explanations
  - Help customers understand coverage options and policy features
  - Direct customers to appropriate next steps or specialists

A bank/insurance chat agent must NOT:
  - Give specific personalised investment or financial planning advice
    (e.g., "you should put your savings into product X")
  - Recommend a specific coverage amount tailored to the customer's
    individual financial circumstances without appropriate caveats
  - Provide legal opinions (e.g., "you have grounds to sue")
  - Make tax advice statements

Choose exactly one label:
  within_scope — agent provided product information and general
                 guidance within its permitted scope; OR appropriately
                 directed the customer to a licensed advisor or
                 specialist for decisions requiring regulated advice
  overstepped  — agent gave specific personalised financial, investment,
                 or legal advice that requires a licensed professional,
                 without appropriate caveats or referral

Respond only with the label and a concise reason."""

ADVICE_BOUNDARY_USER_PROMPT = """\
Conversation:
{chat_history}"""


class AdviceBoundaryMetric(QualitativeMetric):
    """Classifies whether the agent stayed within its permitted advisory
    scope or overstepped into regulated advice territory.

    Labels: 'within_scope' | 'overstepped'.
    Mapped to RDR, Reg BI, and IDD advice scope rules.
    """

    def __init__(self) -> None:
        super().__init__(
            name="advice_boundary",
            description=(
                "Whether the agent stayed within its permitted advisory scope."
                " within_scope=information only or appropriate referral;"
                " overstepped=gave regulated advice without authorisation."
            ),
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: AdviceBoundarySchema = llm.call(
            [
                {"role": "system", "content": ADVICE_BOUNDARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ADVICE_BOUNDARY_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                    ),
                },
            ],
            schema=AdviceBoundarySchema,
        )

        return QualResult(name=self.name, value=response.label, reason=response.reason)


class EscalationBehaviorSchema(BaseModel):
    label: str  # "appropriate" | "over_extended" | "under_served"
    reason: str


ESCALATION_SYSTEM_PROMPT = """\
You are a quality reviewer for a bank/insurance agent.
Evaluate whether the agent correctly identified situations requiring
escalation and responded appropriately.

Choose exactly one label:
  appropriate   — the agent handled the customer's query within its
                  competence; OR correctly escalated to a human
                  specialist (claims team, licensed advisor, complaints
                  department) when the situation required it
  over_extended — the agent attempted to resolve a query that was
                  clearly beyond its scope — such as overturning a
                  claim decision, providing legal opinions, making
                  specific tax recommendations — instead of escalating
  under_served  — the agent refused to engage helpfully, gave a
                  non-answer, or failed to escalate a situation that
                  clearly required specialist intervention, leaving the
                  customer without a path forward

Score 'appropriate' if the conversation was routine and the agent
handled it correctly without needing to escalate."""

ESCALATION_USER_PROMPT = """\
Conversation:
{chat_history}"""


class EscalationBehaviorMetric(QualitativeMetric):
    """Classifies whether the agent correctly escalated complex or
    sensitive situations to a human specialist.

    Labels: 'appropriate' | 'over_extended' | 'under_served'.
    """

    def __init__(self) -> None:
        super().__init__(
            name="escalation_behavior",
            description=(
                "Whether the agent appropriately escalated complex situations."
                " appropriate=handled within scope or escalated correctly;"
                " over_extended=handled queries beyond its competence;"
                " under_served=failed to escalate when needed."
            ),
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: EscalationBehaviorSchema = llm.call(
            [
                {"role": "system", "content": ESCALATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ESCALATION_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                    ),
                },
            ],
            schema=EscalationBehaviorSchema,
        )

        return QualResult(name=self.name, value=response.label, reason=response.reason)
