# SPDX-License-Identifier: Apache-2.0
"""
Example custom metrics for e-commerce agent evaluation.

This file is referenced by config.yaml via ``custom_metrics_file_paths``
and loaded automatically by the evaluator. Every public ``QuantitativeMetric``
or ``QualitativeMetric`` subclass found in the file is instantiated with
no arguments.

To create your own metric:
  1. Subclass ``QuantitativeMetric`` or ``QualitativeMetric``.
  2. Implement ``score()`` (quantitative) or ``evaluate()`` (qualitative).
     Both receive a ``ScoreInput`` with ``chat_history``, ``knowledge``,
     ``user_goal``, and ``profile``.
  3. Return a ``QuantResult`` or ``QualResult`` with ``name``, ``value``,
     and ``reason``.
  4. Add the file path to ``custom_metrics_file_paths`` in config.yaml
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


# ── Conversion Metric


class ConversionSchema(BaseModel):
    intent_strength: float
    conversion_outcome: float
    evidence: list[str]
    reason: str


CONVERSION_SYSTEM_PROMPT = """\
You are an impartial evaluator for an e-commerce shopping agent.
Your job is to score (1) the shopper's purchase intent and (2) whether the agent achieved a conversion outcome
(or the closest valid proxy outcome when purchase is not possible).
Be strict, evidence-based, and only use information present in the conversation and provided metadata.
Do not assume outcomes that are not stated.

DEFINITIONS:
- "Intent strength" reflects how ready the shopper is to buy NOW in this session.
- "Conversion outcome" is whether the agent caused or enabled the user to commit to a purchase decision
  (or reached a high-confidence "ready to purchase" state with a specific product/SKU and plan).
  If actual purchase is unknown, score a proxy: "ReadyToBuy" if user explicitly commits to buying a specific item,
  asks for checkout/links, or confirms final selection.

INTENT STRENGTH SCALE (choose one):
0.0 = browsing / curiosity; no stated need
0.3 = early research; vague preferences; comparing options
0.6 = clear need; some constraints; shopping with intent but undecided
0.8 = high intent; strong constraints; near decision (budget/size/brand known)
1.0 = immediate purchase intent; asks for "best one to buy", "link", "checkout", "buy now", "add to cart"

CONVERSION OUTCOME SCALE (choose one):
0   = NoProgress: agent did not advance toward a purchase decision
0.5 = Partial: agent advanced decision-making but user not ready / key constraints unresolved
1.0 = ConvertedProxy: user explicitly commits to a specific product choice or next step that implies purchase
      (e.g., "I'll buy X", "Send me the link to X", "Adding X to cart", "I'll checkout")

EVIDENCE RULE:
- Every score must be justified with 2-5 short quotes (<=15 words each) from the conversation."""

CONVERSION_USER_PROMPT = """\
Evaluate the following conversation turn between a shopper and an e-commerce agent.

User goal: {user_goal}

Conversation:
{chat_history}"""


class ConversionMetric(QuantitativeMetric):
    """Custom metric that uses an LLM to evaluate purchase intent strength
    and conversion outcome for an e-commerce shopping agent.

    The final score is: (intent_strength + conversion_outcome) / 2,
    normalised to 0-5.
    """

    def __init__(self) -> None:
        super().__init__(
            name="conversion",
            score_range=(0, 5),
            description="Scores purchase intent strength and conversion outcome. "
            "Final score = (intent_strength + conversion_outcome) / 2, scaled to 0-5.",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        chat_history = format_chat_history(score_input.chat_history)
        user_goal = score_input.user_goal or "N/A"

        response: ConversionSchema = llm.call(
            [
                {"role": "system", "content": CONVERSION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CONVERSION_USER_PROMPT.format(
                        chat_history=chat_history,
                        user_goal=user_goal,
                    ),
                },
            ],
            schema=ConversionSchema,
        )

        # Combine into a 0-5 score: average of intent (0-1) and conversion (0-1), scaled to 5
        value = ((response.intent_strength + response.conversion_outcome) / 2) * 5

        full_reason = (
            f"Intent: {response.intent_strength}, Conversion: {response.conversion_outcome}. "
            f"{response.reason} Evidence: {'; '.join(response.evidence)}"
        )

        return QuantResult(name=self.name, value=value, reason=full_reason)


# ── Product Recommendation Metric


class RecommendationSchema(BaseModel):
    relevance: float
    specificity: float
    reason: str


RECOMMENDATION_SYSTEM_PROMPT = """\
You are an impartial evaluator for an e-commerce shopping assistant.
Score the agent's product recommendations on two dimensions.

RELEVANCE (0.0 - 1.0):
  How well do the recommended products match the user's stated needs,
  preferences, and constraints (budget, size, brand, use-case, etc.)?
  0.0 = completely off-topic or no recommendation given
  0.5 = partially relevant but missing key constraints
  1.0 = perfectly aligned with every stated requirement

SPECIFICITY (0.0 - 1.0):
  Does the agent give concrete, actionable recommendations
  (specific product names, models, prices) rather than vague advice?
  0.0 = only generic suggestions ("look at our website")
  0.5 = some product names but missing details
  1.0 = specific products with relevant details (name, price, features)

Be strict and evidence-based. Only use information present in the
conversation."""

RECOMMENDATION_USER_PROMPT = """\
User goal: {user_goal}

Conversation:
{chat_history}"""


class ProductRecommendationMetric(QuantitativeMetric):
    """Evaluates whether the agent recommends products that match
    the user's stated needs and provides specific, actionable
    suggestions.

    Final score: ``(relevance + specificity) / 2``, scaled to 0-5.
    """

    def __init__(self) -> None:
        super().__init__(
            name="product_recommendation",
            score_range=(0, 5),
            description="Evaluates relevance and specificity of product recommendations. "
            "Final score = (relevance + specificity) / 2, scaled to 0-5.",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: RecommendationSchema = llm.call(
            [
                {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": RECOMMENDATION_USER_PROMPT.format(
                        chat_history=format_chat_history(score_input.chat_history),
                        user_goal=score_input.user_goal or "N/A",
                    ),
                },
            ],
            schema=RecommendationSchema,
        )

        value = ((response.relevance + response.specificity) / 2) * 5

        reason = (
            f"Relevance: {response.relevance}, "
            f"Specificity: {response.specificity}. "
            f"{response.reason}"
        )

        return QuantResult(name=self.name, value=value, reason=reason)


# ── Upsell Behavior Metric


UPSELL_SYSTEM_PROMPT = """\
You are an impartial evaluator for an e-commerce shopping assistant.
Classify the agent's upselling and cross-selling behaviour in the conversation.

Choose exactly one label:
- "appropriate"       : the agent suggested complementary or upgraded products
                        at a natural moment without pressuring the customer.
- "too_pushy"         : the agent repeatedly pushed products the customer
                        did not ask for, ignored stated constraints (budget,
                        preference), or made the customer feel pressured.
- "missed_opportunity": the customer showed clear buying intent or asked a
                        question that opened a natural upsell window, but the
                        agent did not take it.
- "not_applicable"    : the conversation did not involve any product
                        recommendation or the customer showed no buying intent.

Be strict and evidence-based. Only use information present in the conversation."""

UPSELL_USER_PROMPT = """\
User goal: {user_goal}

Conversation:
{chat_history}"""


class UpsellBehaviorSchema(BaseModel):
    label: str
    reason: str


class UpsellBehaviorMetric(QualitativeMetric):
    """Classifies the agent's upsell and cross-sell behaviour.

    Labels: ``appropriate``, ``too_pushy``, ``missed_opportunity``,
    or ``not_applicable``.
    """

    def __init__(self) -> None:
        super().__init__(
            name="upsell_behavior",
            description="Classifies whether the agent's upselling was appropriate, "
            "too pushy, or a missed opportunity.",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        chat_history = format_chat_history(score_input.chat_history)

        response: UpsellBehaviorSchema = llm.call(
            [
                {"role": "system", "content": UPSELL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": UPSELL_USER_PROMPT.format(
                        chat_history=chat_history,
                        user_goal=score_input.user_goal or "N/A",
                    ),
                },
            ],
            schema=UpsellBehaviorSchema,
        )

        return QualResult(name=self.name, value=response.label, reason=response.reason)
