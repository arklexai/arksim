# SPDX-License-Identifier: Apache-2.0
"""
Provides the abstract base for all evaluation metrics, both built-in and user-defined custom metrics.
"""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str


def format_chat_history(messages: list[ChatMessage]) -> str:
    """Serialise a message list to the standard prompt string."""
    return "".join(f"{m.role}: {m.content}\n" for m in messages)


class QuantResult(BaseModel):
    name: str
    value: float
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class QualResult(BaseModel):
    """Result of a qualitative metric evaluation (e.g. agent behavior failure)."""

    name: str
    value: str
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class ScoreInput(BaseModel):
    """Input context provided to custom metrics for each turn.

    Extra fields are allowed — users can pass additional context
    and access it via score_input.model_extra.
    """

    model_config = ConfigDict(extra="allow")

    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Full conversation up to and including this turn",
    )
    current_turn: list[ChatMessage] = Field(
        default_factory=list,
        description="Only the user + assistant exchange for this turn",
    )
    knowledge: str = ""
    user_goal: str = ""
    profile: str = ""


class QuantitativeMetric(abc.ABC):
    """Abstract base class for numeric score metrics.

    Args:
        name: The name of the metric. If not provided, uses the class name as default.
        score_range: A (min, max) tuple describing the possible output range of
            :meth:`score`. Defaults to ``(1, 5)`` (the same scale as the built-in
            metrics). The range is used to normalise the value for the aggregated
            turn score and to derive a per-metric failure threshold
            (60 % of *max*).

    Example:
        >>> from arksim.evaluator import QuantitativeMetric, ScoreInput, QuantResult
        >>>
        >>> class MyMetric(QuantitativeMetric):
        >>>     def __init__(self, name: str, threshold: float = 0.5):
        >>>         super().__init__(
        >>>             name=name,
        >>>             score_range=(0, 5),
        >>>             additional_input={"threshold": threshold},
        >>>         )
        >>>
        >>>     def score(self, score_input: ScoreInput) -> QuantResult:
        >>>         # Access standard fields: score_input.chat_history, score_input.knowledge, etc.
        >>>         # Access extra input, example: score_input.model_extra["threshold"]
        >>>
        >>>         return QuantResult(
        >>>             value=0,
        >>>             name=self.name,
        >>>             reason="Optional reason for the score"
        >>>         )
    """

    def __init__(
        self,
        name: str | None = None,
        score_range: tuple[float, float] = (1, 5),
        additional_input: dict[str, Any] | None = None,
        description: str = "",
    ) -> None:
        self.name = name if name is not None else self.__class__.__name__
        self.score_range = score_range
        self.additional_input = additional_input or {}
        self.description = description

    @abc.abstractmethod
    def score(self, score_input: ScoreInput) -> QuantResult:
        raise NotImplementedError()


class QualitativeMetric(abc.ABC):
    """Abstract base for qualitative (categorical-label) metrics.

    Unlike QuantitativeMetric which returns a numeric score, qualitative
    metrics classify behaviour into a categorical label.

    Example:
        >>> from arksim.evaluator import QualitativeMetric, QualResult, ScoreInput
        >>>
        >>> class MyQualMetric(QualitativeMetric):
        >>>     def __init__(self):
        >>>         super().__init__(
        >>>             name="my_qual_metric",
        >>>             label_colors={
        >>>                 "pass": "#22c55e",  # green
        >>>                 "fail": "#ef4444",  # red
        >>>             },
        >>>         )
        >>>
        >>>     def evaluate(self, score_input: ScoreInput) -> QualResult:
        >>>         return QualResult(
        >>>             name=self.name,
        >>>             value="pass",
        >>>             reason="Optional reason",
        >>>         )
    """

    def __init__(
        self,
        name: str | None = None,
        description: str = "",
        label_colors: dict[str, str] | None = None,
    ) -> None:
        self.name = name if name is not None else self.__class__.__name__
        self.description = description
        self.label_colors: dict[str, str] = label_colors or {}

    @abc.abstractmethod
    def evaluate(self, score_input: ScoreInput) -> QualResult:
        raise NotImplementedError()
