# SPDX-License-Identifier: Apache-2.0
"""Smolagents tracing adapter (source=ToolCallSource.SMOLAGENTS).

Verified against smolagents 1.24.0. Pass via the ``step_callbacks`` list on
``MultiStepAgent``; the SDK invokes ``__call__`` after each ``ActionStep``.
Only ``ActionStep`` instances carry tool calls; other step types
(``PlanningStep``, ``TaskStep``, ``FinalAnswerStep``, ``SystemPromptStep``)
are ignored. The adapter is stateless across emissions.

When a callback is registered for ``ActionStep`` via the list-form
``step_callbacks=[...]``, smolagents already filters by step class through
``CallbackRegistry``. The explicit ``isinstance`` check below preserves
correctness if a user instead registers this callback for a broader step
type via the dict form.

Tool wrapping fallback for users with framework-instantiated tools that
aren't observable via ``step_callbacks`` is documented in the example
README, not shipped here.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from smolagents.memory import ActionStep, MemoryStep
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimSmolagentsCallback requires the 'smolagents' extra. "
        "Install with: pip install 'arksim[smolagents]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter

logger = logging.getLogger(__name__)


class ArksimSmolagentsCallback(BaseTracingAdapter):
    """Smolagents step callback capturing tool calls into arksim.

    Pass via ``MultiStepAgent(step_callbacks=[ArksimSmolagentsCallback()])``.
    Stateless across emissions.
    """

    def __call__(
        self,
        memory_step: MemoryStep,
        agent: Any = None,  # noqa: ANN401, ARG002  (signature fixed by smolagents)
    ) -> None:
        if not isinstance(memory_step, ActionStep):
            return
        tool_calls = memory_step.tool_calls or []
        observations = memory_step.observations
        result = str(observations) if observations is not None else None
        for tc in tool_calls:
            self._submit(
                ToolCall(
                    id=str(tc.id) if tc.id else "",
                    name=tc.name or "",
                    arguments=parse_tool_arguments(tc.arguments),
                    result=result,
                    source=ToolCallSource.SMOLAGENTS,
                )
            )
