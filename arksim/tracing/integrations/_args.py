# SPDX-License-Identifier: Apache-2.0
"""Argument normalization for SDK-emitted tool-call inputs.

SDKs vary in how they hand tool arguments to callbacks: LangChain emits a
JSON string, Claude Agent SDK and Google ADK hand a dict directly, Strands
gives a ``tool_use`` mapping. ``parse_tool_arguments`` normalizes all of
these to a plain ``dict[str, Any]`` so adapters stay one-liners and the
downstream ``ToolCall.arguments`` schema is consistent across sources.

Scalar or list JSON values get wrapped in ``{"_value": <parsed>}`` rather
than discarded - they're rare in real tool schemas but should round-trip.
"""

from __future__ import annotations

import json
from typing import Any


def parse_tool_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize SDK tool-call arguments to a dict.

    - ``dict`` -> returned as-is
    - ``None`` or empty string -> ``{}``
    - JSON string parsing to a ``dict`` -> the parsed dict
    - JSON string parsing to a non-dict (number, list, ``null``, scalar)
      -> ``{"_value": <parsed>}``
    - non-JSON string -> ``{"_value": <raw>}``
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"_value": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}
