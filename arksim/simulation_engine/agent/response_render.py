# SPDX-License-Identifier: Apache-2.0
"""Render a custom_http agent's JSON response into assistant-turn text.

An assistant turn is a plain string (``AgentResponse.content``), so a response
carrying the reply in one field and supporting data in others (a link list,
citations, recommendations) loses everything but the reply unless the rest is
rendered into the text. ``response_path`` names the reply field and
``response_extras`` names what to append::

    response_path: raw_text
    response_extras:
      - path: links
        label: Links
        template: "{title}: https://{id}.com"

renders::

    Reset your password from the account page.

    Links:
    - Reset your password: https://a1b2c3.com
    - Enable 2FA: https://d4e5f6.com

Gated assumptions:

* Configuration is data, not code. A ``template`` is constant text with
  ``{name}`` holes filled by substitution: no operators, filters, calls or
  conditionals, so output is bounded by input and no config can make this
  renderer allocate more than it was handed. Holes are filled by concatenation,
  never ``str.format`` or ``%``, so a format spec is not a hole and
  ``{:1000000000}`` is literal text.
* Shape comes from the response, not the config. One item, a list of items, or a
  map of key to value all render from the same ``path``, so an endpoint that
  returns a single link today and three tomorrow needs no config change.
* A missing or empty field drops its whole section. That is how prose is
  conditioned on data being present without needing a conditional.
* The reply stays strict: a ``response_path`` that does not resolve to a string
  is a configuration error, because the alternative is an empty turn.
"""

from __future__ import annotations

import re
from typing import Any

# The reply becomes a turn fed to an LLM judge, so it is bounded for token cost.
# Both limits truncate rather than raise: an over-long response is the agent's
# doing, not a misconfiguration, and a simulation should keep running.
MAX_RENDERED_CHARS = 100_000
MAX_ITEMS_PER_SECTION = 50

# A hole is a path in braces. Anything else between braces is literal text,
# which is what keeps a format spec from being interpreted as one.
_HOLE = re.compile(r"\{([A-Za-z0-9_.\[\]]*)\}")

_MISSING = object()


class ResponseRenderError(RuntimeError):
    """Raised when a response cannot be rendered into a usable reply."""


def _lookup(value: Any, segment: str) -> Any:  # noqa: ANN401
    if isinstance(value, dict):
        return value.get(segment, _MISSING)
    if isinstance(value, list):
        try:
            index = int(segment)
        except ValueError:
            return _MISSING
        return value[index] if -len(value) <= index < len(value) else _MISSING
    return _MISSING


def _resolve(data: Any, path: str) -> tuple[list[Any], bool]:  # noqa: ANN401
    """Resolve a dot-path leniently, fanning out on ``[]`` segments.

    Returns the values found and whether any segment fanned out, which the
    caller needs to tell ``links[]`` (already a sequence of items) from
    ``links`` (one value that may itself be a list).
    """
    values: list[Any] = [data]
    fanned = False
    for raw in path.split("."):
        segment, iterate = (raw[:-2], True) if raw.endswith("[]") else (raw, False)
        found: list[Any] = []
        for value in values:
            got = _lookup(value, segment) if segment else value
            if got is _MISSING or got is None:
                continue
            if iterate and isinstance(got, list | tuple):
                found.extend(got)
            else:
                found.append(got)
        values = found
        fanned = fanned or iterate
    return values, fanned


def _as_items(values: list[Any], fanned: bool, pairs: bool) -> list[Any]:
    """Normalize a resolved value into the items a section renders."""
    if pairs:
        return [
            {"key": key, "value": item}
            for value in values
            if isinstance(value, dict)
            for key, item in value.items()
        ]
    if not fanned and len(values) == 1 and isinstance(values[0], list | tuple):
        return list(values[0])
    return values


def _text(value: Any) -> str:  # noqa: ANN401
    """Readable text for a value the config gave no template for."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_text(item)}" for key, item in value.items() if item is not None
        )
    if isinstance(value, list | tuple):
        return ", ".join(_text(item) for item in value)
    return str(value)


def _substitute(template: str, item: Any) -> str:  # noqa: ANN401
    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if not name:
            return _text(item)
        values, fanned = _resolve(item, name)
        found = _as_items(values, fanned, pairs=False)
        return ", ".join(_text(value) for value in found)

    return _HOLE.sub(fill, template)


def _render_section(response: Any, extra: dict[str, Any]) -> str:  # noqa: ANN401
    path = str(extra.get("path") or "")
    template = str(extra.get("template") or "")
    label = str(extra.get("label") or "").strip()
    if not path:
        return ""

    # {key}/{value} in the template is what asks for a map to be walked as pairs
    # rather than treated as one record whose fields the holes address.
    pairs = "{key}" in template or "{value}" in template
    values, fanned = _resolve(response, path)
    items = _as_items(values, fanned, pairs)

    lines = []
    for item in items[:MAX_ITEMS_PER_SECTION]:
        text = (_substitute(template, item) if template else _text(item)).strip()
        if text:
            lines.append(text)
    if not lines:
        return ""
    hidden = len(items) - MAX_ITEMS_PER_SECTION
    if hidden > 0:
        lines.append(f"... and {hidden} more")

    if len(lines) == 1:
        return f"{label}: {lines[0]}" if label else lines[0]
    body = "\n".join(f"- {line}" for line in lines)
    return f"{label}:\n{body}" if label else body


def extract_reply_text(response: Any, response_path: str) -> str:  # noqa: ANN401
    """Walk a dot-path over parsed JSON and return the reply string.

    Supports list indices as numeric path segments, e.g.
    ``choices.0.message.content``. Raises ``ResponseRenderError`` if any segment
    is missing or the final value is not a string.
    """
    current = response
    traversed: list[str] = []
    for segment in response_path.split("."):
        traversed.append(segment)
        location = ".".join(traversed)
        if isinstance(current, dict):
            if segment not in current:
                raise ResponseRenderError(
                    f"CustomHTTP response_path '{response_path}' not found: "
                    f"key '{segment}' missing at '{location}'"
                )
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
            except ValueError as e:
                raise ResponseRenderError(
                    f"CustomHTTP response_path '{response_path}' invalid: "
                    f"segment '{segment}' is not a list index at '{location}'"
                ) from e
            if index < 0 or index >= len(current):
                raise ResponseRenderError(
                    f"CustomHTTP response_path '{response_path}' out of range: "
                    f"index {index} at '{location}' (list len {len(current)})"
                )
            current = current[index]
        else:
            raise ResponseRenderError(
                f"CustomHTTP response_path '{response_path}' invalid: cannot "
                f"descend into a {type(current).__name__} at '{location}'"
            )

    if not isinstance(current, str):
        raise ResponseRenderError(
            f"CustomHTTP response_path '{response_path}' did not resolve to a "
            f"string (got {type(current).__name__})"
        )
    return current


def render_reply(
    response: Any,  # noqa: ANN401
    response_path: str,
    response_extras: list[dict[str, Any]] | None = None,
) -> str:
    """Render the reply text plus a section per configured extra."""
    parts = [extract_reply_text(response, response_path)]
    for extra in response_extras or []:
        if isinstance(extra, dict):
            section = _render_section(response, extra)
            if section:
                parts.append(section)

    reply = "\n\n".join(part for part in parts if part.strip()).strip()
    if len(reply) > MAX_RENDERED_CHARS:
        reply = reply[:MAX_RENDERED_CHARS].rstrip() + "\n... (truncated)"
    return reply


__all__ = [
    "MAX_ITEMS_PER_SECTION",
    "MAX_RENDERED_CHARS",
    "ResponseRenderError",
    "extract_reply_text",
    "render_reply",
]
