# SPDX-License-Identifier: Apache-2.0
"""Security helpers for the Claude Code MCP server.

Two responsibilities:

* Redact recognised secret patterns from any subprocess output that
  travels back to the LLM client (where it would persist in chat
  transcripts and MCP-client telemetry).
* Validate path arguments accepted from LLM-controlled tool inputs so
  that ``cwd``, ``directory``, and ``simulation_file_path`` cannot
  escape into ``/``, ``$HOME``, or arbitrary parents.

The module is dependency-free so it can be imported by both the CLI
wrapper and the FastMCP server without circular issues.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Maximum bytes captured per subprocess stream before truncation.
# 256 KB is enough for typical arksim output; longer streams keep their
# tail (most-recent slice).
MAX_CAPTURED_BYTES = 256 * 1024

# Patterns to redact from subprocess output before returning to the LLM.
# These match common shapes for OpenAI / Anthropic / Google API keys and
# Authorization headers. Each pattern intentionally requires a length
# that distinguishes it from short user-supplied identifiers like
# scenario IDs.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI and Anthropic-style keys.
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"),
    # Google.
    re.compile(r"\bAIza[A-Za-z0-9_-]{35}"),
    # Authorization headers and X-Api-Key.
    re.compile(r"(?i)Authorization:\s*Bearer\s+\S+"),
    re.compile(r"(?i)x-api-key:\s*\S+"),
    # AWS access keys.
    re.compile(r"\b(AKIA|ASIA|AGPA|AROA|AIDA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b"),
    # GitHub PATs and OAuth tokens.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # Slack tokens.
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    # HuggingFace.
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    # Stripe live keys.
    re.compile(r"\b(?:sk|rk|pk)_live_[A-Za-z0-9]{16,}\b"),
    # Twilio SIDs.
    re.compile(r"\bAC[0-9a-fA-F]{32}\b"),
    re.compile(r"\bSK[0-9a-fA-F]{32}\b"),
    # JWTs.
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    # Provider-prefixed env-style assignments.
    re.compile(
        r"(?i)(?:OPENAI|ANTHROPIC|GOOGLE|AZURE|RASA|GEMINI|XAI|DEEPSEEK|MISTRAL|COHERE|GROQ|HUGGINGFACE|HF|TOGETHER|REPLICATE|FIREWORKS|PERPLEXITY|VOYAGE|ELEVENLABS)_(?:API_)?KEY[\s]*[:=][\s]*[A-Za-z0-9_\-+/=]{8,}"
    ),
    # Generic *_SECRET / *_TOKEN / *_PASSWORD env-style assignments.
    # Trailing value is bounded to typical credential characters so
    # a JSON tail like `"}, "next":` is not gobbled.
    re.compile(
        r"(?i)\b[A-Z][A-Z0-9_]*_(SECRET|TOKEN|PASSWORD|PASSWD|PWD)[\s]*[:=][\s]*[A-Za-z0-9_\-+/=]{8,}"
    ),
)


def redact_secrets(text: str) -> str:
    """Replace recognised secret patterns in ``text`` with ``[REDACTED]``.

    Returns ``text`` unchanged if it is empty or has no matches.
    """
    if not text:
        return text
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def tail_capture(text: str, max_bytes: int = MAX_CAPTURED_BYTES) -> str:
    """Return the tail of ``text`` no larger than ``max_bytes``.

    The tail (most recent output) is more useful than the head when the
    user is debugging a failure, so we drop the prefix. The byte count
    uses UTF-8 encoding; the result is always a valid UTF-8 string.
    """
    if not text:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[-max_bytes:]
    return f"...[truncated]\n{truncated.decode('utf-8', errors='replace')}"


class PathValidationError(ValueError):
    """Raised when a path argument fails validation."""


def validate_path_arg(
    candidate: str | os.PathLike[str] | None,
    *,
    allow_none: bool = True,
    require_exists: bool = True,
    require_dir: bool = False,
    require_file: bool = False,
) -> Path | None:
    """Validate and resolve a path argument from an MCP tool input.

    Rejections:
        * NUL bytes anywhere in the input.
        * Empty strings.
        * Paths that resolve to the filesystem root (``/``).
        * Paths that resolve to the user's home directory directly
          (``$HOME`` or ``~``). Subdirectories of ``$HOME`` are allowed.
        * Non-existent paths when ``require_exists`` is set.
        * Wrong type (file vs directory) when the matching flag is set.

    The returned ``Path`` is fully resolved (symlinks followed,
    ``~`` expanded). Returns ``None`` only when ``candidate`` is ``None``
    and ``allow_none`` is true.
    """
    if candidate is None:
        if allow_none:
            return None
        raise PathValidationError("path is required and cannot be None")

    if not isinstance(candidate, str | os.PathLike):
        raise PathValidationError(
            f"path must be a string, got {type(candidate).__name__}"
        )

    s = os.fspath(candidate)
    if not s:
        raise PathValidationError("path is empty")
    if "\x00" in s:
        raise PathValidationError("path contains NUL byte")

    try:
        resolved = Path(s).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise PathValidationError(f"could not resolve path {s!r}: {exc}") from exc

    if resolved == Path(resolved.anchor or "/"):
        raise PathValidationError(f"path {resolved} resolves to the filesystem root")

    home = Path.home().resolve()
    if resolved == home:
        raise PathValidationError(
            f"path {resolved} resolves to the user home directory; "
            f"specify a project subdirectory instead"
        )

    if require_exists and not resolved.exists():
        raise PathValidationError(f"path does not exist: {resolved}")
    if require_dir and resolved.exists() and not resolved.is_dir():
        raise PathValidationError(f"path is not a directory: {resolved}")
    if require_file and resolved.exists() and not resolved.is_file():
        raise PathValidationError(f"path is not a file: {resolved}")

    return resolved


def is_inside(path: Path, root: Path) -> bool:
    """Return True if ``path`` is at or under ``root`` after resolution."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
