# SPDX-License-Identifier: Apache-2.0
"""Custom HTTP agent connector.

A generic single-turn JSON-over-HTTP connector. The caller supplies a
request ``body`` template (with a ``{{message}}`` placeholder, optional
``{{chat_id}}`` and ``{{timestamp}}`` placeholders, and the reserved
response-extraction keys) and arksim posts a templated request each turn,
then renders the reply text from the JSON response. This keeps all field names in config so any
endpoint can be driven without code changes.

Extraction is configured by two reserved keys:

* ``response_path`` - a dot-path to the reply string. Required.
* ``response_extras`` - optional ``{path, label, template}`` entries that
  append other response fields to the turn, for a response that also
  carries a link list or product ids the evaluator needs to see. See
  ``agent/response_render.py``.

Gated assumptions (intentionally not hard-coded here):

* Field names and the extraction config come entirely from config; this
  client works with any dot-path and any extras.
* Conversation state: a fresh templated request is posted each turn. For
  endpoints that hold state server-side, place ``{{chat_id}}`` wherever
  the session/conversation id belongs in the template; it is substituted
  with this conversation's chat id (stable per conversation). Endpoints
  that want the full history resent each turn are not supported.
* Single-JSON responses only. SSE / streaming responses are a gated TODO.
* Tool calls and traces are out of scope; ``tool_calls`` is always empty.
* Structured related content (search results, recommendations, suggested
  questions) reaches the evaluator only as rendered text, via
  ``response_extras``; it is not carried as structured data on
  ``AgentResponse``.
"""

from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from arksim.config import (
    RESPONSE_EXTRAS_KEY,
    RESPONSE_PATH_KEY,
    AgentConfig,
    AgentType,
    CustomHTTPConfig,
)
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.agent.response_render import render_reply
from arksim.simulation_engine.agent.utils import rate_limit_handler
from arksim.simulation_engine.tool_types import AgentResponse
from arksim.tracing.context import request_headers

logger = logging.getLogger(__name__)

# Reserved keys / tokens in the body template. These are the FIXED config
# contract that the Go backend and frontend must match exactly.
MESSAGE_PLACEHOLDER = "{{message}}"
# Replaced with this conversation's chat id (BaseAgent.get_chat_id()). Lets
# endpoints that keep conversation state server-side (e.g. keyed on a session
# id) receive a stable-per-conversation, unique-across-conversations id.
CHAT_ID_PLACEHOLDER = "{{chat_id}}"
# Replaced with the current UTC time, generated fresh for each request. Lets
# endpoints that require a request timestamp receive a real value instead of
# a fixed one baked into the template.
TIMESTAMP_PLACEHOLDER = "{{timestamp}}"

# RESPONSE_PATH_KEY and RESPONSE_EXTRAS_KEY are part of the same contract and
# are imported from arksim.config, which owns them.


def _current_timestamp() -> str:
    """Current UTC time in ISO 8601 with milliseconds and a ``Z`` suffix.

    Example: ``2025-12-09T18:21:15.673Z``. The ``Z`` form is emitted (rather
    than ``+00:00``) because it is the common JavaScript ``toISOString()``
    shape that HTTP endpoints typically expect.
    """
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class CustomHTTPAgent(BaseAgent):
    """Generic custom HTTP agent implementation (templated JSON request)."""

    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        super().__init__(agent_config)
        if agent_config.agent_type != AgentType.CUSTOM_HTTP.value:
            raise ValueError("Agent config must be of type custom_http")
        self.config: CustomHTTPConfig = agent_config.api_config

        try:
            self.endpoint = self.config.get_endpoint()
            self.headers = self.config.get_headers()
            self.chat_id = str(uuid.uuid4())
            self._client = (http_client_factory or httpx.AsyncClient)(timeout=120)
        except Exception as e:
            logger.error(f"Error: Could not initialize custom HTTP agent: {e}")
            raise

    async def get_chat_id(self) -> str:
        """Get the chat ID."""
        return self.chat_id

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        if hasattr(self, "_client"):
            await self._client.aclose()

    @staticmethod
    def _substitute_placeholders(
        value: Any,  # noqa: ANN401
        substitutions: dict[str, str],
    ) -> Any:  # noqa: ANN401
        """Recursively replace known placeholders with their values.

        A string value equal to a placeholder token (e.g. ``{{message}}`` or
        ``{{chat_id}}``) is replaced wholesale by its mapped value; any other
        string is left as-is. Dicts and lists are walked so placeholders can
        live at any nesting depth.
        """
        if isinstance(value, str):
            return substitutions.get(value, value)
        if isinstance(value, dict):
            return {
                k: CustomHTTPAgent._substitute_placeholders(v, substitutions)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                CustomHTTPAgent._substitute_placeholders(item, substitutions)
                for item in value
            ]
        return value

    async def _post_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Per-request headers (trace context) layered over the agent's own
        # for this call only; self.headers is never mutated.
        headers = {**self.headers, **(request_headers.get() or {})}

        @rate_limit_handler
        async def _raw_post() -> httpx.Response:
            return await self._client.post(self.endpoint, headers=headers, json=payload)

        try:
            response = await _raw_post()
        except httpx.HTTPStatusError as e:
            # rate_limit_handler calls raise_for_status() on non-2xx; surface
            # the status and body so config mistakes are easy to diagnose.
            status = e.response.status_code
            body = e.response.text
            logger.error(f"CustomHTTP API HTTP {status}: {body}")
            raise RuntimeError(f"CustomHTTP API error: HTTP {status}: {body}") from e

        try:
            return response.json()
        except ValueError as e:
            logger.error("CustomHTTP API returned a non-JSON response")
            raise RuntimeError(
                f"CustomHTTP API returned a non-JSON response: {response.text[:500]}"
            ) from e

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        """Execute a user query against the custom HTTP endpoint.

        Deep-copies the body template, strips the reserved extraction keys,
        substitutes the ``{{message}}`` placeholder with the user query,
        ``{{chat_id}}`` with this conversation's chat id, and ``{{timestamp}}``
        with the current UTC time, POSTs the result, then renders the reply
        from ``response_path`` and any ``response_extras``. Tool calls are
        always empty (out of scope).
        """
        try:
            template = copy.deepcopy(self.config.body)
            response_path = template.pop(RESPONSE_PATH_KEY, None)
            response_extras = template.pop(RESPONSE_EXTRAS_KEY, None)
            payload = self._substitute_placeholders(
                template,
                {
                    MESSAGE_PLACEHOLDER: user_query,
                    CHAT_ID_PLACEHOLDER: self.chat_id,
                    TIMESTAMP_PLACEHOLDER: _current_timestamp(),
                },
            )

            result = await self._post_request(payload)
            reply = render_reply(result, response_path, response_extras)
            return AgentResponse(content=reply, tool_calls=[])

        except Exception as e:
            logger.error(f"Error: Error calling custom HTTP API: {str(e)}")
            raise
