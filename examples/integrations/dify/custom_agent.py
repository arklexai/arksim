# SPDX-License-Identifier: Apache-2.0
"""Dify integration for arksim.

Connects to a Dify Agent app via the Chat API using httpx and surfaces any
tool invocations the Dify-side agent emits as ``ToolCall`` instances on
the returned ``AgentResponse``.

Dify's blocking-mode response for Agent apps includes an
``agent_thoughts`` list. Each thought records a tool name, the JSON
arguments the agent passed, and the observation returned by the tool.
This wrapper parses that list into arksim's ``ToolCall`` shape so
``simulation.json`` gets populated tool-call data without any tracing.

Auth: export DIFY_API_KEY="<your-app-api-key>"
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import (
    AgentResponse,
    ToolCall,
    ToolCallSource,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.dify.ai/v1"
_REQUEST_TIMEOUT = httpx.Timeout(120.0)


def _parse_arguments(raw: object) -> dict[str, Any]:
    """Coerce Dify's ``tool_input`` (string or dict) into a JSON dict.

    Dify returns ``tool_input`` as either a JSON-encoded string or a
    pre-parsed object depending on the tool and the Dify version.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def _extract_tool_calls(payload: dict[str, Any]) -> list[ToolCall]:
    """Build ``ToolCall`` instances from a Dify Chat API response.

    Looks for ``agent_thoughts`` (Agent app blocking-mode shape). Each
    thought may bundle multiple tools in a single ``tool`` field
    (comma-separated) with a parallel ``tool_input`` dict, so we expand
    those into one ``ToolCall`` per tool.
    """
    thoughts = payload.get("agent_thoughts") or []
    if not isinstance(thoughts, list):
        return []

    tool_calls: list[ToolCall] = []
    for thought in thoughts:
        if not isinstance(thought, dict):
            continue
        tool_field = thought.get("tool")
        if not tool_field:
            continue
        tool_input = thought.get("tool_input")
        observation = thought.get("observation")

        # Dify joins multiple tools in one thought with commas; the
        # tool_input dict is keyed by tool name in that case.
        names = [n.strip() for n in str(tool_field).split(",") if n.strip()]
        for name in names:
            if isinstance(tool_input, dict) and name in tool_input:
                arguments = _parse_arguments(tool_input[name])
            else:
                arguments = _parse_arguments(tool_input)

            if isinstance(observation, dict) and name in observation:
                result: str | None = str(observation[name])
            elif observation is None:
                result = None
            else:
                result = str(observation)

            tool_calls.append(
                ToolCall(
                    id=str(thought.get("id") or uuid.uuid4()),
                    name=name,
                    arguments=arguments,
                    result=result,
                    source=ToolCallSource.DIFY,
                )
            )
    return tool_calls


class DifyAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        api_key = os.environ.get("DIFY_API_KEY")
        if not api_key:
            raise ValueError(
                "DIFY_API_KEY environment variable is required. "
                "Get it from API Access in your Dify app dashboard."
            )
        base_url = os.environ.get("DIFY_BASE_URL", _DEFAULT_BASE_URL)
        self._endpoint = f"{base_url.rstrip('/')}/chat-messages"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
        self._conversation_id: str | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        body: dict[str, object] = {
            "inputs": {},
            "query": user_query,
            "user": self._chat_id,
            "response_mode": "blocking",
        }
        if self._conversation_id is not None:
            body["conversation_id"] = self._conversation_id

        try:
            response = await self._client.post(
                self._endpoint, headers=self._headers, json=body
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Dify API at {self._endpoint}. "
                "Is the server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Request to Dify API at {self._endpoint} timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text
            if status == 401:
                raise RuntimeError(
                    "Dify API authentication failed. "
                    "Check your DIFY_API_KEY environment variable."
                ) from exc
            raise RuntimeError(f"Dify API returned HTTP {status}: {detail}") from exc
        data = response.json()

        new_id = data.get("conversation_id")
        if new_id is not None:
            self._conversation_id = new_id
        answer = data.get("answer")
        if answer is None:
            raise RuntimeError(
                f"Dify response missing 'answer' field. Response: {data}"
            )

        tool_calls = _extract_tool_calls(data)
        if not tool_calls:
            # Chatbot apps (vs Agent apps) never emit agent_thoughts. Log
            # once at debug so the user can spot the mismatch without
            # noise on every turn.
            logger.debug(
                "No agent_thoughts in Dify response; ensure you are running "
                "an Agent app with tools configured to capture tool calls."
            )
        return AgentResponse(content=answer, tool_calls=tool_calls)

    async def close(self) -> None:
        await self._client.aclose()
