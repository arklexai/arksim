# SPDX-License-Identifier: Apache-2.0
"""Dify integration for ArkSim.

Connects to a Dify Chatbot app via the Chat API using httpx.
Uses Dify's non-streaming (blocking) response mode.

Auth: export DIFY_API_KEY="<your-app-api-key>"
"""

from __future__ import annotations

import os
import uuid

import httpx

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent

_DEFAULT_BASE_URL = "https://api.dify.ai/v1"
_REQUEST_TIMEOUT = httpx.Timeout(120.0)


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

    async def execute(self, user_query: str, **kwargs: object) -> str:
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
        return answer

    async def close(self) -> None:
        await self._client.aclose()
