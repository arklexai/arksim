# SPDX-License-Identifier: Apache-2.0
"""Dify integration for ArkSim.

Install: pip install dify-client
Auth:    export DIFY_API_KEY="<your-app-api-key>"
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dify_client import ChatClient

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


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
        self._client = ChatClient(api_key)
        self._client.base_url = os.environ.get(
            "DIFY_BASE_URL", "https://api.dify.ai/v1"
        )
        self._conversation_id: str | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        call_kwargs: dict[str, object] = {
            "inputs": {},
            "query": user_query,
            "user": self._chat_id,
            "response_mode": "blocking",
        }
        if self._conversation_id is not None:
            call_kwargs["conversation_id"] = self._conversation_id

        response = await asyncio.to_thread(
            self._client.create_chat_message, **call_kwargs
        )
        response.raise_for_status()
        result = response.json()

        self._conversation_id = result.get("conversation_id")
        answer = result.get("answer")
        if answer is None:
            raise RuntimeError(
                f"Dify response missing 'answer' field. Response: {result}"
            )
        return answer
