# SPDX-License-Identifier: Apache-2.0
"""Rasa integration for ArkSim.

Requires a running Rasa server with the REST channel enabled.
Start Rasa:  rasa run --enable-api --cors "*"
Endpoint:    RASA_ENDPOINT env var or http://localhost:5005/webhooks/rest/webhook
"""

from __future__ import annotations

import os
import uuid

import httpx

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent

_DEFAULT_ENDPOINT = "http://localhost:5005/webhooks/rest/webhook"


class RasaAgent(BaseAgent):
    """Rasa agent wrapper using the REST webhook channel.

    Sends messages to a running Rasa server over HTTP. Rasa tracks
    conversation state per sender ID server-side, so no local history
    is needed. Multiple response messages are joined into a single string.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._endpoint = os.environ.get("RASA_ENDPOINT", _DEFAULT_ENDPOINT)
        self._client = httpx.AsyncClient(timeout=60)

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        response = await self._client.post(
            self._endpoint,
            json={"sender": self._chat_id, "message": user_query},
        )
        response.raise_for_status()

        messages = response.json()
        texts = [m["text"] for m in messages if "text" in m]
        return "\n".join(texts) if texts else ""

    async def close(self) -> None:
        await self._client.aclose()
