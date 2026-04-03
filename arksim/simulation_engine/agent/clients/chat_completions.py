# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import logging
import uuid
from typing import Any

import httpx

from arksim.config import (
    AgentConfig,
    AgentType,
    ChatCompletionsConfig,
)
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.agent.response_parsers import parse_response
from arksim.simulation_engine.agent.utils import rate_limit_handler
from arksim.simulation_engine.tool_types import AgentResponse
from arksim.tracing.propagation import inject_trace_context

logger = logging.getLogger(__name__)


class ChatCompletionsAgent(BaseAgent):
    """Chat completion agent implementation (OpenAI-compatible)."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        if agent_config.agent_type != AgentType.CHAT_COMPLETIONS.value:
            raise ValueError("Agent config must be of type chat_completions")
        self.config: ChatCompletionsConfig = agent_config.api_config

        try:
            self.chat_endpoint = self.config.get_endpoint()
            self.chat_headers = self.config.get_headers()
            self.chat_id = str(uuid.uuid4())
            self.conversation_history: list[dict[str, Any]] = []
            self._client = httpx.AsyncClient(
                timeout=120,
                event_hooks={"request": [inject_trace_context]},
            )

        except Exception as e:
            logger.error(f"Error: Could not initialize chat completion agent: {e}")
            raise

    async def get_chat_id(self) -> str:
        """Get the chat ID."""
        return self.chat_id

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        if hasattr(self, "_client"):
            await self._client.aclose()

    async def _post_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        @rate_limit_handler
        async def _raw_post() -> httpx.Response:
            return await self._client.post(
                self.chat_endpoint, headers=self.chat_headers, json=payload
            )

        response = await _raw_post()
        data = response.json()

        # Surface API error payloads (e.g. OpenAI "error", Anthropic "error")
        if response.status_code >= 400:
            err_msg = data.get("error", data)
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", err_msg)
            logger.error(f"Chat API HTTP {response.status_code}: {err_msg}")
            raise RuntimeError(f"Chat API error: {err_msg}")

        return data

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        """Execute user query using chat completions API."""
        metadata = kwargs.get("metadata")
        self.conversation_history.append({"role": "user", "content": user_query})

        try:
            payload_data = copy.deepcopy(self.config.body)
            payload_data.pop("messages", None)
            enable_metadata = payload_data.pop("enable_metadata", False)
            initial_messages = copy.deepcopy(self.config.body.get("messages", []))
            payload_data["messages"] = initial_messages + self.conversation_history

            if enable_metadata:
                if metadata:
                    payload_data["metadata"] = metadata
                else:
                    logger.warning(
                        "Metadata is not provided. Please provide metadata "
                        "to the chat completions API."
                    )

            result = await self._post_request(payload_data)
            response = parse_response(result)
            self.conversation_history.append(
                {"role": "assistant", "content": response.content}
            )
            return response

        except Exception as e:
            logger.error(f"Error: Error calling chat completions API: {str(e)}")
            raise
