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
from arksim.simulation_engine.agent.utils import rate_limit_handler

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

        except Exception as e:
            logger.error(f"Error: Could not initialize chat completion agent: {e}")
            raise

    async def get_chat_id(self) -> str:
        """Get the chat ID."""
        return self.chat_id

    async def _post_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        @rate_limit_handler
        async def _raw_post() -> httpx.Response:
            async with httpx.AsyncClient(timeout=120) as client:
                return await client.post(
                    self.chat_endpoint, headers=self.chat_headers, json=payload
                )

        response = await _raw_post()
        return response.json()

    async def execute(self, user_query: str, **kwargs: object) -> str:
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
                        "Metadata is not provided. Please provide metadata to the chat completions API."
                    )

            result = await self._post_request(payload_data)
            answer = result["choices"][0]["message"]["content"]
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            logger.error(f"Error: Error calling chat completions API: {str(e)}")
            raise
