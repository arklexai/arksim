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
from arksim.simulation_engine.agent.clients.response_format import (
    build_assistant_tool_message,
    build_tool_results,
    detect_format,
    extract_content,
    extract_tool_calls,
)
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
        data = response.json()

        # Surface API error payloads (e.g. OpenAI "error", Anthropic "error")
        if response.status_code >= 400:
            err_msg = data.get("error", data)
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", err_msg)
            logger.error(f"Chat API HTTP {response.status_code}: {err_msg}")
            raise RuntimeError(f"Chat API error: {err_msg}")

        return data

    async def execute(self, user_query: str, **kwargs: object) -> str:
        """Execute user query using chat completions API."""
        metadata = kwargs.get("metadata")
        self.conversation_history.append({"role": "user", "content": user_query})
        try:
            payload_data = copy.deepcopy(self.config.body)
            payload_data.pop("messages", None)
            enable_metadata = payload_data.pop("enable_metadata", False)
            initial_messages = copy.deepcopy(self.config.body.get("messages", []))

            if enable_metadata:
                if metadata:
                    payload_data["metadata"] = metadata
                else:
                    logger.warning(
                        "Metadata is not provided. Please provide metadata to the chat completions API."
                    )

            for _round in range(self.max_tool_call_rounds):
                payload_data["messages"] = initial_messages + self.conversation_history
                result = await self._post_request(payload_data)
                fmt = detect_format(result)

                tool_calls = extract_tool_calls(fmt, result)
                if tool_calls is None:
                    answer = extract_content(fmt, result)
                    self.conversation_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    return answer

                logger.info(
                    "Agent responded with %d tool call(s), round %d",
                    len(tool_calls),
                    _round + 1,
                )
                self.conversation_history.append(
                    build_assistant_tool_message(fmt, result)
                )
                self.conversation_history.extend(
                    build_tool_results(fmt, tool_calls, self.tool_call_result)
                )

            raise RuntimeError(
                f"Agent exceeded {self.max_tool_call_rounds} tool-call rounds "
                "without producing a text response"
            )

        except Exception as e:
            logger.error(f"Error: Error calling chat completions API: {str(e)}")
            raise
