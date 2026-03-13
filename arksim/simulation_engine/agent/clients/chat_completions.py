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
            answer = self._extract_content(result)
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            logger.error(f"Error: Error calling chat completions API: {str(e)}")
            raise

    def _extract_content(self, result: dict[str, Any]) -> str:
        """Extract assistant text from API response.

        Supports:
        - OpenAI-style: result["choices"][0]["message"]["content"]
        - Anthropic-style: result["content"] as list of { "type": "text", "text": "..." }
        - Google-style: result["candidates"][0]["content"]["parts"][*]["text"]
        """
        # OpenAI-compatible format
        if "choices" in result:
            choices = result["choices"]
            if not choices:
                raise ValueError("API response has empty 'choices'")
            msg = choices[0].get("message") or choices[0].get("delta")
            if not msg:
                raise ValueError("API response choice has no 'message' or 'delta'")
            content = msg.get("content")
            if content is None and "delta" in choices[0]:
                content = choices[0]["delta"].get("content")
            if content is not None:
                return content if isinstance(content, str) else str(content)

        # Anthropic Messages API format: content = [{"type": "text", "text": "..."}]
        if "content" in result and isinstance(result["content"], list):
            parts = []
            for block in result["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            if parts:
                return "".join(parts)

        # Google Gemini API format: candidates[0].content.parts[*].text
        if "candidates" in result:
            candidates = result["candidates"]
            if not candidates:
                raise ValueError("API response has empty 'candidates'")
            content = candidates[0].get("content")
            if content and isinstance(content, dict):
                parts = content.get("parts") or []
                if isinstance(parts, list):
                    text_parts = [
                        p.get("text", "") if isinstance(p, dict) else "" for p in parts
                    ]
                    return "".join(text_parts)

        raise ValueError(
            "Unsupported response format: expected 'choices' (OpenAI), "
            "'content' list (Anthropic), or 'candidates' (Google). "
            f"Keys present: {list(result.keys())}"
        )
