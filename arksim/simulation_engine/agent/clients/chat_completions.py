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

                tool_calls = self._extract_tool_calls(result)
                if tool_calls is None:
                    answer = self._extract_content(result)
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
                    self._build_assistant_tool_message(result)
                )
                self.conversation_history.extend(self._build_tool_results(tool_calls))

            raise RuntimeError(
                f"Agent exceeded {self.max_tool_call_rounds} tool-call rounds "
                "without producing a text response"
            )

        except Exception as e:
            logger.error(f"Error: Error calling chat completions API: {str(e)}")
            raise

    # ── Tool-call helpers ──

    def _extract_tool_calls(
        self, result: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Detect tool calls in an API response (OpenAI or Anthropic format).

        Returns a list of tool-call dicts, or ``None`` when the response
        contains no tool calls.
        """
        # OpenAI format
        if "choices" in result:
            msg = (result.get("choices") or [{}])[0].get("message", {})
            tc = msg.get("tool_calls")
            if tc:
                return tc

        # Anthropic format
        if "content" in result and isinstance(result["content"], list):
            tool_use_blocks = [
                block
                for block in result["content"]
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            if tool_use_blocks:
                return tool_use_blocks

        return None

    def _build_assistant_tool_message(self, result: dict[str, Any]) -> dict[str, Any]:
        """Build the assistant message to append for a tool-call turn."""
        # OpenAI format
        if "choices" in result:
            msg = (result.get("choices") or [{}])[0].get("message", {})
            return {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": msg["tool_calls"],
            }

        # Anthropic format
        return {"role": "assistant", "content": result["content"]}

    def _build_tool_results(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build synthetic tool-result messages for each tool call."""
        # Detect format: OpenAI tool calls have a top-level "function" key,
        # Anthropic tool_use blocks have "type": "tool_use".
        if tool_calls and tool_calls[0].get("type") == "tool_use":
            # Anthropic: single user message with tool_result content blocks
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": self.tool_call_result,
                        }
                        for tc in tool_calls
                    ],
                }
            ]

        # OpenAI: one tool message per call
        return [
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": self.tool_call_result,
            }
            for tc in tool_calls
        ]

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
