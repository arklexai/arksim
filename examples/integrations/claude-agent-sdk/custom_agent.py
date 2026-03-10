# SPDX-License-Identifier: Apache-2.0
"""Claude Agent SDK integration for ArkSim.

Install: pip install claude-agent-sdk
Auth:    export ANTHROPIC_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from claude_agent_sdk import ClaudeAgentOptions, query

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class ClaudeAgentSDKAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._history: list[dict[str, str]] = []

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        self._history.append({"role": "user", "content": user_query})
        # query() is stateless; include prior turns as context in the prompt.
        if len(self._history) > 1:
            context = "\n".join(
                f"{m['role']}: {m['content']}" for m in self._history[:-1]
            )
            prompt = f"Conversation so far:\n{context}\n\nLatest message: {user_query}"
        else:
            prompt = user_query
        result = None
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(allowed_tools=[]),
        ):
            if hasattr(message, "result"):
                result = message.result
        answer = result or ""
        self._history.append({"role": "assistant", "content": answer})
        return answer
