# SPDX-License-Identifier: Apache-2.0
"""Claude Agent SDK integration for arksim.

Install:
    pip install 'arksim[claude-agent]'
Auth:
    export ANTHROPIC_API_KEY="<your-key>"

Wires arksim's Claude Agent SDK tracing adapter into a ``ClaudeSDKClient``
with two mock tools (lookup_order, book_table) registered through an
in-process SDK MCP server. Running ``arksim simulate-evaluate`` produces a
simulation.json whose ``tool_calls`` field is populated by the captured
invocations.
"""

from __future__ import annotations

import uuid

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
)
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.claude_agent_sdk import ArksimClaudeHooks


class ClaudeAgentSDKAgent(BaseAgent):
    """Claude Agent SDK client wired with arksim's PostToolUse hook.

    The two mock tools are exposed to Claude via an in-process SDK MCP
    server named ``arksim_tools``. Inside the SDK they are presented as
    ``mcp__arksim_tools__lookup_order`` and
    ``mcp__arksim_tools__book_table``; ``allowed_tools`` whitelists those
    fully-qualified names so the client only invokes our mock tools.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._hooks = ArksimClaudeHooks()
        server = create_sdk_mcp_server(
            name="arksim_tools",
            tools=[lookup_order, book_table],
        )
        self._client = ClaudeSDKClient(
            options=ClaudeAgentOptions(
                mcp_servers={"arksim_tools": server},
                allowed_tools=[
                    "mcp__arksim_tools__lookup_order",
                    "mcp__arksim_tools__book_table",
                ],
                hooks=self._hooks.hooks_dict(),
            ),
        )
        self._connected = False

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if not self._connected:
            await self._client.connect()
            self._connected = True

        await self._client.query(user_query)
        result = ""
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result += block.text
            elif isinstance(message, ResultMessage):
                break
        return result
