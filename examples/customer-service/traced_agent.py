# SPDX-License-Identifier: Apache-2.0
"""Customer-service agent with automatic tool call capture via tracing.

Captures tool calls via arksim's ``ArksimTracingProcessor`` and the
OpenAI Agents SDK's ``TracingProcessor`` interface. Compare with
``custom_agent.py`` which returns tool calls in ``AgentResponse``.

The agent registers the processor once at module load. The simulator
sets routing context automatically, so no per-turn wrapping is needed.
The module loader caches modules by file path, so this registration
runs exactly once regardless of conversation count.

Tools and database setup are in ``tools.py`` (shared with the explicit
and A2A variants of this example).

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from agents import Agent, Runner, RunResult
from agents.tracing import add_trace_processor
from tools import AGENT_INSTRUCTIONS, TOOLS_LIST, init_db

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.openai import ArksimTracingProcessor

# Register once at module load. The module loader caches by file path,
# so this runs exactly once even when the simulator creates multiple
# agent instances for different conversations.
add_trace_processor(ArksimTracingProcessor())


class TracedToolCallAgent(BaseAgent):
    """Agent that captures tool calls via ArksimTracingProcessor.

    Tool calls are captured automatically by the processor's
    ``on_span_end`` callback. The simulator sets routing context
    (conversation_id, turn_id, receiver) before each ``execute()`` call.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        init_db()
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOLS_LIST,
        )
        self._last_result: RunResult | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        self._last_result = await Runner.run(self._agent, input=input_list)
        return self._last_result.final_output

    async def close(self) -> None:
        pass
