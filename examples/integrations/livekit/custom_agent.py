# SPDX-License-Identifier: Apache-2.0
"""LiveKit Agents integration for arksim.

Install:
    pip install 'arksim[livekit-agents]'
Auth:
    export OPENAI_API_KEY="<your-key>"          # required by LiveKit Cloud inference
    export LIVEKIT_API_KEY="<your-livekit-key>"
    export LIVEKIT_API_SECRET="<your-livekit-secret>"

Wires arksim's LiveKit tracing adapter into a LiveKit Agents text-mode
session with two mock tools (lookup_order, book_table). Running
``arksim simulate-evaluate`` produces a simulation.json whose
``tool_calls`` field is populated by the captured invocations.

Note on text-only mode: LiveKit Agents is voice-first, but
``AgentSession.run(user_input=..., input_modality="text")`` exposes a
text-in / text-out path that does not require an audio room. We start
the session without a ``room`` argument so no RTC plumbing is needed.
"""

from __future__ import annotations

import os
import uuid

from livekit.agents.inference import LLM as InferenceLLM
from livekit.agents.voice import Agent, AgentSession
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.livekit import ArksimLiveKitHandler

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")


class LiveKitAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._handler = ArksimLiveKitHandler()
        self._agent = Agent(
            instructions="You are a helpful assistant.",
            tools=[lookup_order, book_table],
            llm=InferenceLLM(model=f"openai/{_MODEL}"),
        )
        self._session = AgentSession()
        self._handler.attach_to(self._session)
        self._started = False

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if not self._started:
            await self._session.start(self._agent)
            self._started = True
        result = await self._session.run(user_input=user_query, input_modality="text")
        return str(result.final_output)
