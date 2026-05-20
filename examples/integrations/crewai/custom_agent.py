# SPDX-License-Identifier: Apache-2.0
"""CrewAI integration for arksim.

Install:
    pip install 'arksim[crewai]'
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires arksim's CrewAI tracing adapter into a single-agent ``Crew`` with
two mock tools (lookup_order, book_table). Running
``arksim simulate-evaluate`` produces a simulation.json whose
``tool_calls`` field is populated by the captured invocations.
"""

from __future__ import annotations

import os
import uuid

from crewai import Agent as CrewAgent
from crewai import Crew, Task
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.crewai import ArksimCrewEventListener

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")


class CrewAIAgent(BaseAgent):
    """CrewAI agent with conversation history and tool-call tracing.

    CrewAI is task-oriented, so the agent threads prior turns into each
    new ``Task`` description to preserve conversational memory.
    Instantiating ``ArksimCrewEventListener`` registers its handlers on
    the global ``crewai_event_bus`` eagerly; no explicit ``Crew``
    constructor argument is required.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        # Instantiating registers the listener on the global event bus.
        self._listener = ArksimCrewEventListener()
        self._agent = CrewAgent(
            role="Customer service assistant",
            goal="Help customers with their orders and bookings.",
            backstory=(
                "An efficient customer service assistant for an online "
                "retailer that also takes restaurant reservations."
            ),
            tools=[lookup_order, book_table],
            allow_delegation=False,
            verbose=False,
            llm=_MODEL,
        )
        self._history: list[dict[str, str]] = []

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        self._history.append({"role": "user", "content": user_query})
        context = "\n".join(f"{m['role']}: {m['content']}" for m in self._history[:-1])
        description = (
            f"Conversation so far:\n{context}\n\nLatest message: {user_query}"
            if context
            else user_query
        )
        task = Task(
            description=description,
            expected_output="A short, helpful response.",
            agent=self._agent,
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        result = await crew.kickoff_async()
        answer = result.raw
        self._history.append({"role": "assistant", "content": answer})
        return answer
