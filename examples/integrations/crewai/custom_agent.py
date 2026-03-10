# SPDX-License-Identifier: Apache-2.0
"""CrewAI integration for ArkSim.

Install: pip install crewai
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from crewai import Agent as CrewAgent
from crewai import Crew, Task

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class CrewAIAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._agent = CrewAgent(
            role="Assistant",
            goal="Answer user questions helpfully",
            backstory="You are a helpful assistant.",
            llm="gpt-5.1",
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
            expected_output="A clear, helpful answer",
            agent=self._agent,
        )
        crew = Crew(agents=[self._agent], tasks=[task], verbose=False)
        result = await crew.kickoff_async()
        answer = result.raw
        self._history.append({"role": "assistant", "content": answer})
        return answer
