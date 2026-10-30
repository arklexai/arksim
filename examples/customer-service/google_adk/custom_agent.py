# SPDX-License-Identifier: Apache-2.0
"""Google ADK variant of the customer-service agent (arksim BaseAgent wrapper).

Based on arksim/examples/integrations/google-adk/custom_agent.py, with the
shared store_tools.py tools attached via agent.py's build_agent(), and
tool-call capture so arksim can score trajectories the same way it does
for the OpenAI Agents SDK variant.

Install: pip install google-adk
Auth:    export GOOGLE_API_KEY="<your-key>"
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import build_agent, run_and_capture_tool_calls  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from store_tools import init_db  # noqa: E402

from arksim import AgentResponse  # noqa: E402
from arksim.config import AgentConfig  # noqa: E402
from arksim.simulation_engine.agent.base import BaseAgent  # noqa: E402


class GoogleADKAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        init_db()
        self._runner = InMemoryRunner(agent=build_agent(), app_name="arksim")
        self._user_id = f"arksim_{uuid.uuid4()}"
        self._session_id: str | None = None

    async def get_chat_id(self) -> str:
        if self._session_id is None:
            session = await self._runner.session_service.create_session(
                app_name="arksim",
                user_id=self._user_id,
            )
            self._session_id = session.id
        return self._session_id

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        if self._session_id is None:
            await self.get_chat_id()

        answer, tool_calls = await run_and_capture_tool_calls(
            self._runner, self._user_id, self._session_id, user_query
        )
        return AgentResponse(content=answer, tool_calls=tool_calls)
