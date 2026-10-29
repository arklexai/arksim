# SPDX-License-Identifier: Apache-2.0
"""Stateful per-session agent for the A2A customer-service server.

Backend is selected via the AGENT_BACKEND env var ("openai" or "adk",
default "openai"). Both backends expose the same
``invoke(question) -> (answer, tool_calls)`` interface server.py expects,
serialized as plain dicts ready for artifact metadata (the ``source``
field is excluded because arksim sets it client-side).
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from arksim import ToolCall

# Add parent directory (for store_tools.py, google_adk/) and the
# openai_agents/ sibling (for tools.py) so both backends' imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "openai_agents"))


class _OpenAIAgent:
    """OpenAI Agents SDK backend.

    Maintains conversation history across turns so the model has
    multi-turn context.
    """

    def __init__(self) -> None:
        # Imported lazily so an environment with only google-adk installed
        # (AGENT_BACKEND=adk) never needs openai-agents to be present.
        from agents import Agent as SDKAgent
        from tools import AGENT_INSTRUCTIONS, TOOLS_LIST, init_db

        init_db()
        self._agent = SDKAgent(
            name="assistant",
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOLS_LIST,
        )
        self._history: list[dict[str, Any]] = []

    async def invoke(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        from agents import Runner

        from arksim.tracing.openai import extract_tool_calls

        self._history.append({"role": "user", "content": question})
        result = await Runner.run(self._agent, input=self._history)
        answer: str = result.final_output
        self._history.append({"role": "assistant", "content": answer})
        tool_calls: list[ToolCall] = extract_tool_calls(result)
        return answer, [
            tc.model_dump(exclude={"source"}, exclude_none=True) for tc in tool_calls
        ]


class _ADKAgent:
    """Google ADK backend, reusing google_adk's build_agent() and the
    tool-call capture helper - the same pieces custom_agent.py uses for
    arksim, so the two entry points (arksim, A2A) never duplicate the
    ADK session-management logic.
    """

    def __init__(self) -> None:
        # Imported lazily so the default "openai" backend never requires
        # google-adk to be installed.
        from google.adk.runners import InMemoryRunner
        from google_adk.agent import build_agent, run_and_capture_tool_calls
        from store_tools import init_db as adk_init_db

        adk_init_db()
        self._run_and_capture = run_and_capture_tool_calls
        self._runner = InMemoryRunner(agent=build_agent(), app_name="arksim")
        self._user_id = f"arksim_{uuid.uuid4()}"
        self._session_id: str | None = None

    async def invoke(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        if self._session_id is None:
            session = await self._runner.session_service.create_session(
                app_name="arksim", user_id=self._user_id
            )
            self._session_id = session.id

        answer, tool_calls = await self._run_and_capture(
            self._runner, self._user_id, self._session_id, question
        )
        return answer, [
            tc.model_dump(exclude={"source"}, exclude_none=True) for tc in tool_calls
        ]


Agent = _ADKAgent if os.getenv("AGENT_BACKEND", "openai") == "adk" else _OpenAIAgent
