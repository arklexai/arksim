# SPDX-License-Identifier: Apache-2.0
"""Stateless-per-request wrapper around the Google ADK agent, for the
chat_completions HTTP endpoint.

Reuses google_adk/agent.py's build_agent() directly - no ADK agent-building
logic is duplicated here. Deliberately does NOT use build_agent.py's
run_and_capture_tool_calls(): that helper returns arksim ToolCall objects
(needed when arksim scores a trajectory), which the chat_completions
response shape has no field for anyway, and importing it would require
arksim installed in the server's container - unlike bank-insurance/
e-commerce's agent_server/, which never needs arksim. This file's own
_run() below is the same event loop minus the ToolCall bookkeeping.

Note on history: the chat_completions convention has the caller resend the
full transcript on every request (see chat_completions/server.py), unlike
arksim's own per-conversation ADK session in ../../google_adk/custom_agent.py
where one Python object and one ADK session_id persist for a whole
conversation. ADK's session model expects turns to be replayed one by one
into a persistent session_id, which doesn't fit a fresh request each time -
so prior history is folded into the single message sent to the model
instead of replayed turn by turn through the ADK runner.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "google_adk"))

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from google_adk.agent import build_agent  # noqa: E402
from store_tools import init_db  # noqa: E402


async def _run(
    runner: InMemoryRunner, user_id: str, session_id: str, prompt: str
) -> str:
    """Run one turn and return only the text answer - no tool-call capture."""
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_answer = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_answer = event.content.parts[0].text or final_answer
    return final_answer


class Agent:
    """Stateless-per-request wrapper around the shared ADK agent.

    A new instance (and a new ADK session) is created per HTTP request;
    ``history`` carries the prior turns the caller already sent.
    """

    def __init__(
        self,
        context_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        init_db()
        self.context_id = context_id or str(uuid.uuid4())
        self._history: list[dict[str, Any]] = list(history) if history else []
        self._runner = InMemoryRunner(agent=build_agent(), app_name="arksim")
        self._user_id = f"chat-completions_{uuid.uuid4()}"
        self._session_id: str | None = None

    async def invoke(self, question: str) -> str:
        """Process a user message and return the agent's text response.

        Args:
            question: The user's latest message.

        Returns:
            The agent's text response.
        """
        if self._session_id is None:
            session = await self._runner.session_service.create_session(
                app_name="arksim", user_id=self._user_id
            )
            self._session_id = session.id

        prompt = question
        if self._history:
            transcript = "\n".join(
                f"{turn['role']}: {turn['content']}" for turn in self._history
            )
            prompt = f"Conversation so far:\n{transcript}\n\nuser: {question}"

        return await _run(self._runner, self._user_id, self._session_id, prompt)
