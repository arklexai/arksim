# SPDX-License-Identifier: Apache-2.0
"""Builds the Google ADK LlmAgent for the customer-service example, and a
helper that runs one turn while capturing the tool calls made along the way.

Tools live in-process here, the way an ADK agent usually calls its
FunctionTools directly (not over the network) - the MCP exposure in
mcp_server.py is a separate, additional surface for external callers.

``build_agent()`` intentionally has no arksim dependency, even though
``run_and_capture_tool_calls()`` (below) returns arksim ``ToolCall``
objects - the import is deferred into that function's body. This lets
agent_server/core/agent.py (the chat_completions server)
call build_agent() without needing arksim installed in that container,
the same way bank-insurance/e-commerce's agent_server/ never needed it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adk_tools import ADK_TOOLS  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from store_tools import AGENT_INSTRUCTIONS  # noqa: E402

if TYPE_CHECKING:
    from arksim import ToolCall


def build_agent() -> LlmAgent:
    return LlmAgent(
        name="assistant",
        model="gemini-flash-lite-latest",
        instruction=AGENT_INSTRUCTIONS,
        tools=ADK_TOOLS,
    )


async def run_and_capture_tool_calls(
    runner: InMemoryRunner, user_id: str, session_id: str, user_query: str
) -> tuple[str, list[ToolCall]]:
    """Run one turn and capture the tool calls made along the way.

    ADK emits function calls and their responses as separate events in the
    run_async stream (unlike the OpenAI Agents SDK, which bundles a whole
    turn into one RunResult) - this pairs them up by id into ToolCall
    objects arksim can score against expected trajectories.
    """
    from arksim import ToolCall

    content = types.Content(role="user", parts=[types.Part(text=user_query)])
    pending: dict[str, ToolCall] = {}
    final_answer = ""

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        for call in event.get_function_calls():
            if call.id:
                pending[call.id] = ToolCall(
                    id=call.id, name=call.name or "", arguments=call.args or {}
                )
        for response in event.get_function_responses():
            if response.id and response.id in pending:
                pending[response.id].result = str(response.response)

        if event.is_final_response() and event.content and event.content.parts:
            final_answer = event.content.parts[0].text or final_answer

    return final_answer, list(pending.values())
