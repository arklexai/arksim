# SPDX-License-Identifier: Apache-2.0
"""A2A server that embeds tool call data in a DataPart for arksim evaluation.

The server uses the standard A2A protocol, but each response includes both a
TextPart (the agent's answer) and a DataPart containing the tool calls made
during inference. arksim's A2A client extracts the DataPart automatically and
makes the tool calls available to the tool_call_behavior_failure metric.
"""

from __future__ import annotations

import os
import uuid

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Message,
    Part,
    Role,
    TextPart,
)

from .agent import Agent


class ToolCallAgentExecutor(AgentExecutor):
    """A2A executor that embeds tool calls in a DataPart on every response."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        context_id = context.context_id
        if not context_id:
            raise ValueError("Context ID is required")

        if context_id not in self._agents:
            self._agents[context_id] = Agent()

        user_input = context.get_user_input()
        if not user_input or user_input.strip() == "":
            raise ValueError("User input is required")

        answer, tool_calls = await self._agents[context_id].invoke(user_input)

        parts: list[Part] = [Part(root=TextPart(text=answer))]
        if tool_calls:
            parts.append(Part(root=DataPart(data={"tool_calls": tool_calls})))

        msg = Message(
            role=Role.agent,
            parts=parts,
            message_id=str(uuid.uuid4()),
            context_id=context_id,
        )
        await event_queue.enqueue_event(msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


if __name__ == "__main__":
    skill = AgentSkill(
        id="weather_tool_capture",
        name="Weather assistant with tool call capture",
        description=(
            "A weather assistant that reports tool calls via DataPart so "
            "arksim can evaluate tool usage against scenario assertions."
        ),
        tags=["weather", "tool-call-capture", "a2a"],
        examples=[
            "What is the weather in Tokyo?",
            "How is the weather in Paris today?",
        ],
    )

    public_agent_card = AgentCard(
        name="Weather Tool Capture Agent",
        description=(
            "A minimal weather agent that embeds tool call data in A2A "
            "DataPart responses for arksim tool call evaluation."
        ),
        url=os.getenv("A2A_SERVER_URL", "http://localhost:9998/"),
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ToolCallAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9998)
