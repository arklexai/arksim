# SPDX-License-Identifier: Apache-2.0
"""A2A server for the customer-service example.

Surfaces tool calls via the arksim A2A tool capture extension
(``A2AToolCaptureExtension``). The server declares the extension in its
AgentCard and emits tool call data in Task artifact metadata.

Per A2A spec section 3.7, task outputs are delivered through Artifacts
on a Task, not through Messages.
"""

from __future__ import annotations

import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    Part,
    TextPart,
)

from arksim import A2AToolCaptureExtension

from .agent import Agent

_TOOL_CAPTURE_EXTENSION = AgentExtension(
    uri=A2AToolCaptureExtension.URI,
    description=(
        "Tool calls executed during inference are surfaced in artifact "
        "metadata for arksim evaluation."
    ),
    required=False,
)


class CustomerServiceExecutor(AgentExecutor):
    """A2A executor that surfaces tool calls via Task artifacts."""

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

        # Mark the extension as activated so the framework echoes it back
        # in the response header per the A2A extension negotiation protocol.
        # Only emit tool call metadata when the client requested the
        # extension; this is least-privilege by default (see the Security
        # section in the tool-call-capture docs for deployment guidance).
        emit_tool_calls = A2AToolCaptureExtension.URI in context.requested_extensions
        if emit_tool_calls:
            context.add_activated_extension(A2AToolCaptureExtension.URI)

        if not context.task_id:
            raise ValueError(f"context.task_id is required (context_id={context_id})")
        updater = TaskUpdater(event_queue, context.task_id, context_id)
        if context.current_task is None:
            await updater.submit()
        await updater.start_work()

        metadata: dict[str, object] = {}
        extensions: list[str] = []
        if emit_tool_calls and tool_calls:
            metadata[A2AToolCaptureExtension.METADATA_KEY] = tool_calls
            extensions.append(A2AToolCaptureExtension.URI)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=answer))],
            # Artifact.name is a human-readable label, not consumed by arksim.
            metadata=metadata or None,
            extensions=extensions or None,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


if __name__ == "__main__":
    skill = AgentSkill(
        id="customer_service_tool_capture",
        name="Customer service with tool call capture",
        description=(
            "A customer service agent backed by SQLite that reports tool "
            "calls via the arksim A2A tool capture extension."
        ),
        tags=["customer-service", "tool-call-capture", "a2a"],
        examples=[
            "Can you check the status of order ORD-1001?",
            "I want to cancel order ORD-1002.",
        ],
    )

    public_agent_card = AgentCard(
        name="Customer Service Agent",
        description=(
            "A customer service agent for an online store that surfaces "
            "tool call data via the arksim A2A tool capture extension on "
            "Task artifacts."
        ),
        url=os.getenv("A2A_SERVER_URL", "http://localhost:9998/"),
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,
            extensions=[_TOOL_CAPTURE_EXTENSION],
        ),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=CustomerServiceExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9998)
