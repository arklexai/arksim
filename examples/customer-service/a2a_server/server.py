# SPDX-License-Identifier: Apache-2.0
"""A2A server for the customer-service example.

Surfaces tool calls via the arksim A2A tool capture extension
(``A2AToolCaptureExtension``). The server declares the extension in its
AgentCard and emits tool call data in Task artifact metadata.

Per A2A spec section 3.7, task outputs are delivered through Artifacts
on a Task, not through Messages.

Run standalone on port 9998:
    AGENT_BACKEND=adk python -m a2a_server.server

Or build the routes under a path prefix and add them to a parent app.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import uvicorn
from a2a.helpers.proto_helpers import new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    Part,
    TaskState,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, TransportProtocol
from starlette.applications import Starlette
from starlette.routing import Route

from arksim import A2AToolCaptureExtension

from .agent import Agent

DEFAULT_SERVER_URL = "http://localhost:9998/"

# Cap on live per-conversation agents. Each holds an SDK session and, for
# the OpenAI backend, the full transcript; a long-lived server would
# otherwise grow one per context_id forever.
MAX_SESSIONS = 256

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

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        # LRU-ordered: least recently used context is dropped first.
        self._agents: OrderedDict[str, Agent] = OrderedDict()
        self._max_sessions = max_sessions

    def _agent_for(self, context_id: str) -> Agent:
        agent = self._agents.get(context_id)
        if agent is None:
            agent = Agent()
            self._agents[context_id] = agent
            while len(self._agents) > self._max_sessions:
                self._agents.popitem(last=False)
        else:
            self._agents.move_to_end(context_id)
        return agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        context_id = context.context_id
        if not context_id:
            raise ValueError("Context ID is required")

        user_input = context.get_user_input()
        if not user_input or user_input.strip() == "":
            raise ValueError("User input is required")

        answer, tool_calls = await self._agent_for(context_id).invoke(user_input)

        # Only emit tool call metadata when the client requested the
        # extension; this is least-privilege by default (see the Security
        # section in the tool-call-capture docs for deployment guidance).
        # a2a-sdk 1.x dropped activation echo - RequestContext has no
        # add_activated_extension - so the request is the whole negotiation.
        # The artifact still carries the URI in its `extensions` field below,
        # which is what arksim checks before reading the metadata.
        emit_tool_calls = A2AToolCaptureExtension.URI in context.requested_extensions

        if not context.task_id:
            raise ValueError(f"context.task_id is required (context_id={context_id})")
        updater = TaskUpdater(event_queue, context.task_id, context_id)
        if context.current_task is None:
            # The framework requires an actual Task object as the first
            # event for a new task_id - a TaskStatusUpdateEvent alone
            # (which is all updater.submit() emits) is rejected with
            # "Agent should enqueue Task before TaskStatusUpdateEvent
            # event" if no Task has been seen yet.
            await event_queue.enqueue_event(
                new_task(context.task_id, context_id, TaskState.TASK_STATE_SUBMITTED)
            )
        await updater.start_work()

        metadata: dict[str, object] = {}
        extensions: list[str] = []
        if emit_tool_calls and tool_calls:
            metadata[A2AToolCaptureExtension.METADATA_KEY] = tool_calls
            extensions.append(A2AToolCaptureExtension.URI)

        await updater.add_artifact(
            parts=[Part(text=answer)],
            metadata=metadata or None,
            extensions=extensions or None,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


def build_agent_card(server_url: str) -> AgentCard:
    """Build the public AgentCard.

    `server_url` is what clients POST to after resolving the card, so it
    must be the full externally reachable URL including any path prefix.
    """
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
    return AgentCard(
        name="Customer Service Agent",
        description=(
            "A customer service agent for an online store that surfaces "
            "tool call data via the arksim A2A tool capture extension on "
            "Task artifacts."
        ),
        supported_interfaces=[
            AgentInterface(url=server_url, protocol_binding=TransportProtocol.JSONRPC)
        ],
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(
            streaming=False,
            extensions=[_TOOL_CAPTURE_EXTENSION],
        ),
        skills=[skill],
    )


def build_routes(server_url: str, path_prefix: str = "") -> list[Route]:
    """Build the agent-card and JSON-RPC routes, optionally under a prefix.

    Both are exact Routes, so a parent app can append them directly rather
    than mounting a sub-app. That matters: a Starlette Mount at "/x" only
    matches "/x/...", so a bare POST to the RPC url itself would miss it.

    `path_prefix` is the mount path with no trailing slash (e.g.
    "/customer-service-adk-a2a"), or "" when serving at the root.
    """
    agent_card = build_agent_card(server_url)
    request_handler = DefaultRequestHandler(
        agent_executor=CustomerServiceExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    return [
        *create_agent_card_routes(
            agent_card=agent_card,
            card_url=f"{path_prefix}{AGENT_CARD_WELL_KNOWN_PATH}",
        ),
        *create_jsonrpc_routes(
            request_handler=request_handler,
            rpc_url=path_prefix or "/",
        ),
    ]


def build_app(server_url: str | None = None, path_prefix: str = "") -> Starlette:
    """Standalone ASGI app serving the A2A routes."""
    resolved = server_url or os.getenv("A2A_SERVER_URL", DEFAULT_SERVER_URL)
    return Starlette(routes=build_routes(resolved, path_prefix))


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=9998)
