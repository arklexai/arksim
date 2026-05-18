# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Part, Role

from ..core.agent import Agent  # noqa: TID252


class BankInsuranceAgentExecutor(AgentExecutor):
    """Bank-Insurance Agent Executor."""

    def __init__(self) -> None:
        self.agent_cache: dict[str, Agent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        context_id = context.context_id
        if not context_id:
            raise ValueError("Context ID is required")
        if context_id not in self.agent_cache:
            self.agent_cache[context_id] = Agent(context_id)
        agent = self.agent_cache[context_id]

        user_input = context.get_user_input()
        if not user_input or user_input.strip() == "":
            raise ValueError("User input is required")

        result = await agent.invoke(user_input)
        # NOTE: this emits a raw Message (no Task lifecycle). For production
        # use with arksim's tool capture, switch to TaskUpdater; see the
        # customer-service example for the full Task lifecycle pattern.
        msg = Message(
            role=Role.ROLE_AGENT,
            parts=[Part(text=result)],
            message_id=str(uuid.uuid4()),
            context_id=context_id,
        )
        await event_queue.enqueue_event(msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")
