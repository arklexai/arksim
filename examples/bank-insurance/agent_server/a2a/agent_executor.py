# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

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
        await event_queue.enqueue_event(new_agent_text_message(result, context_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")
