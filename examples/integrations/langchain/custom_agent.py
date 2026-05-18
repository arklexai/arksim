# SPDX-License-Identifier: Apache-2.0
"""LangChain / LangGraph integration for arksim.

Install:
    pip install 'arksim[langchain]'
    pip install langgraph langchain-openai
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires arksim's LangChain tracing adapter into a LangGraph ReAct agent
with two mock tools (lookup_order, book_table). Running
``arksim simulate-evaluate`` produces a simulation.json whose
``tool_calls`` field is populated by the captured invocations.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.langchain import ArksimLangChainHandler


class LangChainAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._handler = ArksimLangChainHandler()
        self._graph = create_react_agent(
            ChatOpenAI(model="gpt-5.1"),
            tools=[lookup_order, book_table],
            checkpointer=InMemorySaver(),
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        config = {
            "configurable": {"thread_id": self._chat_id},
            "callbacks": [self._handler],
        }
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}, config
        )
        return result["messages"][-1].content
