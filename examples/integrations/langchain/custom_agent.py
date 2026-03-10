# SPDX-License-Identifier: Apache-2.0
"""LangChain/LangGraph integration for ArkSim.

Install: pip install langgraph langchain-openai
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class LangChainAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        llm = ChatOpenAI(model="gpt-5.1")

        def agent_node(state: MessagesState) -> dict:
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        self._graph = graph.compile(checkpointer=InMemorySaver())

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        config = {"configurable": {"thread_id": self._chat_id}}
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}, config
        )
        return result["messages"][-1].content
