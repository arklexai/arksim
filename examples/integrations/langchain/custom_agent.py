# SPDX-License-Identifier: Apache-2.0
"""LangChain/LangGraph integration for ArkSim.

Install: pip install langgraph langchain-openai
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class LangChainAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        llm = ChatOpenAI(model="gpt-5.1")

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def agent_node(state: State) -> dict:
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(State)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)
        self._graph = graph.compile()
        self._history: list[BaseMessage] = []

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        self._history.append(HumanMessage(content=user_query))
        result = await self._graph.ainvoke({"messages": list(self._history)})
        ai_message = result["messages"][-1]
        self._history.append(AIMessage(content=ai_message.content))
        return ai_message.content
