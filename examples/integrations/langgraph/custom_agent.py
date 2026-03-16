# SPDX-License-Identifier: Apache-2.0
"""LangGraph integration for ArkSim.

Install: pip install langgraph langchain-openai
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class State(TypedDict):
    messages: Annotated[list, add_messages]


class LangGraphAgent(BaseAgent):
    """LangGraph agent with built-in session management via MemorySaver.

    LangGraph's checkpointer handles conversation history internally
    using the thread_id, similar to Google ADK's InMemoryRunner.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._thread_id = str(uuid.uuid4())
        llm = ChatOpenAI(model="gpt-4o")

        def chatbot(state: State) -> State:
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(State)
        graph.add_node("chatbot", chatbot)
        graph.add_edge(START, "chatbot")
        self._app = graph.compile(checkpointer=MemorySaver())

    async def get_chat_id(self) -> str:
        return self._thread_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        config = {"configurable": {"thread_id": self._thread_id}}
        result = await self._app.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config=config,
        )
        return result["messages"][-1].content
