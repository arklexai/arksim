# SPDX-License-Identifier: Apache-2.0
"""LangGraph integration for arksim.

Install:
    pip install 'arksim[langchain]'
    pip install langgraph langchain-openai
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires arksim's LangChain tracing adapter into a hand-built LangGraph
``StateGraph`` with an LLM node, a ``ToolNode`` for ``lookup_order`` and
``book_table``, and a conditional edge that routes between them. Running
``arksim simulate-evaluate`` produces a simulation.json whose
``tool_calls`` field is populated by the captured invocations.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from tools import book_table, lookup_order
from typing_extensions import TypedDict

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.langchain import ArksimLangChainHandler


class State(TypedDict):
    messages: Annotated[list, add_messages]


class LangGraphAgent(BaseAgent):
    """LangGraph agent with built-in session management via MemorySaver.

    The graph routes between an LLM node and a ``ToolNode`` using
    ``tools_condition``. ``ArksimLangChainHandler`` is passed through
    ``callbacks`` on every ``ainvoke`` call, capturing tool invocations
    into the simulator's ``tool_calls`` field via LangChain's callback bus
    (which LangGraph reuses).
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._thread_id = str(uuid.uuid4())
        self._handler = ArksimLangChainHandler()
        tools = [lookup_order, book_table]
        llm = ChatOpenAI(model="gpt-5.1").bind_tools(tools)

        def chatbot(state: State) -> State:
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(State)
        graph.add_node("chatbot", chatbot)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "chatbot")
        graph.add_conditional_edges(
            "chatbot", tools_condition, {"tools": "tools", END: END}
        )
        graph.add_edge("tools", "chatbot")
        self._app = graph.compile(checkpointer=MemorySaver())

    async def get_chat_id(self) -> str:
        return self._thread_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        config = {
            "configurable": {"thread_id": self._thread_id},
            "callbacks": [self._handler],
        }
        result = await self._app.ainvoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
        )
        return result["messages"][-1].content
