# SPDX-License-Identifier: Apache-2.0
"""RAG-powered e-commerce customer service agent using the OpenAI Agents SDK."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agents import Agent as SDKAgent
from agents import Runner, function_tool

from .retriever import FaissRetriever, build_rag

_AGENT_SERVER_DIR = Path(__file__).parent.parent
_knowledge_config = [{"type": "local", "source": "./data"}]

build_rag(str(_AGENT_SERVER_DIR), _knowledge_config)
_retriever = FaissRetriever.load(str(_AGENT_SERVER_DIR))


@function_tool
async def retrieve_context(query: str) -> str:
    """Search the product knowledge base for information relevant to the customer's question.

    Args:
        query: A concise, standalone question derived from the conversation.

    Returns:
        Relevant product excerpts, separated by dividers, or a fallback message.
    """
    results = await _retriever.retrieve(query)
    if not results:
        return "No relevant product information found in the knowledge base."

    parts: list[str] = []
    for r in results:
        header = r["title"] or r["source"] or "Product"
        parts.append(f"[{header}]\n{r['content']}")
    return "\n\n---\n\n".join(parts)


_SYSTEM_INSTRUCTIONS = (
    "You are a helpful e-commerce customer service agent. "
    "Before answering, always call the retrieve_context tool with a concise "
    "standalone version of the customer's question to look up relevant product "
    "information. Keep your final response concise (no more than 40 words) and "
    "do not prefix it with 'Assistant:' or 'AI:'."
)


class Agent:
    """Stateful per-session wrapper around the shared SDK agent.

    Maintains the full conversation history for the session and passes it on
    every ``Runner.run`` call so the model has multi-turn context.
    """

    def __init__(
        self,
        context_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sdk_agent = SDKAgent(
            name="EcommerceAgent",
            instructions=_SYSTEM_INSTRUCTIONS,
            tools=[retrieve_context],
            model="gpt-4o-mini",
        )
        self.context_id = context_id or str(uuid.uuid4())
        self._history: list[dict[str, Any]] = list(history) if history else []

    async def invoke(self, question: str) -> str:
        """Process a user message and return the agent's response.

        Args:
            question: The user's latest message.

        Returns:
            The agent's text response.
        """
        self._history.append({"role": "user", "content": question})
        result = await Runner.run(self.sdk_agent, input=self._history)
        answer: str = result.final_output
        self._history.append({"role": "assistant", "content": answer})
        return answer
