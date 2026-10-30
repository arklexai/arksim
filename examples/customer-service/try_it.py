# SPDX-License-Identifier: Apache-2.0
"""Scratch script for manually poking at the agent while developing.

Not part of the official example - just a quick way to ask one question
and see which tools got called, without running the full arksim pipeline.

Usage:
    export OPENAI_API_KEY="..."
    python try_it.py "What is the return policy for laptops?"
"""

from __future__ import annotations

import asyncio
import sys

from openai_agents.custom_agent import ToolCallExampleAgent

from arksim import AgentConfig
from arksim.config.core.agent import CustomConfig


async def main() -> None:
    question = (
        sys.argv[1] if len(sys.argv) > 1 else "What is the return policy for laptops?"
    )
    agent = ToolCallExampleAgent(
        AgentConfig(
            agent_name="customer-service",
            agent_type="custom",
            custom_config=CustomConfig(agent_class=ToolCallExampleAgent),
        )
    )
    response = await agent.execute(question)
    print("Question:", question)
    print("Answer:", response.content)
    print("Tool calls:", response.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
