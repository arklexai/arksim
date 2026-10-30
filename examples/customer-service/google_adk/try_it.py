# SPDX-License-Identifier: Apache-2.0
"""Scratch script: ask the Google ADK agent one question and see the answer
plus which tools it called.

Usage:
    export GOOGLE_API_KEY="..."
    cd arksim/examples/customer-service
    python google_adk/try_it.py "What is the status of order ORD-1001?"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_agent import GoogleADKAgent  # noqa: E402

from arksim import AgentConfig  # noqa: E402
from arksim.config.core.agent import CustomConfig  # noqa: E402


async def main() -> None:
    question = (
        sys.argv[1] if len(sys.argv) > 1 else "What is the status of order ORD-1001?"
    )
    agent = GoogleADKAgent(
        AgentConfig(
            agent_name="customer-service-adk",
            agent_type="custom",
            custom_config=CustomConfig(agent_class=GoogleADKAgent),
        )
    )
    response = await agent.execute(question)
    print("Question:", question)
    print("Answer:", response.content)
    print("Tool calls:", response.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
