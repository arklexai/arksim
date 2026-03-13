# SPDX-License-Identifier: Apache-2.0
"""Example agent with real tool calls for ArkSim tool call evaluation.

Demonstrates how a custom agent can return AgentResponse with ToolCall
data so the evaluator can run the tool_call_behavior_failure metric.

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import json
import uuid

from agents import Agent, Runner, RunResult, function_tool
from agents.items import ToolCallItem, ToolCallOutputItem
from openai.types.responses import ResponseFunctionToolCall

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import AgentResponse, ToolCall

# ── Hardcoded tools (no external dependencies) ──


@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {
        "New York": {"temp_f": 72, "condition": "sunny"},
        "London": {"temp_f": 58, "condition": "cloudy"},
        "Tokyo": {"temp_f": 80, "condition": "humid"},
    }
    return json.dumps(data.get(city, {"error": f"No weather data for {city}"}))


@function_tool
def lookup_order(order_id: str) -> str:
    """Look up order status by ID."""
    orders = {
        "ORD-1234": {
            "status": "shipped",
            "carrier": "FedEx",
            "eta": "2026-03-15",
        },
        "ORD-5678": {
            "status": "processing",
            "carrier": None,
            "eta": None,
        },
    }
    return json.dumps(orders.get(order_id, {"error": f"Order {order_id} not found"}))


@function_tool
def search_products(query: str) -> str:
    """Search product catalog."""
    catalog = [
        {"name": "ThinkPad X1 Carbon", "price": 899, "category": "laptop"},
        {"name": "MacBook Air M4", "price": 1099, "category": "laptop"},
        {"name": "Dell XPS 13", "price": 949, "category": "laptop"},
        {"name": "Sony WH-1000XM5", "price": 348, "category": "headphones"},
    ]
    query_lower = query.lower()
    matches = [
        p
        for p in catalog
        if query_lower in p["name"].lower() or query_lower in p["category"].lower()
    ]
    return json.dumps(matches if matches else [{"error": "No products found"}])


# ── Agent ──


class ToolCallExampleAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions="You are a helpful assistant with access to tools. "
            "Use them when the user's request can be answered by a tool.",
            tools=[get_weather, lookup_order, search_products],
        )
        self._last_result: RunResult | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        self._last_result = await Runner.run(self._agent, input=input_list)

        tool_calls = self._extract_tool_calls(self._last_result)
        return AgentResponse(
            content=self._last_result.final_output,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _extract_tool_calls(result: RunResult) -> list[ToolCall]:
        """Pair ToolCallItems with their ToolCallOutputItems from RunResult."""
        # Index outputs by call_id for fast lookup
        outputs: dict[str, str] = {}
        for item in result.new_items:
            if isinstance(item, ToolCallOutputItem):
                raw = item.raw_item
                call_id = (
                    raw.get("call_id", "")
                    if isinstance(raw, dict)
                    else getattr(raw, "call_id", "")
                )
                output = (
                    raw.get("output", "")
                    if isinstance(raw, dict)
                    else getattr(raw, "output", "")
                )
                if isinstance(output, list):
                    output = json.dumps(output)
                outputs[call_id] = str(output)

        # Build ToolCall list from function call items
        tool_calls: list[ToolCall] = []
        for item in result.new_items:
            if not isinstance(item, ToolCallItem):
                continue
            raw = item.raw_item
            if not isinstance(raw, ResponseFunctionToolCall):
                continue
            call_id = raw.call_id
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=raw.name,
                    arguments=json.loads(raw.arguments) if raw.arguments else {},
                    result=outputs.get(call_id),
                )
            )

        return tool_calls
