# SPDX-License-Identifier: Apache-2.0
"""Custom action that forwards unrecognized messages to an LLM."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

_client = AsyncOpenAI()

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers user questions clearly and accurately."
)


class ActionLLMResponse(Action):
    """Send the conversation to an LLM and return its response."""

    def name(self) -> str:
        return "action_llm_response"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        for event in tracker.events_after_latest_restart():
            if event.get("event") == "user":
                text = event.get("text", "")
                if text:
                    messages.append({"role": "user", "content": text})
            elif event.get("event") == "bot":
                text = event.get("text", "")
                if text:
                    messages.append({"role": "assistant", "content": text})

        response = await _client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        reply = response.choices[0].message.content or ""
        dispatcher.utter_message(text=reply)
        return []
