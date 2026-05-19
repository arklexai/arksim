# SPDX-License-Identifier: Apache-2.0
"""Rasa integration for arksim.

Drives a running Rasa server through the REST channel and surfaces any
custom actions Rasa executed as ``ToolCall`` instances on the returned
``AgentResponse``.

The REST webhook only returns bot replies (text/buttons/etc.), not the
underlying action invocations. To capture those we hit
``/conversations/{sender_id}/tracker`` after each message and extract
the ``action`` events Rasa logged during that turn, filtering out the
built-in housekeeping actions (``action_listen``, ``action_session_*``,
etc.) so only meaningful custom actions reach ``simulation.json``.

Start Rasa:  rasa run --enable-api --cors "*"
Endpoint:    RASA_ENDPOINT env var or http://localhost:5005/webhooks/rest/webhook
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import (
    AgentResponse,
    ToolCall,
    ToolCallSource,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:5005/webhooks/rest/webhook"

# Built-in Rasa actions that fire on every turn or session boundary.
# Filtering these out keeps ``tool_calls`` focused on meaningful custom
# actions the assistant invoked.
_BUILTIN_ACTIONS: frozenset[str] = frozenset(
    {
        "action_listen",
        "action_session_start",
        "action_restart",
        "action_default_fallback",
        "action_deactivate_loop",
        "action_revert_fallback_events",
        "action_default_ask_affirmation",
        "action_default_ask_rephrase",
        "action_back",
        "action_two_stage_fallback",
        "action_extract_slots",
    }
)


def _server_root(webhook_endpoint: str) -> str:
    """Derive the Rasa server root from the REST webhook URL.

    The tracker endpoint lives at the server root, not under
    ``/webhooks/rest/webhook``.
    """
    marker = "/webhooks/"
    idx = webhook_endpoint.find(marker)
    if idx >= 0:
        return webhook_endpoint[:idx]
    return webhook_endpoint.rstrip("/")


def _extract_tool_calls(
    tracker: dict[str, Any], cutoff_timestamp: float
) -> list[ToolCall]:
    """Build ``ToolCall`` instances from Rasa tracker events after ``cutoff``.

    Pairs each ``action`` event with the slot events that fire before the
    next action; those slot sets are how Rasa custom actions surface
    their results, so we capture them as the tool's arguments/result.
    """
    events = tracker.get("events") or []
    if not isinstance(events, list):
        return []

    # Drop events that predate this turn; the tracker accumulates across
    # the whole conversation but we only want the slice this turn
    # produced.
    fresh = [
        e
        for e in events
        if isinstance(e, dict) and (e.get("timestamp") or 0) > cutoff_timestamp
    ]

    tool_calls: list[ToolCall] = []
    current_action: dict[str, Any] | None = None
    pending_slots: dict[str, Any] = {}

    def _flush() -> None:
        if current_action is None:
            return
        name = current_action.get("name") or ""
        if not name or name in _BUILTIN_ACTIONS:
            return
        result = pending_slots.pop("__result__", None)
        tool_calls.append(
            ToolCall(
                id=str(current_action.get("action_id") or uuid.uuid4()),
                name=str(name),
                arguments=dict(pending_slots),
                result=str(result) if result is not None else None,
                source=ToolCallSource.RASA,
            )
        )

    for event in fresh:
        kind = event.get("event")
        if kind == "action":
            _flush()
            current_action = event
            pending_slots = {}
        elif kind == "slot" and current_action is not None:
            slot_name = event.get("name")
            slot_value = event.get("value")
            if slot_name:
                pending_slots[str(slot_name)] = slot_value
                # Heuristic: a slot whose name contains "status" or
                # "result" is the action's payload; everything else is
                # input the action consumed.
                lowered = str(slot_name).lower()
                if "status" in lowered or "result" in lowered:
                    pending_slots["__result__"] = slot_value
    _flush()
    return tool_calls


class RasaAgent(BaseAgent):
    """Rasa wrapper that captures custom actions via the tracker endpoint."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._endpoint = os.environ.get("RASA_ENDPOINT", _DEFAULT_ENDPOINT)
        self._tracker_url = (
            f"{_server_root(self._endpoint)}/conversations/{self._chat_id}/tracker"
        )
        self._client = httpx.AsyncClient(timeout=60)
        # Track the latest event timestamp we have already consumed so a
        # subsequent turn only surfaces newly emitted actions.
        self._cutoff_timestamp: float = 0.0

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def _fetch_tracker(self) -> dict[str, Any] | None:
        try:
            resp = await self._client.get(self._tracker_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # Tracker fetch is best-effort: warn but keep the
            # conversation alive so the example still produces a
            # transcript.
            logger.warning("Failed to fetch Rasa tracker: %s", exc)
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        try:
            response = await self._client.post(
                self._endpoint,
                json={"sender": self._chat_id, "message": user_query},
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            msg = (
                f"Could not connect to Rasa server at {self._endpoint}. Is it running?"
            )
            raise RuntimeError(msg) from exc

        messages = response.json()
        texts = [m["text"] for m in messages if isinstance(m, dict) and "text" in m]
        content = "\n".join(texts) if texts else ""
        if not texts:
            logger.warning("Rasa returned no text messages for query: %s", user_query)

        tracker = await self._fetch_tracker()
        if tracker is None:
            return AgentResponse(content=content, tool_calls=[])

        tool_calls = _extract_tool_calls(tracker, self._cutoff_timestamp)
        # Advance the cutoff so the next turn only sees newly emitted
        # events. ``latest_event_time`` is the timestamp of the most
        # recent tracker event; Rasa updates it monotonically.
        latest = tracker.get("latest_event_time")
        if isinstance(latest, int | float):
            self._cutoff_timestamp = float(latest)

        return AgentResponse(content=content, tool_calls=tool_calls)

    async def close(self) -> None:
        await self._client.aclose()
