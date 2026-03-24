# SPDX-License-Identifier: Apache-2.0
"""Custom action that looks up order status from a mock database."""

from __future__ import annotations

from typing import Any

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

_ORDERS: dict[str, str] = {
    "ORD-001": "Shipped - expected delivery March 28",
    "ORD-002": "Processing - estimated ship date March 26",
}


class ActionCheckOrderStatus(Action):
    """Look up order status by order ID."""

    def name(self) -> str:
        return "action_check_order_status"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict[str, Any],
    ) -> list[dict[str, Any]]:
        order_id = tracker.get_slot("order_id") or ""
        status = _ORDERS.get(order_id.upper(), "not_found")
        return [SlotSet("order_status", status)]
