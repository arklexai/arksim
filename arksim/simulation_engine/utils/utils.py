# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


def flip_hist(hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flip user/assistant roles in conversation history, preserving all fields."""
    new_hist = []
    for turn in hist:
        if "role" not in turn:
            new_hist.append(turn)
        elif turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            new_hist.append({**turn, "role": "assistant"})
        else:
            new_hist.append({**turn, "role": "user"})
    return new_hist
