# SPDX-License-Identifier: Apache-2.0
"""Shared OTLP attribute extraction helpers."""

from __future__ import annotations

from typing import Any


def get_attr(attrs: list[dict[str, Any]], key: str) -> str | None:
    """Extract an attribute value by key from an OTLP attribute list.

    OTLP attribute values are typed (stringValue, intValue, boolValue, etc.).
    Handles both JSON-style (``stringValue``) and protobuf-converted
    (``string_value``) field names. We check each with ``is not None`` to
    avoid dropping falsy values like empty strings, ``0``, or ``False``.
    """
    for attr in attrs:
        if attr.get("key") == key:
            value = attr.get("value", {})
            str_val = value.get("stringValue", value.get("string_value"))
            if str_val is not None:
                return str(str_val)
            int_val = value.get("intValue", value.get("int_value"))
            if int_val is not None:
                return str(int_val)
            bool_val = value.get("boolValue", value.get("bool_value"))
            if bool_val is not None:
                return str(bool_val)
    return None


def first_attr(attrs: list[dict[str, Any]], *keys: str) -> str | None:
    """Return the first matching attribute value from the given keys."""
    for key in keys:
        val = get_attr(attrs, key)
        if val is not None:
            return val
    return None
