# SPDX-License-Identifier: Apache-2.0
"""Tests for parse_tool_arguments normalization."""

from __future__ import annotations

from arksim.tracing.integrations._args import parse_tool_arguments


def test_dict_passthrough() -> None:
    raw = {"city": "NYC", "units": "F"}
    assert parse_tool_arguments(raw) == {"city": "NYC", "units": "F"}


def test_none_yields_empty() -> None:
    assert parse_tool_arguments(None) == {}


def test_empty_string_yields_empty() -> None:
    assert parse_tool_arguments("") == {}


def test_json_dict_string_parses() -> None:
    assert parse_tool_arguments('{"city": "NYC"}') == {"city": "NYC"}


def test_json_scalar_wrapped_in_value() -> None:
    assert parse_tool_arguments("42") == {"_value": 42}


def test_json_list_wrapped_in_value() -> None:
    assert parse_tool_arguments("[1, 2, 3]") == {"_value": [1, 2, 3]}


def test_json_null_wrapped_in_value() -> None:
    assert parse_tool_arguments("null") == {"_value": None}


def test_json_string_literal_wrapped_in_value() -> None:
    assert parse_tool_arguments('"hello"') == {"_value": "hello"}


def test_non_json_string_wrapped_in_value() -> None:
    assert parse_tool_arguments("not-json") == {"_value": "not-json"}


def test_malformed_json_falls_back_to_raw_value() -> None:
    assert parse_tool_arguments("{incomplete") == {"_value": "{incomplete"}
