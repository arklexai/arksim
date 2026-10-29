# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.tracing.span_converter."""

from __future__ import annotations

import base64
import json

from arksim.tracing._attrs import get_attr
from arksim.tracing.span_converter import (
    span_to_tool_call,
    spans_to_tool_calls,
)


class TestSpanToToolCall:
    """Test single span conversion."""

    def test_otel_genai_convention(self) -> None:
        span = {
            "name": "execute_tool search_flights",
            "spanId": "abc123",
            "attributes": [
                {"key": "gen_ai.tool.name", "value": {"stringValue": "search_flights"}},
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {"stringValue": '{"origin": "NYC"}'},
                },
                {
                    "key": "gen_ai.tool.call.result",
                    "value": {"stringValue": '{"flights": []}'},
                },
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": "call-99"}},
            ],
            "status": {"code": 1},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.name == "search_flights"
        assert tc.arguments == {"origin": "NYC"}
        assert tc.result == '{"flights": []}'
        assert tc.id == "call-99"
        assert tc.error is None

    def test_openinference_convention(self) -> None:
        span = {
            "name": "tool_call",
            "spanId": "def456",
            "attributes": [
                {"key": "tool.name", "value": {"stringValue": "get_weather"}},
                {
                    "key": "tool_call.function.arguments",
                    "value": {"stringValue": '{"city": "London"}'},
                },
                {"key": "tool_call.id", "value": {"stringValue": "tc-42"}},
            ],
            "status": {"code": 1},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "London"}
        assert tc.id == "tc-42"

    def test_fallback_to_span_name(self) -> None:
        span = {
            "name": "execute_tool book_hotel",
            "spanId": "ghi789",
            "attributes": [],
            "status": {"code": 1},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.name == "book_hotel"
        assert tc.id == "ghi789"

    def test_plain_span_name_without_prefix_returns_none(self) -> None:
        """Spans without tool attributes or execute_tool prefix are skipped."""
        span = {
            "name": "my_custom_tool",
            "spanId": "jkl012",
            "attributes": [],
            "status": {},
        }
        tc = span_to_tool_call(span)
        assert tc is None

    def test_empty_span_name_returns_none(self) -> None:
        span = {"name": "", "spanId": "", "attributes": [], "status": {}}
        tc = span_to_tool_call(span)
        assert tc is None

    def test_error_status(self) -> None:
        span = {
            "name": "execute_tool failing_tool",
            "spanId": "err1",
            "attributes": [
                {"key": "gen_ai.tool.name", "value": {"stringValue": "failing_tool"}},
            ],
            "status": {"code": 2, "message": "connection timeout"},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.error == "connection timeout"

    def test_error_status_string_code(self) -> None:
        span = {
            "name": "tool",
            "spanId": "err2",
            "attributes": [
                {"key": "tool.name", "value": {"stringValue": "tool"}},
            ],
            "status": {"code": "STATUS_CODE_ERROR", "message": "fail"},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.error == "fail"

    def test_invalid_json_arguments(self) -> None:
        span = {
            "name": "tool",
            "spanId": "bad-json",
            "attributes": [
                {"key": "gen_ai.tool.name", "value": {"stringValue": "tool"}},
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {"stringValue": "not json"},
                },
            ],
            "status": {},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.arguments == {}

    def test_missing_attributes(self) -> None:
        span = {
            "name": "execute_tool minimal",
            "spanId": "min1",
            "status": {},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.name == "minimal"
        assert tc.arguments == {}
        assert tc.result is None

    def test_openinference_tool_parameters_are_not_arguments(self) -> None:
        """tool.parameters is the tool's schema; the call's arguments are input.value."""
        span = {
            "name": "tool",
            "spanId": "param1",
            "attributes": [
                {"key": "tool.name", "value": {"stringValue": "search"}},
                {
                    "key": "tool.parameters",
                    "value": {"stringValue": '{"q": {"type": "string"}}'},
                },
                {"key": "input.value", "value": {"stringValue": '{"q": "test"}'}},
            ],
            "status": {},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.arguments == {"q": "test"}

    def test_declared_non_tool_kind_is_skipped(self) -> None:
        """An LLM span carrying a tool name is a proposed call, not an executed one."""
        span = {
            "name": "chat",
            "spanId": "llm1",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                {"key": "tool.name", "value": {"stringValue": "search"}},
            ],
            "status": {},
        }
        assert span_to_tool_call(span) is None

    def test_openinference_output_value_result(self) -> None:
        """OpenInference output.value is used as tool result."""
        span = {
            "name": "tool",
            "spanId": "oi-result",
            "attributes": [
                {"key": "tool.name", "value": {"stringValue": "search"}},
                {"key": "output.value", "value": {"stringValue": "found 3 results"}},
            ],
            "status": {},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.result == "found 3 results"

    def test_langsmith_bytes_payloads(self) -> None:
        def payload(text: str) -> dict:
            return {"bytesValue": base64.b64encode(text.encode()).decode()}

        span = {
            "name": "get_order",
            "spanId": "ls-1",
            "attributes": [
                {"key": "langsmith.span.kind", "value": {"stringValue": "tool"}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": "get_order"}},
                {"key": "gen_ai.prompt", "value": payload('{"order_id":"ORD-1"}')},
                {"key": "gen_ai.completion", "value": payload('{"output":"ok"}')},
            ],
            "status": {},
        }
        tc = span_to_tool_call(span)
        assert tc is not None
        assert tc.arguments == {"order_id": "ORD-1"}
        assert tc.result == '{"output":"ok"}'


class TestGetAttrFalsyValues:
    """Verify get_attr handles falsy OTLP attribute values correctly."""

    def test_empty_string_preserved(self) -> None:
        attrs = [{"key": "k", "value": {"stringValue": ""}}]
        assert get_attr(attrs, "k") == ""

    def test_zero_int_preserved(self) -> None:
        attrs = [{"key": "k", "value": {"intValue": 0}}]
        assert get_attr(attrs, "k") == "0"

    def test_false_bool_preserved(self) -> None:
        attrs = [{"key": "k", "value": {"boolValue": False}}]
        assert get_attr(attrs, "k") == "False"

    def test_missing_key_returns_none(self) -> None:
        attrs = [{"key": "other", "value": {"stringValue": "val"}}]
        assert get_attr(attrs, "k") is None

    def test_bytes_value_decoded(self) -> None:
        """LangSmith sends JSON payloads as bytes, base64 in OTLP JSON."""
        encoded = base64.b64encode(b'{"order_id":"ORD-1001"}').decode()
        assert get_attr([{"key": "k", "value": {"bytesValue": encoded}}], "k") == (
            '{"order_id":"ORD-1001"}'
        )
        assert get_attr([{"key": "k", "value": {"bytes_value": b"hi"}}], "k") == "hi"
        binary = base64.b64encode(b"\xff\xfe").decode()
        assert get_attr([{"key": "k", "value": {"bytesValue": binary}}], "k") == binary
        assert get_attr([{"key": "k", "value": {"bytesValue": "not!b64"}}], "k") == (
            "not!b64"
        )


class TestSpansToToolCalls:
    """Test batch conversion."""

    def test_multiple_spans(self) -> None:
        spans = [
            {
                "name": "execute_tool a",
                "spanId": "s1",
                "attributes": [
                    {"key": "gen_ai.tool.name", "value": {"stringValue": "a"}},
                ],
                "status": {},
            },
            {
                "name": "execute_tool b",
                "spanId": "s2",
                "attributes": [
                    {"key": "gen_ai.tool.name", "value": {"stringValue": "b"}},
                ],
                "status": {},
            },
        ]
        result = spans_to_tool_calls(spans)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_empty_list(self) -> None:
        assert spans_to_tool_calls([]) == []

    def test_skips_unconvertible_spans(self) -> None:
        spans = [
            {"name": "", "spanId": "", "attributes": [], "status": {}},
            {
                "name": "execute_tool valid",
                "spanId": "v1",
                "attributes": [],
                "status": {},
            },
        ]
        result = spans_to_tool_calls(spans)
        assert len(result) == 1
        assert result[0].name == "valid"

    def test_langchain_single_argument_uses_proposed_call(self) -> None:
        """OpenInference records a single-argument LangChain tool's bare value;
        the argument name comes from the model's call with the same id."""
        prefix = "llm.output_messages.0.message.tool_calls.0.tool_call"

        def s(key: str, value: str) -> dict:
            return {"key": key, "value": {"stringValue": value}}

        message = {"type": "tool", "data": {"content": "{}", "tool_call_id": "c-1"}}
        spans = [
            {
                "name": "ChatOpenAI",
                "spanId": "llm",
                "attributes": [
                    s("openinference.span.kind", "LLM"),
                    s(f"{prefix}.id", "c-1"),
                    s(f"{prefix}.function.name", "get_order"),
                    s(f"{prefix}.function.arguments", '{"order_id": "ORD-1"}'),
                ],
            },
            {
                "name": "get_order",
                "spanId": "tool",
                "attributes": [
                    s("openinference.span.kind", "TOOL"),
                    s("tool.name", "get_order"),
                    s("input.value", "ORD-1"),
                    s("output.value", json.dumps(message)),
                ],
            },
        ]
        [tc] = spans_to_tool_calls(spans)
        assert tc.id == "c-1"
        assert tc.arguments == {"order_id": "ORD-1"}
