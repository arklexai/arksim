# SPDX-License-Identifier: Apache-2.0
"""Tests for A2A tool call extraction from Task artifacts.

Tests go through ``A2AAgent.execute()`` so they exercise the real
public entrypoint (event loop, streaming vs non-streaming handling,
and artifact accumulation) rather than private helpers.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)

from arksim.simulation_engine.agent.clients.a2a import A2AAgent
from arksim.simulation_engine.tool_types import (
    A2AToolCaptureExtension,
    AgentResponse,
    ToolCallSource,
)

# ---------------------------------------------------------------------------
# Builder helpers: _text_part, _make_artifact, _make_task, _make_message,
# _task_response, _message_response, _artifact_update_response,
# _status_update_response, _mock_agent.
# ---------------------------------------------------------------------------


def _text_part(text: str) -> Part:
    return Part(text=text)


def _make_artifact(
    text: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    extensions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    name: str = "response",
) -> Artifact:
    """Build an Artifact, optionally with text parts and tool capture metadata."""
    parts = [_text_part(text)] if text is not None else []
    if metadata is None:
        metadata = {}
    if tool_calls is not None:
        metadata[A2AToolCaptureExtension.METADATA_KEY] = tool_calls
    if extensions is None and tool_calls is not None:
        extensions = [A2AToolCaptureExtension.URI]
    artifact = Artifact(
        artifact_id=str(uuid.uuid4()),
        parts=parts,
    )
    if extensions:
        artifact.extensions.extend(extensions)
    if metadata:
        artifact.metadata.update(metadata)
    return artifact


def _make_task(artifacts: list[Artifact], status_text: str | None = None) -> Task:
    """Build a Task with the given artifacts and optional status message text."""
    status_message = None
    if status_text is not None:
        status_message = Message(
            role=Role.ROLE_AGENT,
            parts=[_text_part(status_text)],
            message_id=str(uuid.uuid4()),
        )
    task = Task(
        id=str(uuid.uuid4()),
        context_id="test-ctx",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )
    if status_message is not None:
        task.status.message.CopyFrom(status_message)
    task.artifacts.extend(artifacts)
    return task


def _make_message(text: str) -> Message:
    return Message(
        role=Role.ROLE_AGENT,
        parts=[_text_part(text)],
        message_id=str(uuid.uuid4()),
        context_id="test-ctx",
    )


def _task_response(task: Task) -> StreamResponse:
    """Wrap a Task in a StreamResponse."""
    resp = StreamResponse()
    resp.task.CopyFrom(task)
    return resp


def _message_response(msg: Message) -> StreamResponse:
    """Wrap a Message in a StreamResponse."""
    resp = StreamResponse()
    resp.message.CopyFrom(msg)
    return resp


def _artifact_update_response(
    task_id: str, context_id: str, artifact: Artifact
) -> StreamResponse:
    """Wrap a TaskArtifactUpdateEvent in a StreamResponse."""
    resp = StreamResponse()
    resp.artifact_update.task_id = task_id
    resp.artifact_update.context_id = context_id
    resp.artifact_update.artifact.CopyFrom(artifact)
    return resp


def _status_update_response(
    task_id: str, context_id: str, status: TaskStatus
) -> StreamResponse:
    """Wrap a TaskStatusUpdateEvent in a StreamResponse."""
    resp = StreamResponse()
    resp.status_update.task_id = task_id
    resp.status_update.context_id = context_id
    resp.status_update.status.CopyFrom(status)
    return resp


def _mock_agent(
    events: list[StreamResponse],
    card_declares_extension: bool = True,
) -> A2AAgent:
    """Build an A2AAgent with a mock client that yields the given events."""
    config = MagicMock()
    config.agent_type = "a2a"
    config.api_config = MagicMock()
    config.api_config.endpoint = "http://localhost:9998"
    config.api_config.get_headers.return_value = {}

    agent = A2AAgent(config)

    async def mock_send_message(_payload: object) -> AsyncIterator[StreamResponse]:
        for ev in events:
            yield ev

    # Bypass _ensure_initialized: no live server needed in unit tests.
    mock_client = MagicMock()
    mock_client.send_message = mock_send_message
    agent._client = mock_client
    agent._initialized = True
    agent._card_declares_tool_capture = card_declares_extension
    return agent


class TestTaskResponse:
    """Tests for task-based responses (the spec-preferred path)."""

    @pytest.mark.asyncio()
    async def test_artifact_with_text_and_tool_calls(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="The weather in Tokyo is 72F and sunny.",
                    tool_calls=[
                        {
                            "id": "call_001",
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                            "result": "Weather in Tokyo: 72F, sunny",
                        }
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("What is the weather?")

        assert isinstance(result, AgentResponse)
        assert result.content == "The weather in Tokyo is 72F and sunny."
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_001"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}
        assert tc.result == "Weather in Tokyo: 72F, sunny"
        assert tc.source == ToolCallSource.A2A_PROTOCOL

    @pytest.mark.asyncio()
    async def test_text_only_artifact(self) -> None:
        task = _make_task([_make_artifact(text="Hello.")])
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("hi")
        assert result.content == "Hello."
        assert result.tool_calls == []

    @pytest.mark.asyncio()
    async def test_artifact_without_extension_uri_ignored(self) -> None:
        """Tool calls in metadata are ignored if the extension URI is not on the artifact."""
        artifact = _make_artifact(
            text="Some text.",
            metadata={
                A2AToolCaptureExtension.METADATA_KEY: [
                    {"id": "x", "name": "should_be_ignored", "arguments": {}}
                ]
            },
            extensions=[],
        )
        task = _make_task([artifact])
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.content == "Some text."
        assert result.tool_calls == []

    @pytest.mark.asyncio()
    async def test_multiple_artifacts_collect_all(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="First.",
                    tool_calls=[
                        {"id": "c1", "name": "tool_a", "arguments": {}, "result": "a"}
                    ],
                ),
                _make_artifact(
                    text="Second.",
                    tool_calls=[
                        {"id": "c2", "name": "tool_b", "arguments": {}, "result": "b"}
                    ],
                ),
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert "First." in result.content
        assert "Second." in result.content
        assert {tc.name for tc in result.tool_calls} == {"tool_a", "tool_b"}

    @pytest.mark.asyncio()
    async def test_falls_back_to_status_message_when_no_artifact_text(self) -> None:
        task = _make_task(artifacts=[], status_text="Done.")
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.content == "Done."
        assert result.tool_calls == []

    @pytest.mark.asyncio()
    async def test_empty_task(self) -> None:
        task = Task(
            id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.content == ""
        assert result.tool_calls == []

    @pytest.mark.asyncio()
    async def test_artifact_with_none_parts_does_not_crash(self) -> None:
        """An artifact with empty parts is handled gracefully."""
        artifact = Artifact(artifact_id="a1")
        artifact.extensions.append(A2AToolCaptureExtension.URI)
        artifact.metadata.update(
            {
                A2AToolCaptureExtension.METADATA_KEY: [
                    {"id": "c1", "name": "t", "arguments": {}, "result": "r"}
                ]
            }
        )
        task = _make_task([artifact])
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.tool_calls[0].name == "t"


class TestMalformedToolCallData:
    """Adversarial tests: malformed tool call data in artifact metadata."""

    @pytest.mark.asyncio()
    async def test_missing_name_skipped(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="Partial.",
                    tool_calls=[
                        {"id": "bad", "arguments": {"x": 1}},
                        {"id": "good", "name": "valid_tool", "arguments": {}},
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "valid_tool"

    @pytest.mark.asyncio()
    async def test_entry_not_dict_skipped(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="Text.",
                    tool_calls=[42, {"id": "ok", "name": "valid", "arguments": {}}],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "valid"

    @pytest.mark.asyncio()
    async def test_arguments_not_dict_skipped(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="Text.",
                    tool_calls=[
                        {"id": "bad", "name": "bad_args", "arguments": "str"},
                        {"id": "ok", "name": "good_args", "arguments": {"k": "v"}},
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "good_args"

    @pytest.mark.asyncio()
    async def test_metadata_tool_calls_not_a_list(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="Text.",
                    metadata={A2AToolCaptureExtension.METADATA_KEY: "not_a_list"},
                    extensions=[A2AToolCaptureExtension.URI],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.content == "Text."
        assert result.tool_calls == []

    @pytest.mark.asyncio()
    async def test_non_string_result_coerced_to_json(self) -> None:
        """Dict/list results are JSON-serialized into the str-typed ToolCall field."""
        task = _make_task(
            [
                _make_artifact(
                    text="Search.",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "search",
                            "arguments": {"q": "laptop"},
                            "result": [{"name": "item1"}, {"name": "item2"}],
                        }
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result == '[{"name": "item1"}, {"name": "item2"}]'

    @pytest.mark.asyncio()
    async def test_non_string_error_coerced(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="Fail.",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "t",
                            "arguments": {},
                            "error": {"code": 500, "msg": "Internal"},
                        }
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        # Protobuf Struct may normalize ints to floats; 500 == 500.0 in Python.
        assert json.loads(result.tool_calls[0].error) == {
            "code": 500,
            "msg": "Internal",
        }

    @pytest.mark.asyncio()
    async def test_error_field_preserved(self) -> None:
        task = _make_task(
            [
                _make_artifact(
                    text="Failure.",
                    tool_calls=[
                        {
                            "id": "err",
                            "name": "failing_tool",
                            "arguments": {},
                            "error": "ConnectionTimeout",
                        }
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.tool_calls[0].error == "ConnectionTimeout"


class TestMessageResponse:
    """Tests for text-only Message responses (fallback path)."""

    @pytest.mark.asyncio()
    async def test_message_text_extracted(self) -> None:
        msg = _make_message("Just text.")
        agent = _mock_agent([_message_response(msg)])
        result = await agent.execute("hi")
        assert result.content == "Just text."
        assert result.tool_calls == []


class TestStreamingArtifactEvents:
    """Tests for streaming: TaskArtifactUpdateEvent before final Task."""

    @pytest.mark.asyncio()
    async def test_streaming_artifact_update_captures_tool_calls(self) -> None:
        """Tool calls in a TaskArtifactUpdateEvent are captured even when
        the final Task snapshot omits them (e.g. append=True streaming)."""
        artifact = _make_artifact(
            text="Streamed.",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "streamed_tool",
                    "arguments": {"x": 1},
                    "result": "ok",
                }
            ],
        )
        update_resp = _artifact_update_response("t1", "c1", artifact)
        # Final Task snapshot has the same artifact (typical server behavior).
        final_task = _make_task([artifact])

        agent = _mock_agent([update_resp, _task_response(final_task)])
        result = await agent.execute("q")
        # Final snapshot replaces accumulated state; no duplication.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "streamed_tool"

    @pytest.mark.asyncio()
    async def test_streaming_update_only_captures_tool_calls(self) -> None:
        """When an update event arrives without a corresponding Task snapshot,
        the update's tool calls are captured rather than lost."""
        artifact = _make_artifact(
            text="Partial.",
            tool_calls=[
                {"id": "c1", "name": "streamed_tool", "arguments": {}, "result": "r"}
            ],
        )
        update_resp = _artifact_update_response("t1", "c1", artifact)
        agent = _mock_agent([update_resp])
        result = await agent.execute("q")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "streamed_tool"


class TestMissingExtensionDeclaration:
    """Tests for the AgentCard extension declaration guard."""

    @pytest.mark.asyncio()
    async def test_warns_when_card_did_not_declare_extension(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the server sends tool calls but the card didn't declare the
        extension, we still extract but log a warning."""
        import logging

        task = _make_task(
            [
                _make_artifact(
                    text="Sneaky.",
                    tool_calls=[{"id": "c1", "name": "sneaky_tool", "arguments": {}}],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)], card_declares_extension=False)
        with caplog.at_level(logging.WARNING):
            result = await agent.execute("q")

        assert len(result.tool_calls) == 1
        assert any(
            "AgentCard does not declare" in rec.message for rec in caplog.records
        )


class TestTypeGuards:
    """Tests that non-string types in tool call fields are rejected clearly."""

    @pytest.mark.asyncio()
    async def test_truthy_non_string_name_rejected(self) -> None:
        """A truthy integer name (e.g. 42) must be skipped, not coerced."""
        task = _make_task(
            [
                _make_artifact(
                    text="Text.",
                    tool_calls=[
                        {"id": "bad", "name": 42, "arguments": {}},
                        {"id": "good", "name": "valid", "arguments": {}},
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "valid"

    @pytest.mark.asyncio()
    async def test_arguments_none_skipped(self) -> None:
        """arguments=None (explicit null) is skipped, not treated as empty dict."""
        task = _make_task(
            [
                _make_artifact(
                    text="Text.",
                    tool_calls=[
                        {"id": "bad", "name": "x", "arguments": None},
                        {"id": "good", "name": "y", "arguments": {}},
                    ],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "y"


class TestStreamingSequences:
    """Tests for streaming event ordering and accumulation."""

    @pytest.mark.asyncio()
    async def test_status_update_does_not_wipe_accumulated_state(self) -> None:
        """A TaskStatusUpdateEvent after an artifact update must not clear accumulated tool calls."""
        artifact = _make_artifact(
            text="Streamed.",
            tool_calls=[
                {"id": "c1", "name": "streamed_tool", "arguments": {}, "result": "r"}
            ],
        )
        update_resp = _artifact_update_response("t1", "c1", artifact)
        status_resp = _status_update_response(
            "t1",
            "c1",
            TaskStatus(state=TaskState.TASK_STATE_WORKING),
        )

        agent = _mock_agent([update_resp, status_resp])
        result = await agent.execute("q")

        # The streamed tool call survives the status update.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "streamed_tool"

    @pytest.mark.asyncio()
    async def test_final_task_snapshot_replaces_partial_updates(self) -> None:
        """A final Task snapshot is authoritative: it replaces earlier update state."""
        old_artifact = _make_artifact(
            text="Stale.",
            tool_calls=[{"id": "c_old", "name": "old_tool", "arguments": {}}],
        )
        final_artifact = _make_artifact(
            text="Final.",
            tool_calls=[
                {"id": "c_new", "name": "new_tool", "arguments": {}, "result": "r"}
            ],
        )
        update_resp = _artifact_update_response("t1", "c1", old_artifact)
        final_task = _make_task([final_artifact])
        agent = _mock_agent([update_resp, _task_response(final_task)])
        result = await agent.execute("q")

        # The Task snapshot is authoritative; only its artifacts are kept.
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "new_tool"


class TestInitLifecycle:
    """Tests for _ensure_initialized and close() lifecycle safety."""

    @pytest.mark.asyncio()
    async def test_headers_none_does_not_crash(self) -> None:
        """A2AConfig.get_headers() returning None is handled."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.agent_type = "a2a"
        config.api_config = MagicMock()
        config.api_config.endpoint = "http://localhost:9998"
        config.api_config.get_headers.return_value = None

        agent = A2AAgent(config)
        # Just verify construction doesn't crash - actual init requires a real server.
        assert agent._httpx_client is None
        assert not agent._initialized

    @pytest.mark.asyncio()
    async def test_close_resets_card_declaration_flag(self) -> None:
        """close() must clear cached card flags so re-init starts fresh."""
        agent = _mock_agent([], card_declares_extension=True)
        agent._warned_about_missing_declaration = True
        await agent.close()

        assert agent._card_declares_tool_capture is False
        assert agent._warned_about_missing_declaration is False
        assert not agent._initialized


class TestExtraArtifactsInFinalSnapshot:
    """Task snapshots can carry MORE artifacts than streamed updates."""

    @pytest.mark.asyncio()
    async def test_final_snapshot_adds_new_artifacts(self) -> None:
        """A Task snapshot with additional artifacts beyond streamed updates keeps them all."""
        stream_artifact = _make_artifact(
            text="Streamed.",
            tool_calls=[
                {"id": "s1", "name": "stream_tool", "arguments": {}, "result": "s"}
            ],
        )
        extra_artifact = _make_artifact(
            text="Extra.",
            tool_calls=[
                {"id": "e1", "name": "extra_tool", "arguments": {}, "result": "e"}
            ],
        )
        update_resp = _artifact_update_response("t1", "c1", stream_artifact)
        # Final snapshot has BOTH artifacts: the earlier streamed one AND
        # a new one that wasn't streamed.
        final_task = _make_task([stream_artifact, extra_artifact])
        agent = _mock_agent([update_resp, _task_response(final_task)])
        result = await agent.execute("q")

        names = {tc.name for tc in result.tool_calls}
        assert names == {"stream_tool", "extra_tool"}


class TestMetadataEdgeCases:
    """Edge cases on artifact metadata shape."""

    @pytest.mark.asyncio()
    async def test_explicit_null_metadata_value(self) -> None:
        """Metadata with `tool_calls: None` (explicit null) is safely ignored."""
        task = _make_task(
            [
                _make_artifact(
                    text="Text.",
                    metadata={A2AToolCaptureExtension.METADATA_KEY: None},
                    extensions=[A2AToolCaptureExtension.URI],
                )
            ]
        )
        agent = _mock_agent([_task_response(task)])
        result = await agent.execute("q")
        assert result.content == "Text."
        assert result.tool_calls == []


class TestSnapshotDeserialization:
    """Tests for ToolCall deserialization from saved snapshots."""

    def test_unknown_source_string_does_not_crash_loading(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Old snapshots with an unknown `source` value load with a warning,
        not a ValidationError that aborts the whole replay."""
        import logging

        from arksim.simulation_engine.simulator import _tool_call_from_dict

        tc_dict = {
            "id": "c1",
            "name": "some_tool",
            "arguments": {},
            "result": "ok",
            "source": "future_capture_source_v2",
        }
        with caplog.at_level(logging.WARNING):
            tc = _tool_call_from_dict(tc_dict)

        assert tc.name == "some_tool"
        assert tc.source is None
        assert any("unknown tool_call source" in rec.message for rec in caplog.records)

    def test_missing_required_field_returns_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A snapshot dict missing ``name`` returns None + warning, doesn't crash."""
        import logging

        from arksim.simulation_engine.simulator import _tool_call_from_dict

        tc_dict = {"id": "bad", "arguments": {}}  # missing name
        with caplog.at_level(logging.WARNING):
            tc = _tool_call_from_dict(tc_dict)

        assert tc is None
        assert any(
            "Skipping malformed tool call" in rec.message for rec in caplog.records
        )

    def test_extra_field_ignored(self) -> None:
        """Unknown fields in the snapshot dict are silently ignored."""
        from arksim.simulation_engine.simulator import _tool_call_from_dict

        tc_dict = {
            "id": "c1",
            "name": "tool",
            "arguments": {},
            "future_field_we_dont_know_about": "whatever",
        }
        tc = _tool_call_from_dict(tc_dict)
        assert tc is not None
        assert tc.name == "tool"


class TestEmptyTaskWithMessageFallback:
    """If a Task arrives with no artifact text, fall back to Message text."""

    @pytest.mark.asyncio()
    async def test_message_text_used_when_task_artifacts_empty(self) -> None:
        """A text-only Message followed by an empty Task yields the Message text."""
        msg = _make_message("Hello from message.")
        empty_task = Task(
            id="t1",
            context_id="c1",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        )
        agent = _mock_agent([_message_response(msg), _task_response(empty_task)])
        result = await agent.execute("hi")
        assert result.content == "Hello from message."


class TestCloseExecuteRace:
    """Race: _client nulled between init fast-path and send_message()."""

    @pytest.mark.asyncio()
    async def test_none_client_raises_clear_runtime_error(self) -> None:
        """If self._client is None when execute() reaches send_message(), we
        raise a clear RuntimeError rather than an opaque AttributeError."""
        agent = _mock_agent([])
        # Simulate the race: init fast-path thinks we're initialized, but
        # close() ran and nulled _client.
        agent._initialized = True
        agent._client = None

        import pytest as _pytest

        with _pytest.raises(RuntimeError, match="closed"):
            await agent.execute("q")


class TestToolCallFromDictCallerSemantics:
    """Behavioral contract: all-malformed tool_calls yield None, not []."""

    def test_all_none_collapses_to_none(self) -> None:
        """Invariant: if every tool call fails to parse, the Message carries
        tool_calls=None (not an empty list) so downstream logic can tell
        "no tool calls recorded" from "tool calls were all malformed"."""
        from arksim.simulation_engine.simulator import _tool_call_from_dict

        # Two malformed dicts (both missing required 'name')
        parsed = [_tool_call_from_dict({"id": "a"}), _tool_call_from_dict({"id": "b"})]
        assert parsed == [None, None]

        # Mirror the caller logic from _to_conversation_output:
        tool_calls = [tc for tc in parsed if tc is not None] or None
        assert tool_calls is None
