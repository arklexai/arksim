# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
)
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
)
from google.protobuf.json_format import MessageToDict

from arksim.config import A2AConfig, AgentConfig, AgentType
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import (
    A2AToolCaptureExtension,
    AgentResponse,
    ToolCall,
    ToolCallSource,
)

logger = logging.getLogger(__name__)


class A2AAgent(BaseAgent):
    """A2A (Agent-to-Agent) agent implementation.

    Surfaces tool calls made by the remote agent via the arksim A2A tool
    capture extension (see ``A2AToolCaptureExtension``): the remote agent
    declares the extension URI in its AgentCard and emits ``Task.artifacts``
    whose ``metadata`` carries the tool call payload keyed by the extension
    URI. arksim negotiates the extension by including the URI in the
    ``Message.extensions`` field on every ``SendMessageRequest``.

    The AgentCard is fetched once per instance in ``_ensure_initialized``;
    long-lived instances will not see card changes until re-instantiated.
    For the simulator's per-simulation agent lifecycle this is intentional.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        if agent_config.agent_type != AgentType.A2A.value:
            raise ValueError("Agent config must be of type a2a")
        self.config: A2AConfig = agent_config.api_config
        self.chat_id = str(uuid.uuid4())

        # Initialized lazily on first execute() call.
        self._client = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        # Set once per init; reset by close() so a re-init against a new
        # endpoint doesn't carry the old card's declaration flag.
        self._card_declares_tool_capture = False
        self._warned_about_missing_declaration = False

    async def get_chat_id(self) -> str:
        """Get the chat ID."""
        return self.chat_id

    async def _ensure_initialized(self) -> None:
        """Lazily initialize the A2A client on first use.

        Serialized via ``asyncio.Lock`` so concurrent ``execute()`` calls
        on the same agent instance don't race and leak httpx clients.
        On any error during init, any partially constructed ``httpx``
        client is closed before re-raising so callers don't leak sockets.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            try:
                # Copy headers to avoid mutating the caller's config.
                headers = dict(self.config.get_headers() or {})

                self._httpx_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0),
                    headers=headers or None,
                )

                resolver = A2ACardResolver(
                    httpx_client=self._httpx_client,
                    base_url=self.config.endpoint,
                )
                agent_card = await resolver.get_agent_card()

                # Check that the remote agent card declares the tool capture
                # extension. If not, tool call data in artifact metadata is
                # non-contractual; we still read it but log a warning.
                # HasField works because capabilities is proto3 optional in
                # the a2a-sdk proto schema; a default empty message is
                # always truthy in protobuf, so truthiness is unreliable.
                declared_uris: set[str] = set()
                if agent_card.HasField("capabilities"):
                    declared_uris = {
                        ext.uri for ext in agent_card.capabilities.extensions
                    }
                self._card_declares_tool_capture = (
                    A2AToolCaptureExtension.URI in declared_uris
                )

                streaming = (
                    agent_card.capabilities.streaming
                    if agent_card.HasField("capabilities")
                    else False
                )

                config = ClientConfig(
                    httpx_client=self._httpx_client,
                    streaming=streaming,
                )

                self._client = ClientFactory(config).create(agent_card)
                self._initialized = True

            except Exception:
                logger.error("Could not initialize A2A client", exc_info=True)
                # Don't leak the httpx client on failure.
                if self._httpx_client is not None:
                    with contextlib.suppress(Exception):
                        await self._httpx_client.aclose()
                    self._httpx_client = None
                self._client = None
                raise

    @staticmethod
    def _coerce_to_string(value: Any) -> str | None:  # noqa: ANN401
        """Coerce a tool result/error value to a string for ToolCall storage.

        Non-string values (dicts, lists, numbers) are JSON-serialized so
        they round-trip cleanly into ``ToolCall.result`` / ``ToolCall.error``
        (both typed ``str | None``). Values that cannot be JSON-serialized
        fall back to ``str()``; a debug log is emitted so leaked repr
        strings are visible at debug level.
        """
        if value is None or isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            logger.debug(
                "Tool call value %r is not JSON-serializable; falling back to str()",
                type(value).__name__,
            )
            return str(value)

    @staticmethod
    def _extract_tool_calls_from_artifact(artifact: Artifact) -> list[ToolCall]:
        """Extract tool calls from an artifact's metadata.

        Per the arksim A2A tool capture extension, tool calls are carried
        in ``Artifact.metadata[A2AToolCaptureExtension.METADATA_KEY]`` as a
        list of dicts matching the ToolCall schema. The artifact lists the
        extension URI in its ``extensions`` field to flag the convention.
        """
        if A2AToolCaptureExtension.URI not in artifact.extensions:
            return []
        # Protobuf Struct is always truthy even when empty (same proto3
        # issue as HasField above); use len() to check emptiness.
        if not len(artifact.metadata):
            return []
        # Protobuf Struct does not support .get(); convert to a native dict.
        metadata = MessageToDict(artifact.metadata)
        raw_calls = metadata.get(A2AToolCaptureExtension.METADATA_KEY)
        if not isinstance(raw_calls, list):
            if raw_calls is not None:
                logger.debug(
                    "Ignoring artifact metadata with non-list tool_calls: %s",
                    type(raw_calls).__name__,
                )
            return []

        tool_calls: list[ToolCall] = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                logger.warning(
                    "Skipping malformed tool call in artifact metadata: "
                    "entry is not a dict"
                )
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                logger.warning(
                    "Skipping malformed tool call in artifact metadata: "
                    "'name' must be a non-empty string"
                )
                continue
            arguments = raw.get("arguments", {})
            if not isinstance(arguments, dict):
                logger.warning(
                    "Skipping malformed tool call in artifact metadata: "
                    "'arguments' is not a dict"
                )
                continue
            tool_calls.append(
                ToolCall(
                    id=raw.get("id", ""),
                    name=name,
                    arguments=arguments,
                    result=A2AAgent._coerce_to_string(raw.get("result")),
                    error=A2AAgent._coerce_to_string(raw.get("error")),
                    source=ToolCallSource.A2A_PROTOCOL,
                )
            )
        return tool_calls

    @staticmethod
    def _text_from_parts(parts: list[Part] | None) -> str:
        """Concatenate text from a list of A2A Parts (None-safe)."""
        if not parts:
            return ""
        return "\n".join(part.text for part in parts if part.HasField("text"))

    def _merge_artifact(
        self,
        artifact: Artifact,
        text_parts: list[str],
        tool_calls: list[ToolCall],
    ) -> None:
        """Extract and append an artifact's text and tool calls in place.

        Warns if tool call data is present but the agent card didn't
        declare the tool capture extension.
        """
        # list() copies the repeated field to satisfy the list[Part] type.
        text = self._text_from_parts(list(artifact.parts))
        if text:
            text_parts.append(text)
        calls = self._extract_tool_calls_from_artifact(artifact)
        if calls:
            if (
                not self._card_declares_tool_capture
                and not self._warned_about_missing_declaration
            ):
                logger.warning(
                    "Server sent tool capture data in artifact metadata but "
                    "its AgentCard does not declare the %s extension",
                    A2AToolCaptureExtension.URI,
                )
                self._warned_about_missing_declaration = True
            tool_calls.extend(calls)

    def _apply_task_snapshot(
        self,
        task: Task,
        text_parts: list[str],
        tool_calls: list[ToolCall],
    ) -> None:
        """Replace accumulated state with the full Task snapshot.

        Task snapshots carry the authoritative artifact list at a point
        in time; earlier artifact update deliveries are assumed to already
        be represented in the snapshot. Clearing and re-merging avoids
        double-counting. Falls back to the task status message for text
        if the snapshot has no textual artifacts.
        """
        text_parts.clear()
        tool_calls.clear()
        for artifact in task.artifacts:
            self._merge_artifact(artifact, text_parts, tool_calls)
        # HasField guards: status and message are proto3 optional message
        # fields; default empty messages are always truthy in protobuf.
        if (
            not text_parts
            and task.HasField("status")
            and task.status.HasField("message")
        ):
            # list() copies the repeated field to satisfy the list[Part] type.
            text_parts.append(self._text_from_parts(list(task.status.message.parts)))

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        """Execute user query using A2A protocol.

        Reads the agent's text reply and any tool calls surfaced via the
        arksim A2A tool capture extension on Task artifacts. Handles the
        ``StreamResponse`` variants from ``Client.send_message()``:

        * ``message`` - text-only, no Task lifecycle (spec says messages
          do not carry task outputs, so no tool calls are extracted).
        * ``task`` - Task snapshot with the full artifact list.
        * ``artifact_update`` - streaming artifact delta; accumulate its
          tool calls.
        * ``status_update`` - streaming status change; the Task snapshot
          may be intermediate, so we do NOT reset accumulated state here.
        """
        await self._ensure_initialized()

        try:
            request = SendMessageRequest(
                message=Message(
                    role=Role.ROLE_USER,
                    parts=[Part(text=user_query)],
                    message_id=str(uuid.uuid4()),
                    context_id=self.chat_id,
                    extensions=[A2AToolCaptureExtension.URI],
                ),
            )

            # Guard against a race where close() cleared self._client
            # between _ensure_initialized's fast-path check and this call.
            if self._client is None:
                raise RuntimeError(
                    "A2A client is closed; execute() called after close()"
                )

            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            final_message_text: str | None = None

            response: StreamResponse
            async for response in self._client.send_message(request):
                if response.HasField("message"):
                    # Per A2A spec 3.7, Messages do not carry task outputs.
                    # Record text as a fallback if no Task artifacts have text.
                    final_message_text = self._text_from_parts(
                        list(response.message.parts)
                    )
                elif response.HasField("artifact_update"):
                    # Streaming delta: accumulate the update's artifact.
                    if not response.artifact_update.HasField("artifact"):
                        logger.debug("Ignoring artifact_update with no artifact")
                        continue
                    self._merge_artifact(
                        response.artifact_update.artifact,
                        text_parts,
                        tool_calls,
                    )
                elif response.HasField("status_update"):
                    # Status-only update: no artifact data, don't reset.
                    pass
                elif response.HasField("task"):
                    # Task snapshot: authoritative full artifact list.
                    self._apply_task_snapshot(response.task, text_parts, tool_calls)

            if text_parts:
                content = "\n".join(text_parts)
            elif final_message_text is not None:
                # Task was empty or no Task arrived; fall back to the
                # Message text (A2A spec allows text-only Message responses,
                # and servers occasionally emit a Task snapshot with empty
                # artifacts after a conversational Message).
                content = final_message_text
            else:
                content = ""
            return AgentResponse(content=content, tool_calls=tool_calls)

        except Exception:
            logger.error("Error calling A2A agent", exc_info=True)
            raise

    async def close(self) -> None:
        """Close the httpx client and cleanup resources.

        Acquires ``_init_lock`` so we never race with an in-flight
        ``_ensure_initialized`` call (which would otherwise leave us with
        a live client after we cleared state).
        """
        async with self._init_lock:
            if self._httpx_client is not None:
                with contextlib.suppress(Exception):
                    await self._httpx_client.aclose()
            self._httpx_client = None
            self._client = None
            self._initialized = False
            # Reset per-session flags so a re-init against a different
            # endpoint starts with a clean slate.
            self._card_declares_tool_capture = False
            self._warned_about_missing_declaration = False
