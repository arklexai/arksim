# SPDX-License-Identifier: Apache-2.0
"""Server-side state manager for arksim UI jobs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Generator
from dataclasses import dataclass, field

from fastapi import WebSocket
from pydantic import BaseModel

# ── WebSocket event models ────────────────────────────────


class StatusEvent(BaseModel):
    """Job status change event."""

    type: str = "status"
    job: str
    status: str
    output_dir: str | None = None
    error: str | None = None


class ProgressEvent(BaseModel):
    """Job progress update event."""

    type: str = "progress"
    job: str
    completed: int
    total: int


class LogEvent(BaseModel):
    """Log record event."""

    type: str = "log"
    level: str
    message: str


# ── State ─────────────────────────────────────────────────


@dataclass
class JobState:
    """State for a single job (simulate or evaluate)."""

    status: str = "idle"  # idle | running | done | failed | cancelled
    output_dir: str | None = None
    error: str | None = None
    result: object | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class AppState:
    """Application-wide state, single user."""

    simulate: JobState = field(default_factory=JobState)
    evaluate: JobState = field(default_factory=JobState)
    ws_connections: list[WebSocket] = field(default_factory=list)
    loop: asyncio.AbstractEventLoop | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected WebSockets."""
        dead: list[WebSocket] = []
        for ws in self.ws_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_connections.remove(ws)

    # ── Async broadcast (same event loop) ─────────────

    async def broadcast_status_async(
        self,
        job: str,
        status: str,
        output_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        """Broadcast a status event (call from async context)."""
        event = StatusEvent(job=job, status=status, output_dir=output_dir, error=error)
        await self.broadcast(event.model_dump(exclude_none=True))

    async def broadcast_progress_async(
        self,
        job: str,
        completed: int,
        total: int,
    ) -> None:
        """Broadcast a progress event (call from async context)."""
        event = ProgressEvent(job=job, completed=completed, total=total)
        await self.broadcast(event.model_dump())

    # ── Thread-safe broadcast (background threads) ────

    def broadcast_status(
        self,
        job: str,
        status: str,
        output_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        """Schedule a status broadcast (thread-safe)."""
        event = StatusEvent(job=job, status=status, output_dir=output_dir, error=error)
        loop = self.loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast(event.model_dump(exclude_none=True)), loop
            )

    def broadcast_progress(
        self,
        job: str,
        completed: int,
        total: int,
    ) -> None:
        """Schedule a progress broadcast (thread-safe)."""
        event = ProgressEvent(job=job, completed=completed, total=total)
        loop = self.loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(event.model_dump()), loop)

    @contextlib.contextmanager
    def ws_log_handler(self) -> Generator[logging.Handler]:
        """Attach a WebSocket log handler for the duration."""
        root = logging.getLogger("arksim")
        handler = WebSocketLogHandler(self)
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)
        try:
            yield handler
        finally:
            root.removeHandler(handler)


class WebSocketLogHandler(logging.Handler):
    """Push log records to all connected WebSocket clients."""

    def __init__(self, app_state: AppState) -> None:
        super().__init__()
        self.app_state = app_state
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        event = LogEvent(level=record.levelname, message=self.format(record))
        loop = self.app_state.loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.app_state.broadcast(event.model_dump()), loop
            )
