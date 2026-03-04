# SPDX-License-Identifier: Apache-2.0
"""ArkSim Control Plane — FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import threading
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse

from arksim.ui.api.routes_evaluate import router as _oss_evaluate_router
from arksim.ui.api.routes_filesystem import router as fs_router
from arksim.ui.api.routes_results import router as results_router
from arksim.ui.api.routes_simulate import router as simulate_router
from arksim.ui.api.state import AppState
from arksim.ui.api.ws_logs import router as ws_router

FRONTEND_DIR = Path(__file__).parent / "frontend"


def create_app(evaluate_router: APIRouter | None = None) -> FastAPI:
    """Create the FastAPI application."""
    _evaluate_router = evaluate_router or _oss_evaluate_router

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.arksim.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="ArkSim", lifespan=lifespan)
    app.state.arksim = AppState()

    # API routes — must be registered before the frontend
    app.include_router(simulate_router, prefix="/api")
    app.include_router(_evaluate_router, prefix="/api")
    app.include_router(results_router, prefix="/api")
    app.include_router(fs_router, prefix="/api")
    app.include_router(ws_router, prefix="/api")

    # Serve frontend files explicitly (avoids catch-all
    # mount that intercepts WebSocket routes)
    if FRONTEND_DIR.exists():

        @app.get("/")
        async def _index() -> FileResponse:
            return FileResponse(
                FRONTEND_DIR / "index.html",
                headers={"Cache-Control": "no-cache"},
            )

        @app.get("/app.js")
        async def _app_js() -> FileResponse:
            return FileResponse(
                FRONTEND_DIR / "app.js",
                media_type="application/javascript",
                headers={"Cache-Control": "no-cache"},
            )

    return app


def launch_ui(port: int = 8080) -> None:
    """Launch the ArkSim web UI."""
    for name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)

    url = f"http://localhost:{port}"
    threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=port)
