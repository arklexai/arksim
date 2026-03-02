"""WebSocket endpoint for real-time log streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from arksim.ui.api.state import AppState

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket) -> None:
    """Stream log messages and status updates to the client."""
    await websocket.accept()

    app_state: AppState = websocket.app.state.arksim
    app_state.ws_connections.append(websocket)

    # Send current state on connect
    await websocket.send_json(
        {
            "type": "status",
            "job": "simulate",
            "status": app_state.simulate.status,
            "output_dir": app_state.simulate.output_dir,
        }
    )
    await websocket.send_json(
        {
            "type": "status",
            "job": "evaluate",
            "status": app_state.evaluate.status,
            "output_dir": app_state.evaluate.output_dir,
        }
    )

    try:
        while True:
            # Keep connection alive; client doesn't send messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in app_state.ws_connections:
            app_state.ws_connections.remove(websocket)
