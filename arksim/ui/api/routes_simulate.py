"""Simulation API endpoints."""

import asyncio
import threading

from fastapi import APIRouter, Request
from pydantic import BaseModel

from arksim.constants import DEFAULT_MODEL, DEFAULT_PROVIDER
from arksim.ui.api.state import AppState

# NOTE: _run_simulation runs in a background thread with its own
# event loop (via asyncio.run) so that the main FastAPI loop stays
# responsive.  Broadcasts use run_coroutine_threadsafe to push
# updates immediately to the main loop — this avoids the latency
# that would occur if the simulation task shared the same loop
# (broadcasts would queue behind CPU-bound simulation work).

router = APIRouter(tags=["simulate"])


class SimulateRequest(BaseModel):
    """Request body for starting a simulation."""

    agent_config_file_path: str
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    num_conversations: int = 3
    max_turns: int = 5
    num_workers: int | str = "auto"
    scenario_file: str | None = None
    output_file_path: str | None = None


@router.post("/simulate")
def start_simulation(body: SimulateRequest, request: Request) -> dict:
    """Start a simulation job in a background thread."""
    app_state: AppState = request.app.state.arksim

    with app_state._lock:
        if app_state.simulate.status == "running":
            return {"error": "Simulation already running", "status": "running"}

        app_state.simulate.status = "running"
        app_state.simulate.error = None
        app_state.simulate.result = None
        app_state.simulate.output_dir = None

    thread = threading.Thread(
        target=_run_simulation,
        args=(app_state, body),
        daemon=True,
    )
    thread.start()
    return {"status": "running"}


@router.get("/simulate/status")
def simulation_status(request: Request) -> dict:
    """Get current simulation job status."""
    app_state: AppState = request.app.state.arksim
    job = app_state.simulate
    return {
        "status": job.status,
        "output_dir": job.output_dir,
        "error": job.error,
    }


@router.post("/simulate/cancel")
def cancel_simulation(request: Request) -> dict:
    """Cancel a running simulation."""
    app_state: AppState = request.app.state.arksim
    if app_state.simulate.status != "running":
        return {"error": "No simulation running"}
    app_state.simulate.cancel_event.set()
    with app_state._lock:
        app_state.simulate.status = "cancelled"
        app_state.simulate.error = None
    app_state.broadcast_status("simulate", "cancelled")
    return {"status": "cancelled"}


def _run_simulation(app_state: AppState, body: SimulateRequest) -> None:
    """Execute simulation in a background thread.

    Uses asyncio.run() to drive the async run_simulation_func
    in its own event loop, keeping the main FastAPI loop free.
    """
    app_state.simulate.cancel_event.clear()

    with app_state.ws_log_handler():
        try:
            import os

            from arksim.simulation_engine import SimulationInput
            from arksim.utils.output import resolve_output_dir
            from arksim.utils.output.types import OutputDir

            output_dir = resolve_output_dir(
                os.path.join(body.output_file_path, OutputDir.SIMULATION.value)
            )
            output_file_path = os.path.join(output_dir, "simulation.json")

            settings = SimulationInput(
                agent_config_file_path=body.agent_config_file_path,
                model=body.model,
                provider=body.provider,
                num_conversations_per_scenario=body.num_conversations,
                max_turns=body.max_turns,
                num_workers=body.num_workers,
                scenario_file_path=body.scenario_file or None,
                output_file_path=output_file_path,
            )

            # A scenario file is required for simulation.
            if not body.scenario_file:
                raise ValueError(
                    "A scenario file is required. Please provide one in the "
                    "Scenario Input section, or use Build Scenarios to create one."
                )

            from arksim.scenario import Scenarios
            from arksim.utils.output.utils import load_json_file

            scenario_data = load_json_file(body.scenario_file)
            scenario_output = Scenarios.model_validate(scenario_data)

            from arksim.simulation_engine import run_simulation

            def _on_progress(completed: int, total: int) -> None:
                if app_state.simulate.cancel_event.is_set():
                    raise InterruptedError("Simulation cancelled")
                app_state.broadcast_progress("simulate", completed, total)

            result = asyncio.run(
                run_simulation(
                    settings,
                    scenario_output=scenario_output,
                    on_progress=_on_progress,
                )
            )

            with app_state._lock:
                if app_state.simulate.status == "cancelled":
                    return
                app_state.simulate.status = "done"
                app_state.simulate.output_dir = output_dir
                app_state.simulate.result = result

            app_state.broadcast_status("simulate", "done", output_dir=output_dir)

        except InterruptedError:
            pass  # Already marked cancelled by the cancel endpoint

        except Exception as e:
            with app_state._lock:
                if app_state.simulate.status == "cancelled":
                    return
                app_state.simulate.status = "failed"
                app_state.simulate.error = str(e)

            app_state.broadcast_status("simulate", "failed", error=str(e))
