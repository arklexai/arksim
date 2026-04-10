# SPDX-License-Identifier: Apache-2.0
"""Evaluation API endpoints."""

from __future__ import annotations

import threading

from fastapi import APIRouter, Request
from pydantic import BaseModel

from arksim.constants import DEFAULT_MODEL, DEFAULT_PROVIDER
from arksim.ui.api.state import AppState

router = APIRouter(tags=["evaluate"])


class EvaluateRequest(BaseModel):
    """Request body for starting an evaluation."""

    simulation_file_path: str | None = None
    scenario_file_path: str | None = None
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    num_workers: int | str = 50
    custom_metrics_file_paths: list[str] = []
    metrics_to_run: list[str] | None = None
    generate_html_report: bool = True
    numeric_thresholds: dict[str, float] | None = None
    qualitative_failure_labels: dict[str, list[str]] | None = None


@router.post("/evaluate")
def start_evaluation(body: EvaluateRequest, request: Request) -> dict:
    """Start an evaluation job in a background thread."""
    app_state: AppState = request.app.state.arksim

    with app_state._lock:
        if app_state.evaluate.status == "running":
            return {"error": "Evaluation already running", "status": "running"}

        app_state.evaluate.status = "running"
        app_state.evaluate.error = None
        app_state.evaluate.result = None
        app_state.evaluate.output_dir = None

    thread = threading.Thread(
        target=_run_evaluation,
        args=(app_state, body),
        daemon=True,
    )
    thread.start()
    return {"status": "running"}


@router.post("/evaluate/cancel")
def cancel_evaluation(request: Request) -> dict:
    """Cancel a running evaluation."""
    app_state: AppState = request.app.state.arksim
    if app_state.evaluate.status != "running":
        return {"error": "No evaluation running"}
    app_state.evaluate.cancel_event.set()
    with app_state._lock:
        app_state.evaluate.status = "cancelled"
        app_state.evaluate.error = None
    app_state.broadcast_status("evaluate", "cancelled")
    return {"status": "cancelled"}


@router.get("/evaluate/status")
def evaluation_status(request: Request) -> dict:
    """Get current evaluation job status."""
    app_state: AppState = request.app.state.arksim
    job = app_state.evaluate
    resp: dict = {
        "status": job.status,
        "output_dir": job.output_dir,
        "error": job.error,
    }
    if job.result is not None:
        resp["results"] = job.result.model_dump()
    return resp


def _run_evaluation(app_state: AppState, body: EvaluateRequest) -> None:
    """Execute evaluation in a background thread.

    Evaluate uses sync code with a ThreadPoolExecutor
    internally, so a background thread is appropriate here
    (unlike simulate which is fully async).
    """
    app_state.evaluate.cancel_event.clear()

    with app_state.ws_log_handler():
        try:
            from arksim.evaluator import EvaluationInput
            from arksim.utils.output.types import OutputDir

            output_dir = OutputDir.EVALUATION.value

            settings = EvaluationInput(
                simulation_file_path=body.simulation_file_path,
                scenario_file_path=body.scenario_file_path,
                model=body.model,
                provider=body.provider,
                num_workers=body.num_workers,
                custom_metrics_file_paths=body.custom_metrics_file_paths,
                metrics_to_run=body.metrics_to_run or [],
                output_dir=output_dir,
                generate_html_report=body.generate_html_report,
                numeric_thresholds=body.numeric_thresholds,
                qualitative_failure_labels=body.qualitative_failure_labels,
            )

            from arksim.evaluator import run_evaluation

            def _on_progress(completed: int, total: int) -> None:
                if app_state.evaluate.cancel_event.is_set():
                    raise InterruptedError("Evaluation cancelled")
                app_state.broadcast_progress("evaluate", completed, total)

            result = run_evaluation(settings, on_progress=_on_progress)

            from arksim.evaluator import (
                check_numeric_thresholds,
                check_qualitative_failure_labels,
            )

            threshold_error: str | None = None
            if body.numeric_thresholds or body.qualitative_failure_labels:
                numeric_ok = check_numeric_thresholds(
                    result, body.numeric_thresholds or {}
                )
                qual_ok = check_qualitative_failure_labels(
                    result, body.qualitative_failure_labels or {}
                )
                if not numeric_ok or not qual_ok:
                    parts = []
                    if not numeric_ok:
                        parts.append("numeric thresholds")
                    if not qual_ok:
                        parts.append("qualitative failure labels")
                    threshold_error = (
                        f"Threshold checks failed: {' and '.join(parts)} did not pass."
                    )

            with app_state._lock:
                if app_state.evaluate.status == "cancelled":
                    return
                app_state.evaluate.status = "done"
                app_state.evaluate.output_dir = output_dir
                app_state.evaluate.result = result
                if threshold_error:
                    app_state.evaluate.error = threshold_error

            app_state.broadcast_status(
                "evaluate",
                "done",
                output_dir=output_dir,
                error=threshold_error,
            )

        except InterruptedError:
            pass  # Already marked cancelled by the cancel endpoint

        except Exception as e:
            with app_state._lock:
                if app_state.evaluate.status == "cancelled":
                    return
                app_state.evaluate.status = "failed"
                app_state.evaluate.error = str(e)

            app_state.broadcast_status("evaluate", "failed", error=str(e))
