# SPDX-License-Identifier: Apache-2.0
"""Results API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from arksim.ui.api.routes_filesystem import PROJECT_ROOT

router = APIRouter(tags=["results"])


def _validate_results_dir(dir_path: str) -> str | None:
    """Validate that a results directory is within PROJECT_ROOT.

    Returns the resolved path or None if invalid.
    """
    resolved = os.path.abspath(dir_path)
    root = os.path.abspath(PROJECT_ROOT)
    if not Path(resolved).is_relative_to(root):
        return None
    return resolved


@router.get("/results")
def get_results(request: Request, dir: str | None = None) -> dict:
    """Get evaluation results from memory or disk."""
    app_state = request.app.state.arksim

    # Try in-memory results first
    if app_state.evaluate.result is not None:
        return {
            "results": app_state.evaluate.result.model_dump(),
            "output_dir": app_state.evaluate.output_dir,
        }

    # Try loading from disk
    if dir:
        resolved_dir = _validate_results_dir(dir)
        if not resolved_dir:
            return {"results": None, "output_dir": None}
        results_file = os.path.join(resolved_dir, "evaluation_results.json")
        if os.path.exists(results_file):
            import json

            with open(results_file) as f:
                data = json.load(f)
            return {"results": data, "output_dir": dir}

    return {"results": None, "output_dir": None}


@router.get("/results/report", response_model=None)
def get_report(dir: str) -> FileResponse | JSONResponse:
    """Serve the HTML report file."""
    resolved_dir = _validate_results_dir(dir)
    if not resolved_dir:
        return JSONResponse({"error": "Invalid directory"}, status_code=403)
    report_path = os.path.join(resolved_dir, "final_report.html")
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="text/html")
    return JSONResponse(
        {"error": "Report not found"},
        status_code=404,
    )


_ALLOWED_FILES = {
    "final_report.html",
    "evaluation.json",
}


@router.get("/results/file", response_model=None)
def get_result_file(dir: str, name: str) -> FileResponse | JSONResponse:
    """Serve an output file from the results directory."""
    if name not in _ALLOWED_FILES:
        return JSONResponse(
            {"error": "File not allowed"},
            status_code=403,
        )
    resolved_dir = _validate_results_dir(dir)
    if not resolved_dir:
        return JSONResponse({"error": "Invalid directory"}, status_code=403)
    file_path = os.path.join(resolved_dir, name)
    if not os.path.exists(file_path):
        return JSONResponse(
            {"error": f"{name} not found"},
            status_code=404,
        )
    return FileResponse(file_path)
