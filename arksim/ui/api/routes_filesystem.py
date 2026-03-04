# SPDX-License-Identifier: Apache-2.0
"""Filesystem browsing API endpoints."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["filesystem"])

PROJECT_ROOT = os.getcwd()

_PATH_KEYS = {
    "agent_config_file_path",
    "scenario_file_path",
    "simulation_file_path",
    "output_dir",
}

# Keys whose values are lists of file paths (each element resolved).
_LIST_PATH_KEYS = {
    "custom_metrics_file_paths",
}


def _resolve_path(path: str) -> str:
    """Resolve a relative path against PROJECT_ROOT.

    The resolved path must remain within PROJECT_ROOT to
    prevent directory traversal attacks.
    """
    if not path:
        return path
    if os.path.isabs(path):
        resolved = os.path.abspath(path)
    else:
        resolved = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    root = os.path.abspath(PROJECT_ROOT)
    if not Path(resolved).is_relative_to(root):
        raise ValueError(f"Path must be within the project directory: {root}")
    return resolved


def _validate_write_path(path: str) -> str:
    """Validate that a write path is within PROJECT_ROOT.

    Raises ValueError if the resolved path escapes the
    project directory.
    """
    resolved = os.path.abspath(os.path.expanduser(path))
    root = os.path.abspath(PROJECT_ROOT)
    if not Path(resolved).is_relative_to(root):
        raise ValueError(f"Write path must be within the project directory: {root}")
    return resolved


@router.get("/fs/browse")
def browse_directory(
    path: str = "~",
    show_hidden: bool = False,
) -> dict:
    """List contents of a directory on the server."""
    resolved = Path(path).expanduser().resolve()
    root = Path(PROJECT_ROOT).resolve()
    if not resolved.is_dir() or not (resolved == root or root in resolved.parents):
        resolved = root

    try:
        entries_raw = list(resolved.iterdir())
    except PermissionError:
        entries_raw = []

    if not show_hidden:
        entries_raw = [e for e in entries_raw if not e.name.startswith(".")]

    entries_raw.sort(key=lambda p: p.name.lower())
    entries_raw.sort(key=lambda p: not p.is_dir())

    entries = [
        {
            "name": e.name,
            "type": "directory" if e.is_dir() else "file",
            "path": str(e),
        }
        for e in entries_raw
    ]

    parent_path = resolved.parent
    if parent_path != resolved and (parent_path == root or root in parent_path.parents):
        parent = str(parent_path)
    else:
        parent = None

    return {
        "current": str(resolved),
        "parent": parent,
        "entries": entries,
    }


@router.get("/fs/root")
def get_project_root() -> dict:
    """Return the working directory arksim ui was launched from."""
    return {"root": PROJECT_ROOT}


@router.get("/fs/configs")
def list_configs() -> dict:
    """Discover YAML config files in the project tree."""
    patterns = [
        os.path.join(PROJECT_ROOT, "config*.yaml"),
        os.path.join(PROJECT_ROOT, "config*.yml"),
        os.path.join(PROJECT_ROOT, "examples", "**", "config*.yaml"),
        os.path.join(PROJECT_ROOT, "arksim", "config*.yaml"),
        os.path.join(PROJECT_ROOT, "arksim", "examples", "**", "config*.yaml"),
    ]
    found: set[str] = set()
    for p in patterns:
        found.update(glob.glob(p, recursive=True))

    def _key(path: str) -> tuple[int, str]:
        name = os.path.basename(path).lower()
        if "simulate" in name:
            return (0, path)
        if "evaluate" in name:
            return (2, path)
        return (1, path)

    configs = sorted(found, key=_key)
    return {
        "configs": [
            {"path": p, "relative": os.path.relpath(p, PROJECT_ROOT)} for p in configs
        ]
    }


@router.get("/fs/config")
def load_config(path: str) -> dict:
    """Load and return a YAML config file."""
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return {"error": str(e)}
    if not resolved or not os.path.exists(resolved):
        return {"error": f"File not found: {path}"}

    with open(resolved) as f:
        cfg = yaml.safe_load(f) or {}

    config_dir = os.path.dirname(resolved)

    def _resolve_rel(val: str) -> str:
        if not val or os.path.isabs(val):
            return val
        if val.startswith("./"):
            val = val.removeprefix("./")
        return os.path.join(config_dir, val)

    for key in _PATH_KEYS:
        val = cfg.get(key)
        if val and isinstance(val, str):
            cfg[key] = _resolve_rel(val)

    for key in _LIST_PATH_KEYS:
        val = cfg.get(key)
        if isinstance(val, list):
            cfg[key] = [_resolve_rel(v) for v in val if isinstance(v, str)]

    return {"settings": cfg}


class SaveConfigRequest(BaseModel):
    """Request body for saving a config file."""

    settings: dict[str, Any]
    path: str | None = None


@router.post("/fs/config")
def save_config(body: SaveConfigRequest) -> dict:
    """Save form values as a YAML config file."""
    # Filter out empty/null values
    cfg = {k: v for k, v in body.settings.items() if v not in (None, "")}

    raw_path = body.path or os.path.join(PROJECT_ROOT, "config_simulate.yaml")

    try:
        save_path = _validate_write_path(raw_path)
        with open(save_path, "w") as f:
            yaml.dump(
                cfg,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        return {"path": save_path}
    except Exception as e:
        return {"error": str(e)}


@router.get("/fs/scenario/demo")
def load_demo_scenario() -> dict:
    """Load the built-in demo scenario."""
    import json

    demo_path = os.path.join(
        PROJECT_ROOT,
        "examples",
        "demo",
        "results",
        "scenario",
        "scenario.json",
    )
    if not os.path.exists(demo_path):
        return {"error": "Demo scenario not found"}

    with open(demo_path) as f:
        data = json.load(f)
    data["_path"] = demo_path
    return data


@router.get("/fs/scenario")
def load_scenario(path: str) -> dict:
    """Load a scenario.json file."""
    import json

    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return {"error": str(e)}
    if not resolved or not os.path.exists(resolved):
        return {"error": f"File not found: {path}"}

    try:
        with open(resolved) as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}


class SaveScenarioRequest(BaseModel):
    """Request body for saving a scenario file."""

    data: dict[str, Any]
    path: str


@router.post("/fs/scenario")
def save_scenario(body: SaveScenarioRequest) -> dict:
    """Save scenario data as a JSON file."""
    import json

    try:
        save_path = _validate_write_path(body.path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(body.data, f, indent=2)
            f.write("\n")
        return {"path": body.path}
    except Exception as e:
        return {"error": str(e)}
