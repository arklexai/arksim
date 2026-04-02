# SPDX-License-Identifier: Apache-2.0
"""Shared dynamic module loader for custom agents and custom metrics.

Both custom agents and custom metrics need to load user-provided ``.py``
files at runtime.  This module provides a single, robust loading function
so the two features share the same validation, error handling, and
``sys.path`` / ``sys.modules`` management.

Modules are cached by resolved file path so that module-level code
(processor registration, connection setup, logging config) runs exactly
once regardless of how many times the module is loaded.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache by resolved file path. arksim only needs fresh class instances
# (one per conversation), not fresh modules. Caching ensures module-level
# side effects (processor registration, logging setup, etc.) run once.
_module_cache: dict[str, types.ModuleType] = {}


def load_module_from_file(file_path: str) -> types.ModuleType:
    """Load a Python module from a ``.py`` file path.

    Returns a cached module if the same resolved path was loaded before.
    This ensures module-level code runs exactly once even when the
    simulator creates multiple agent instances from the same file.

    Args:
        file_path: Absolute or relative path to a ``.py`` file.

    Returns:
        The loaded module object.

    Raises:
        ValueError: If the path does not end in ``.py``.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the module spec cannot be created or the module
            fails to execute.
    """
    path = Path(file_path).resolve()

    if path.suffix != ".py":
        raise ValueError(f"Module path must be a .py file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")

    cache_key = str(path)
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    # Use a unique name to avoid collisions in sys.modules when multiple
    # files share the same stem (e.g. two different custom_agent.py files).
    module_name = f"_arksim_{uuid.uuid4().hex[:8]}_{path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec from: {path}")

    # Add parent directory to sys.path so the loaded module can import
    # sibling packages (e.g. `from agent_server.core import ...` or
    # `from my_metrics_helpers import ...`).  The entry is kept after
    # loading to support deferred imports inside the agent (e.g. lazy
    # imports in method bodies).  The guard prevents duplicate entries.
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Failed to load module from {path}: {e}") from e

    _module_cache[cache_key] = module
    return module
