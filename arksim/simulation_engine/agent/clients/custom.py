# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import uuid
from pathlib import Path

from arksim.config import AgentConfig, AgentType, CustomConfig
from arksim.simulation_engine.agent.base import BaseAgent

logger = logging.getLogger(__name__)


def _load_module_from_path(file_path: str) -> object:
    """Load a Python module from a file path."""
    path = Path(file_path).resolve()
    if not path.suffix == ".py":
        raise ValueError(f"Module path must be a .py file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from: {path}")

    # Add module's parent dir to sys.path so sibling packages are importable
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_module_from_dotted(module_path: str) -> object:
    """Load a Python module from a dotted module path."""
    return importlib.import_module(module_path)


def _find_base_agent_subclass(module: object) -> type[BaseAgent]:
    """Auto-discover the BaseAgent subclass in a module.

    Raises if zero or multiple subclasses are found.
    """
    candidates: list[tuple[str, type]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseAgent)
            and obj is not BaseAgent
            and obj.__module__ == module.__name__
        ):
            candidates.append((name, obj))

    if len(candidates) == 0:
        raise TypeError(
            f"No BaseAgent subclass found in module '{module.__name__}'. "
            f"Ensure your agent class inherits from "
            f"arksim.simulation_engine.agent.base.BaseAgent."
        )
    if len(candidates) > 1:
        names = ", ".join(name for name, _ in candidates)
        raise TypeError(
            f"Multiple BaseAgent subclasses found in module '{module.__name__}': "
            f"{names}. Specify 'class_name' in your config to choose one."
        )

    return candidates[0][1]


def load_custom_agent_class(
    module_path: str,
    class_name: str | None = None,
) -> type[BaseAgent]:
    """Load the custom agent class from the configured module path.

    If class_name is None, auto-discovers the single BaseAgent subclass
    in the module. If class_name is provided, loads that specific class.

    Supports two formats for module_path:
    - File path: './my_agent.py' or '/abs/path/agent.py'
    - Dotted module: 'my_package.agent'
    """
    if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
        module = _load_module_from_path(module_path)
    else:
        module = _load_module_from_dotted(module_path)

    if class_name is None:
        return _find_base_agent_subclass(module)

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Module '{module_path}' does not have class '{class_name}'"
        )

    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseAgent)):
        raise TypeError(
            f"Custom agent class '{class_name}' must be a subclass of "
            f"arksim.simulation_engine.agent.base.BaseAgent."
        )

    return cls


class CustomAgent(BaseAgent):
    """Wrapper that loads and delegates to a user-provided BaseAgent subclass."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        if agent_config.agent_type != AgentType.CUSTOM.value:
            raise ValueError("Agent config must be of type custom")

        self.custom_config: CustomConfig = agent_config.config
        self.chat_id = str(uuid.uuid4())
        self._inner: BaseAgent | None = None

    def _create_inner_agent(self) -> BaseAgent:
        """Dynamically load and instantiate the custom agent class."""
        cls = load_custom_agent_class(
            self.custom_config.module_path,
            self.custom_config.class_name,
        )
        return cls(self.agent_config)

    async def get_chat_id(self) -> str:
        """Get the chat ID."""
        return self.chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        """Execute user query by delegating to the custom agent."""
        if self._inner is None:
            self._inner = self._create_inner_agent()

        return await self._inner.execute(user_query, **kwargs)

    async def close(self) -> None:
        """Close the inner agent."""
        if self._inner is not None:
            await self._inner.close()
