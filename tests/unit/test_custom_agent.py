# SPDX-License-Identifier: Apache-2.0
"""Tests for custom Python agent connector."""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from arksim.config import AgentConfig, AgentType, CustomConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.agent.clients.custom import (
    CustomAgent,
    load_custom_agent_class,
)
from arksim.simulation_engine.agent.factory import create_agent

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def base_agent_module(tmp_path: Path) -> Path:
    """Create a temp .py file with a single BaseAgent subclass."""
    code = textwrap.dedent("""\
        import uuid
        from arksim.simulation_engine.agent.base import BaseAgent

        class MyTestAgent(BaseAgent):
            def __init__(self, agent_config):
                super().__init__(agent_config)
                self.chat_id = str(uuid.uuid4())
                self.calls = []

            async def get_chat_id(self):
                return self.chat_id

            async def execute(self, user_query, **kwargs):
                self.calls.append(user_query)
                return f"response to: {user_query}"

            async def close(self):
                self.calls.clear()
    """)
    module_file = tmp_path / "base_agent_mod.py"
    module_file.write_text(code)
    return module_file


@pytest.fixture
def multi_agent_module(tmp_path: Path) -> Path:
    """Create a temp .py file with multiple BaseAgent subclasses."""
    code = textwrap.dedent("""\
        import uuid
        from arksim.simulation_engine.agent.base import BaseAgent

        class AgentOne(BaseAgent):
            async def get_chat_id(self):
                return str(uuid.uuid4())
            async def execute(self, user_query, **kwargs):
                return "one"

        class AgentTwo(BaseAgent):
            async def get_chat_id(self):
                return str(uuid.uuid4())
            async def execute(self, user_query, **kwargs):
                return "two"
    """)
    module_file = tmp_path / "multi_agent_mod.py"
    module_file.write_text(code)
    return module_file


@pytest.fixture
def no_agent_module(tmp_path: Path) -> Path:
    """Create a temp .py file with no BaseAgent subclass."""
    code = textwrap.dedent("""\
        class NotAnAgent:
            async def execute(self, user_query, **kwargs):
                return "nope"
    """)
    module_file = tmp_path / "no_agent_mod.py"
    module_file.write_text(code)
    return module_file


def _make_custom_config(
    module_path: str,
    class_name: str | None = None,
) -> dict:
    """Build an AgentConfig dict for a custom agent."""
    config: dict = {
        "module_path": str(module_path),
    }
    if class_name is not None:
        config["class_name"] = class_name
    return {
        "agent_type": "custom",
        "agent_name": "test-custom-agent",
        "config": config,
    }


# ── CustomConfig tests ──────────────────────────────────────────────────────


class TestCustomConfig:
    """Tests for CustomConfig model."""

    def test_valid_config(self) -> None:
        config = CustomConfig(
            module_path="./my_agent.py",
            class_name="MyAgent",
        )
        assert config.module_path == "./my_agent.py"
        assert config.class_name == "MyAgent"

    def test_class_name_optional(self) -> None:
        config = CustomConfig(module_path="./my_agent.py")
        assert config.class_name is None

    def test_agent_class_valid(self) -> None:
        """agent_class alone is sufficient — no module_path needed."""
        config = CustomConfig(
            agent_class=type(
                "Dummy",
                (BaseAgent,),
                {
                    "get_chat_id": lambda self: "id",
                    "execute": lambda self, q, **kw: "ok",
                },
            )
        )
        assert config.agent_class is not None
        assert config.module_path is None

    def test_neither_source_raises(self) -> None:
        with pytest.raises(ValidationError, match="agent_class.*module_path"):
            CustomConfig()

    def test_both_sources_raises(self) -> None:
        with pytest.raises(ValidationError, match="Cannot specify both"):
            CustomConfig(
                agent_class=BaseAgent,
                module_path="./agent.py",
            )


# ── AgentConfig integration tests ──────────────────────────────────────────


class TestAgentConfigCustom:
    """Tests for AgentConfig with custom agent type."""

    def test_valid_custom_config(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)

        assert config.agent_type == AgentType.CUSTOM.value
        assert isinstance(config.config, CustomConfig)
        assert config.api_config is None

    def test_custom_config_without_class_name(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module))
        config = AgentConfig(**config_data)

        assert config.config.class_name is None

    def test_custom_requires_config(self) -> None:
        with pytest.raises(ValueError, match="requires 'config'"):
            AgentConfig(
                agent_type="custom",
                agent_name="test",
            )

    def test_custom_requires_class_source(self) -> None:
        """config with only class_name (no agent_class or module_path) raises."""
        with pytest.raises(ValidationError, match="agent_class.*module_path"):
            AgentConfig(
                agent_type="custom",
                agent_name="test",
                config={"class_name": "Agent"},
            )


# ── Factory tests ───────────────────────────────────────────────────────────


class TestCreateCustomAgent:
    """Tests for create_agent with custom type."""

    def test_factory_creates_custom_agent(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)

        agent = create_agent(config)

        assert isinstance(agent, CustomAgent)
        assert isinstance(agent, BaseAgent)


# ── CustomAgent execution tests ─────────────────────────────────────────────


class TestCustomAgentExecution:
    """Tests for CustomAgent runtime behavior."""

    async def test_execute(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)
        agent = CustomAgent(config)

        result = await agent.execute("hello")

        assert result == "response to: hello"

    async def test_get_chat_id_before_execute(self, base_agent_module: Path) -> None:
        """Before execute, get_chat_id returns the wrapper's own UUID."""
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)
        agent = CustomAgent(config)

        chat_id = await agent.get_chat_id()

        assert isinstance(chat_id, str)
        assert len(chat_id) > 0
        assert chat_id == agent.chat_id  # wrapper's own id

    async def test_get_chat_id_delegates_after_execute(
        self, base_agent_module: Path
    ) -> None:
        """After execute, get_chat_id delegates to the inner agent."""
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)
        agent = CustomAgent(config)

        wrapper_id = await agent.get_chat_id()
        await agent.execute("trigger inner agent creation")
        inner_id = await agent.get_chat_id()

        # Inner agent creates its own UUID, which differs from the wrapper's.
        assert inner_id != wrapper_id
        assert inner_id == await agent._inner.get_chat_id()

    async def test_close_delegates(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)
        agent = CustomAgent(config)

        await agent.execute("test")
        await agent.close()

    async def test_multiple_turns(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module), "MyTestAgent")
        config = AgentConfig(**config_data)
        agent = CustomAgent(config)

        r1 = await agent.execute("turn 1")
        r2 = await agent.execute("turn 2")

        assert r1 == "response to: turn 1"
        assert r2 == "response to: turn 2"


# ── Auto-discovery tests ────────────────────────────────────────────────────


class TestAutoDiscovery:
    """Tests for auto-discovery of BaseAgent subclass when class_name is omitted."""

    def test_auto_discovers_single_subclass(self, base_agent_module: Path) -> None:
        cls = load_custom_agent_class(str(base_agent_module))
        assert issubclass(cls, BaseAgent)
        assert cls.__name__ == "MyTestAgent"

    def test_errors_on_multiple_subclasses(self, multi_agent_module: Path) -> None:
        with pytest.raises(TypeError, match="Multiple BaseAgent subclasses"):
            load_custom_agent_class(str(multi_agent_module))

    def test_errors_on_no_subclass(self, no_agent_module: Path) -> None:
        with pytest.raises(TypeError, match="No BaseAgent subclass found"):
            load_custom_agent_class(str(no_agent_module))

    def test_class_name_overrides_discovery(self, multi_agent_module: Path) -> None:
        cls = load_custom_agent_class(str(multi_agent_module), "AgentOne")
        assert cls.__name__ == "AgentOne"

    async def test_execute_with_auto_discovery(self, base_agent_module: Path) -> None:
        config_data = _make_custom_config(str(base_agent_module))
        config = AgentConfig(**config_data)
        agent = CustomAgent(config)

        result = await agent.execute("auto-discovered")

        assert result == "response to: auto-discovered"


# ── Code-based agent_class tests ───────────────────────────────────────────


class _InlineAgent(BaseAgent):
    """A BaseAgent subclass defined directly in test code (no file loading)."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(__import__("uuid").uuid4())

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        return f"inline: {user_query}"


class TestAgentClassCodePath:
    """Tests for passing agent_class directly (no file loading)."""

    def test_agent_config_with_agent_class(self) -> None:
        config = AgentConfig(
            agent_type="custom",
            agent_name="inline-test",
            config=CustomConfig(agent_class=_InlineAgent),
        )
        assert config.config.agent_class is _InlineAgent
        assert config.config.module_path is None

    async def test_execute_with_agent_class(self) -> None:
        config = AgentConfig(
            agent_type="custom",
            agent_name="inline-test",
            config=CustomConfig(agent_class=_InlineAgent),
        )
        agent = CustomAgent(config)

        result = await agent.execute("hello")

        assert result == "inline: hello"

    async def test_get_chat_id_with_agent_class(self) -> None:
        config = AgentConfig(
            agent_type="custom",
            agent_name="inline-test",
            config=CustomConfig(agent_class=_InlineAgent),
        )
        agent = CustomAgent(config)

        await agent.execute("trigger")
        chat_id = await agent.get_chat_id()

        assert chat_id == await agent._inner.get_chat_id()


# ── Dynamic import error handling tests ─────────────────────────────────────


class TestCustomAgentErrors:
    """Tests for error handling in custom agent loading."""

    def test_missing_module_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="Module file not found"):
            load_custom_agent_class("/nonexistent/path/agent.py", "Agent")

    def test_missing_class_in_module(self, base_agent_module: Path) -> None:
        with pytest.raises(AttributeError, match="does not have class"):
            load_custom_agent_class(str(base_agent_module), "NonexistentClass")

    def test_not_a_base_agent_subclass(self, no_agent_module: Path) -> None:
        with pytest.raises(TypeError, match="must be a subclass of"):
            load_custom_agent_class(str(no_agent_module), "NotAnAgent")

    def test_non_py_file(self) -> None:
        with pytest.raises(ValueError, match="must be a .py file"):
            load_custom_agent_class("/some/path/agent.txt", "Agent")
