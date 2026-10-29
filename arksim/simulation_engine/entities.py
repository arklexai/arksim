# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from arksim.config.core.agent import AgentConfig
from arksim.config.utils import resolve_model_paths
from arksim.constants import DEFAULT_MODEL, DEFAULT_PROVIDER
from arksim.llms.chat.base.usage import TokenUsage
from arksim.simulation_engine.core.state import ConversationLedger
from arksim.simulation_engine.tool_types import ToolCall
from arksim.tracing.config import TraceReceiverConfig
from arksim.utils.concurrency import validate_num_workers


class TurnHooks(Protocol):
    """Async hooks the simulator awaits around each ``agent.execute()``.

    ``turn_id`` is zero-based and equals the assistant ``Message.turn_id``.
    Headers returned by ``before_agent_execute`` apply to that one agent
    request only. An exception from ``before_agent_execute`` aborts the
    conversation before the request is sent; ``after_agent_execute`` runs in
    ``finally`` with the in-flight exception (or None) and never masks it.
    """

    async def before_agent_execute(self, turn_id: int) -> Mapping[str, str] | None: ...

    async def after_agent_execute(
        self, turn_id: int, error: BaseException | None
    ) -> None: ...


class SimulationInput(BaseModel):
    """Input configuration for the simulation engine module."""

    agent_config_file_path: str | None = Field(
        default=None,
        description="Path to the agent API configuration file (JSON). "
        "Alternative to inline agent_config.",
    )
    agent_config: AgentConfig | None = Field(
        default=None,
        description="Inline agent configuration (agent_type, agent_name, api_config). "
        "Takes precedence over agent_config_file_path when both are present.",
    )
    scenario_file_path: str | None = Field(
        default=None,
        description="Path to the scenarios file",
    )
    model: str = Field(default=DEFAULT_MODEL, description="LLM model for simulation")
    provider: str | None = Field(default=DEFAULT_PROVIDER, description="LLM provider")
    num_conversations_per_scenario: int = Field(
        default=5, description="Number of conversations per scenario to simulate"
    )
    max_turns: int = Field(default=5, description="Maximum turns per conversation")
    num_workers: int | str = Field(
        default=50,
        description=(
            "Number of parallel workers (use 'auto' to default to "
            "num_conversations_per_scenario * number of scenarios)"
        ),
    )
    output_file_path: str = Field(
        default="./simulation.json",
        description="Output file path for simulation results",
    )
    simulated_user_prompt_template: str | None = Field(
        default=None,
        description="Jinja2 template for the simulated user system prompt",
    )
    web_search: bool = Field(
        default=True,
        description=(
            "Let the simulated user open a link the agent sent instead of "
            "taking it on faith, using the provider's hosted tool: web_fetch "
            "on anthropic, google_search on google, web_search on openai. The "
            "model decides per turn whether to use it. Ignored with a warning "
            "on providers that have none. Billed by the provider."
        ),
    )
    trace_receiver: TraceReceiverConfig | None = Field(
        default=None,
        description="OTel trace receiver for capturing tool calls from agent spans",
    )

    @model_validator(mode="after")
    def validate_simulation_input(self, info: ValidationInfo) -> Self:
        """Validate simulation input fields."""
        validate_num_workers(self.num_workers)

        # Paths from config.yaml are resolved relative to the config file's
        # directory. Paths set via CLI are left as-is (cwd-relative).
        config_path = info.context and info.context.get("config_path")
        if config_path:
            cli_overrides = (
                info.context and info.context.get("cli_overrides")
            ) or set()
            config_dir = os.path.dirname(config_path)

            resolve_model_paths(
                self,
                path_attrs=(
                    "scenario_file_path",
                    "output_file_path",
                    "agent_config_file_path",
                ),
                list_path_attrs=(),
                config_dir=config_dir,
                cli_overrides=cli_overrides,
            )

            # Resolve custom agent module_path to an absolute path early
            # so it is unambiguous regardless of later cwd changes.
            #   - Config-sourced → relative to config file's directory
            #   - CLI-sourced    → relative to cwd (where the CLI ran)
            #   - Already absolute → unchanged
            custom_cfg = (
                getattr(self.agent_config, "custom_config", None)
                if self.agent_config
                else None
            )
            module_path = (
                getattr(custom_cfg, "module_path", None) if custom_cfg else None
            )
            if module_path and not os.path.isabs(module_path):
                if "module_path" in cli_overrides:
                    resolved = os.path.abspath(module_path)
                else:
                    resolved = os.path.normpath(os.path.join(config_dir, module_path))
                object.__setattr__(custom_cfg, "module_path", resolved)

        if not self.agent_config and not self.agent_config_file_path:
            raise ValueError(
                "Either inline agent_config or agent_config_file_path must be provided."
            )

        return self


class SimulationParams(BaseModel):
    """Parameters for simulation engine."""

    num_convos_per_scenario: int = Field(default=1)
    max_turns: int = Field(default=5)
    num_workers: int | str = Field(default=50)
    output_file_path: str = Field(default="./simulation.json")
    simulated_user_prompt_template: str | None = None
    web_search: bool = Field(default=True)


# ── Conversation state during simulation ──


class ConversationState(BaseModel):
    """Mutable state tracked while a conversation is running."""

    conversation_id: str
    scenario_id: str
    conversation_history: list[dict[str, Any]]
    simulated_user_prompt_template: str
    simulated_user_profile: str
    user_goal: str
    knowledge: list[str]
    agent_context: str
    ledger: ConversationLedger | None = None


class ConversationExecutionError(Exception):
    """Raised when a conversation fails part-way through.

    Carries the partial ``ConversationState`` (turns completed before the
    failure) so callers can persist a partial transcript alongside the error.
    ``partial_state`` is ``None`` when the failure happened before any turn
    completed.
    """

    def __init__(
        self, message: str, partial_state: ConversationState | None = None
    ) -> None:
        super().__init__(message)
        self.partial_state = partial_state


# ── Conversation output file schema models ──


class Message(BaseModel):
    """A single message in a conversation history."""

    turn_id: int
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["simulated_user", "assistant"]
    content: str
    tool_calls: list[ToolCall] | None = None


class SimulatedUserPrompt(BaseModel):
    """Captured simulated-user prompt: template + rendered variables."""

    simulated_user_prompt_template: str
    variables: dict[str, Any]


class Conversation(BaseModel):
    """A single conversation record in the output file."""

    conversation_id: str
    scenario_id: str
    conversation_history: list[Message]
    simulated_user_prompt: SimulatedUserPrompt
    # Lets the evaluator derive outcomes without re-judging. Optional for
    # older records.
    ledger: ConversationLedger | None = None


class Simulation(BaseModel):
    """Top-level envelope for the conversations output file."""

    schema_version: str
    simulator_version: str
    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    conversations: list[Conversation]
    usage: TokenUsage | None = None
