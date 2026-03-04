# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator

from arksim.config.types import AgentType
from arksim.config.utils import resolve_env_vars


class A2AConfig(BaseModel):
    """API configuration for A2A (Agent-to-Agent) agent type."""

    endpoint: str = Field(..., description="Endpoint URL for the A2A agent server")
    headers: dict[str, str] | None = Field(
        None,
        description="HTTP headers for A2A requests. Values can use ${ENV_VAR} syntax for environment variable substitution.",
    )

    def get_headers(self) -> dict[str, str] | None:
        """Get headers with environment variable substitution.

        Supports ${ENV_VAR} syntax in header values, which will be replaced
        with the corresponding environment variable value.
        """
        if not self.headers:
            return None

        return resolve_env_vars(self.headers)


class ChatCompletionsConfig(BaseModel):
    """API configuration for chat completion agent type."""

    endpoint: str | None = Field(None, description="Chat completion endpoint URL")
    headers: dict[str, str] | None = Field(
        None,
        description="HTTP headers for chat requests. Values can use ${ENV_VAR} syntax for env var substitution.",
    )
    body: dict[str, Any] | None = Field(
        None,
        description="Request body template for chat completion requests",
    )

    # Optional fields for Azure OpenAI
    azure_config: dict[str, Any] | None = Field(
        None, description="Azure OpenAI configuration"
    )

    @model_validator(mode="after")
    def validate_config_format(self) -> Self:
        """Validate that required fields are provided."""
        if not self.body:
            raise ValueError(
                "ChatCompletions agent configuration requires 'body' field"
            )

        return self

    def get_endpoint(self) -> str:
        """Get endpoint URL with environment variable substitution."""
        if not self.endpoint:
            raise ValueError("ChatCompletions endpoint is not configured")
        resolved = resolve_env_vars({"endpoint": self.endpoint})
        return resolved["endpoint"]

    def get_headers(self) -> dict[str, str]:
        """Get headers.

        Supports ${ENV_VAR} syntax in header values, which will be replaced
        with the corresponding environment variable value.
        """
        base_headers = self.headers or {}
        resolved_headers = resolve_env_vars(base_headers)
        return resolved_headers


class AgentConfig(BaseModel):
    """Agent configuration."""

    agent_name: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Agent type identifier")
    api_config: ChatCompletionsConfig | A2AConfig = Field(
        ..., description="Agent configuration"
    )

    @model_validator(mode="before")
    @classmethod
    def parse_config(cls, data: object) -> object:
        """Parse config based on top-level agent_type."""
        if isinstance(data, dict):
            agent_type = data.get("agent_type")
            config_data = data.get("api_config")

            if agent_type == AgentType.CHAT_COMPLETIONS.value:
                data["api_config"] = ChatCompletionsConfig(**config_data)
            elif agent_type == AgentType.A2A.value:
                data["api_config"] = A2AConfig(**config_data)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
        else:
            raise ValueError("Agent configuration must be a dictionary")

        return data

    @classmethod
    def load(cls, path: str | Path) -> AgentConfig:
        """Load agent configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

        return cls.model_validate(data)
