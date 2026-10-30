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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


# Reserved keys inside a custom_http ``body`` that configure response extraction
# rather than the outbound request. ``response_path`` is required and names the
# reply text; ``response_extras`` is optional and appends a section per entry.
# Both are stripped from the body before it is POSTed. These names are the fixed
# config contract that the Go backend and the frontend must match exactly; they
# live here rather than in the connector because config is the lower-level module.
RESPONSE_PATH_KEY = "response_path"
RESPONSE_EXTRAS_KEY = "response_extras"
RESERVED_BODY_KEYS = (RESPONSE_PATH_KEY, RESPONSE_EXTRAS_KEY)

# Fields of one response_extras entry. A template is constant text with {name}
# holes; see simulation_engine/agent/response_render.py.
EXTRA_FIELDS = ("path", "label", "template")


class CustomHTTPConfig(BaseModel):
    """API configuration for the custom_http agent type.

    A generic HTTP connector that POSTs a caller-defined request body
    template and extracts the reply text from the JSON response. This lets
    arksim drive any single-turn JSON HTTP endpoint without code changes:
    the field names live entirely in ``body`` and its reserved keys, not in
    arksim.

    The ``body`` template uses these reserved conventions:

    * The literal placeholder string ``{{message}}`` appears as a value
      anywhere in the template (at any nesting depth) and is replaced
      with the current user query at request time.
    * ``response_path``, a dot-path to the reply string, e.g. ``data.reply``
      or ``choices.0.message.content`` (list indices supported).
    * ``response_extras``, optional ``{path, label, template}`` entries that
      append other response fields to the turn, so data the reply text does
      not contain still reaches the evaluator.

    Reserved keys are configuration, not part of the outbound request, so
    the connector strips them from the body before POSTing.
    """

    endpoint: str | None = Field(None, description="Custom HTTP endpoint URL")
    headers: dict[str, str] | None = Field(
        None,
        description="HTTP headers for requests. Values can use ${ENV_VAR} syntax for env var substitution.",
    )
    body: dict[str, Any] | None = Field(
        None,
        description=(
            "Request body template. Use the literal '{{message}}' string as a "
            "value to inject the user query, plus the reserved 'response_path' "
            "(dot-path to the reply text) and optional 'response_extras' (a list "
            "of {path, label, template} entries appending other response fields "
            "to the turn) keys."
        ),
    )

    @model_validator(mode="after")
    def validate_config_format(self) -> Self:
        """Validate that required fields are provided."""
        if not self.body:
            raise ValueError("CustomHTTP agent configuration requires 'body' field")

        path = self.body.get(RESPONSE_PATH_KEY)
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"CustomHTTP agent 'body' must include a non-empty "
                f"'{RESPONSE_PATH_KEY}' key naming the reply text"
            )
        self._validate_extras(self.body.get(RESPONSE_EXTRAS_KEY))

        # The rest of body is the outbound request template, so at least one
        # non-reserved key has to remain or there is nothing to POST.
        if not any(k not in RESERVED_BODY_KEYS for k in self.body):
            raise ValueError(
                "CustomHTTP agent 'body' must include a request template alongside "
                f"'{RESPONSE_PATH_KEY}'"
            )
        return self

    @staticmethod
    def _validate_extras(extras: Any) -> None:  # noqa: ANN401
        """Validate response_extras, which appends response fields to the turn."""
        if extras is None:
            return
        if not isinstance(extras, list):
            raise ValueError(
                f"CustomHTTP agent '{RESPONSE_EXTRAS_KEY}' must be a list of "
                "{path, label, template} entries"
            )
        for position, extra in enumerate(extras):
            where = f"'{RESPONSE_EXTRAS_KEY}' entry {position}"
            if not isinstance(extra, dict):
                raise ValueError(f"CustomHTTP agent {where} must be an object")
            unknown = sorted(set(extra) - set(EXTRA_FIELDS))
            if unknown:
                raise ValueError(
                    f"CustomHTTP agent {where} has unsupported "
                    f"{'keys' if len(unknown) > 1 else 'key'} "
                    f"{', '.join(repr(k) for k in unknown)}; "
                    f"supported: {', '.join(EXTRA_FIELDS)}"
                )
            if not isinstance(extra.get("path"), str) or not extra["path"].strip():
                raise ValueError(f"CustomHTTP agent {where} must have a 'path'")
            for field in ("label", "template"):
                if field in extra and not isinstance(extra[field], str):
                    raise ValueError(
                        f"CustomHTTP agent {where} '{field}' must be a string"
                    )

    def get_endpoint(self) -> str:
        """Get endpoint URL with environment variable substitution."""
        if not self.endpoint:
            raise ValueError("CustomHTTP endpoint is not configured")
        resolved = resolve_env_vars({"endpoint": self.endpoint})
        return resolved["endpoint"]

    def get_headers(self) -> dict[str, str]:
        """Get headers with ${ENV_VAR} substitution applied to values."""
        base_headers = self.headers or {}
        return resolve_env_vars(base_headers)


class CustomConfig(BaseModel):
    """Configuration for custom Python agent type."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_path: str | None = Field(
        None, description="Path to a .py file or dotted module path"
    )
    class_name: str | None = Field(
        None,
        description="Class name to load. If omitted, auto-discovers the BaseAgent subclass.",
    )
    agent_class: type | None = Field(
        None,
        description="Direct class reference for code-based usage (not YAML-serializable).",
    )

    @model_validator(mode="after")
    def validate_class_source(self) -> Self:
        """Require exactly one of agent_class or module_path."""
        if self.agent_class is None and self.module_path is None:
            raise ValueError("Either 'agent_class' or 'module_path' must be provided")
        if self.agent_class is not None and self.module_path is not None:
            raise ValueError("Cannot specify both 'agent_class' and 'module_path'")
        return self


class AgentConfig(BaseModel):
    """Agent configuration."""

    agent_name: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Agent type identifier")
    api_config: ChatCompletionsConfig | A2AConfig | CustomHTTPConfig | None = Field(
        None,
        description="API configuration for chat_completions, a2a, or custom_http agents",
    )
    custom_config: CustomConfig | None = Field(
        None, description="Configuration for custom agents"
    )

    @model_validator(mode="before")
    @classmethod
    def parse_config(cls, data: object) -> object:
        """Parse config based on top-level agent_type."""
        if isinstance(data, dict):
            agent_type = data.get("agent_type")

            if agent_type == AgentType.CUSTOM.value:
                config_data = data.get("custom_config")
                if not config_data:
                    raise ValueError(
                        "Custom agent requires 'custom_config' with 'agent_class' or 'module_path'"
                    )
                # Allow pre-constructed CustomConfig (needed for agent_class
                # which can't round-trip through a dict).
                if not isinstance(config_data, CustomConfig):
                    data["custom_config"] = CustomConfig(**config_data)
            elif agent_type == AgentType.CHAT_COMPLETIONS.value:
                config_data = data.get("api_config")
                if not config_data:
                    raise ValueError("chat_completions agent requires 'api_config'")
                if not isinstance(config_data, ChatCompletionsConfig):
                    data["api_config"] = ChatCompletionsConfig(**config_data)
            elif agent_type == AgentType.A2A.value:
                config_data = data.get("api_config")
                if not config_data:
                    raise ValueError("a2a agent requires 'api_config'")
                if not isinstance(config_data, A2AConfig):
                    data["api_config"] = A2AConfig(**config_data)
            elif agent_type == AgentType.CUSTOM_HTTP.value:
                config_data = data.get("api_config")
                if not config_data:
                    raise ValueError("custom_http agent requires 'api_config'")
                if not isinstance(config_data, CustomHTTPConfig):
                    data["api_config"] = CustomHTTPConfig(**config_data)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")
        else:
            raise ValueError("Agent configuration must be a dictionary")

        return data

    @model_validator(mode="after")
    def validate_type_config(self) -> Self:
        """Enforce that each agent type has the required config section.

        The ``mode='before'`` validator already rejects missing fields from
        raw dicts, but this ``mode='after'`` guard catches programmatic
        construction (e.g. ``AgentConfig(agent_type="a2a", agent_name=...)``)
        and provides defense-in-depth.
        """
        if (
            self.agent_type
            in (
                AgentType.CHAT_COMPLETIONS.value,
                AgentType.A2A.value,
                AgentType.CUSTOM_HTTP.value,
            )
            and self.api_config is None
        ):
            raise ValueError(f"'{self.agent_type}' agent requires 'api_config'")
        if self.agent_type == AgentType.CUSTOM.value and self.custom_config is None:
            raise ValueError("'custom' agent requires 'custom_config'")
        return self

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
