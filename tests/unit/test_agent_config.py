"""Tests for agent configuration models."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from arksim.config import (
    A2AConfig,
    AgentConfig,
    AgentType,
    ChatCompletionsConfig,
)
from arksim.config.utils import resolve_env_vars


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_resolves_single_env_var(self, mock_env_vars: dict) -> None:
        """Test resolving a single environment variable."""
        headers = {"Authorization": "Bearer ${API_TOKEN}"}
        resolved = resolve_env_vars(headers)

        assert resolved["Authorization"] == "Bearer test-token-123"

    def test_resolves_multiple_env_vars(self, mock_env_vars: dict) -> None:
        """Test resolving multiple environment variables."""
        headers = {
            "Authorization": "Bearer ${API_TOKEN}",
            "X-Custom": "${CUSTOM_VAR}",
        }
        resolved = resolve_env_vars(headers)

        assert resolved["Authorization"] == "Bearer test-token-123"
        assert resolved["X-Custom"] == "custom-value"

    def test_resolves_multiple_vars_in_single_value(self, mock_env_vars: dict) -> None:
        """Test resolving multiple env vars in a single header value."""
        headers = {"Combined": "${API_TOKEN}-${CUSTOM_VAR}"}
        resolved = resolve_env_vars(headers)

        assert resolved["Combined"] == "test-token-123-custom-value"

    def test_missing_env_var_returns_empty(self) -> None:
        """Test missing env var returns empty string."""
        with patch.dict(os.environ, {}, clear=True):
            headers = {"Auth": "${NONEXISTENT_VAR}"}
            resolved = resolve_env_vars(headers)

            assert resolved["Auth"] == ""

    def test_no_env_vars_returns_unchanged(self) -> None:
        """Test headers without env vars are returned unchanged."""
        headers = {"Content-Type": "application/json"}
        resolved = resolve_env_vars(headers)

        assert resolved["Content-Type"] == "application/json"

    def test_empty_headers(self) -> None:
        """Test empty headers dict."""
        resolved = resolve_env_vars({})
        assert resolved == {}


class TestA2AConfig:
    """Tests for A2AConfig model."""

    def test_valid_config(self) -> None:
        """Test creating valid A2AConfig."""
        config = A2AConfig(
            endpoint="https://a2a.example.com",
            headers={"Authorization": "Bearer token"},
        )

        assert config.endpoint == "https://a2a.example.com"
        assert config.headers == {"Authorization": "Bearer token"}

    def test_config_without_headers(self) -> None:
        """Test A2AConfig with no headers."""
        config = A2AConfig(endpoint="https://a2a.example.com")

        assert config.headers is None

    def test_get_headers_returns_none_when_no_headers(self) -> None:
        """Test get_headers returns None when headers not set."""
        config = A2AConfig(endpoint="https://a2a.example.com")

        assert config.get_headers() is None

    def test_get_headers_resolves_env_vars(self, mock_env_vars: dict) -> None:
        """Test get_headers resolves environment variables."""
        config = A2AConfig(
            endpoint="https://a2a.example.com",
            headers={"Authorization": "Bearer ${API_TOKEN}"},
        )

        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer test-token-123"


class TestChatCompletionsConfig:
    """Tests for ChatCompletionsConfig model."""

    def test_valid_new_format(
        self, valid_agent_config_chat_completions_new: dict
    ) -> None:
        """Test valid new format config."""
        config_data = valid_agent_config_chat_completions_new["api_config"]
        config = ChatCompletionsConfig(**config_data)

        assert config.endpoint == "https://api.openai.com/v1/chat/completions"
        assert config.headers is not None
        assert config.body is not None

    def test_new_format_requires_body(self) -> None:
        """Test new format requires body field."""
        with pytest.raises(ValidationError, match="requires 'body' field"):
            ChatCompletionsConfig(
                endpoint="https://api.example.com",
                headers={"Content-Type": "application/json"},
            )

    def test_get_endpoint_new_format(
        self, valid_agent_config_chat_completions_new: dict
    ) -> None:
        """Test get_endpoint with new format."""
        config_data = valid_agent_config_chat_completions_new["api_config"]
        config = ChatCompletionsConfig(**config_data)

        endpoint = config.get_endpoint()
        assert endpoint == "https://api.openai.com/v1/chat/completions"

    def test_get_headers_resolves_env_vars(
        self, mock_env_vars: dict, valid_agent_config_chat_completions_new: dict
    ) -> None:
        """Test get_headers resolves environment variables."""
        config_data = valid_agent_config_chat_completions_new["api_config"]
        config = ChatCompletionsConfig(**config_data)

        headers = config.get_headers()
        assert headers["Authorization"] == "Bearer sk-test-key"


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_valid_a2a_config(self, valid_agent_config_a2a: dict) -> None:
        """Test valid a2a agent config."""
        config = AgentConfig(**valid_agent_config_a2a)

        assert config.agent_type == AgentType.A2A.value
        assert isinstance(config.api_config, A2AConfig)

    def test_valid_chat_completions_config(
        self, valid_agent_config_chat_completions_new: dict
    ) -> None:
        """Test valid chat_completions agent config."""
        config = AgentConfig(**valid_agent_config_chat_completions_new)

        assert config.agent_type == AgentType.CHAT_COMPLETIONS.value
        assert isinstance(config.api_config, ChatCompletionsConfig)

    def test_unsupported_agent_type(self) -> None:
        """Test unsupported agent type raises error."""
        with pytest.raises(ValidationError, match="Unsupported agent type"):
            AgentConfig(
                agent_type="unsupported",
                agent_name="test",
                agent_capabilities=[],
                api_config={},
            )

    def test_requires_agent_name(self, valid_agent_config_a2a: dict) -> None:
        """Test agent_name is required."""
        del valid_agent_config_a2a["agent_name"]
        with pytest.raises(ValidationError):
            AgentConfig(**valid_agent_config_a2a)

    def test_parses_api_config_fallback(self) -> None:
        """Test parser falls back to api_config field."""
        config_data = {
            "agent_type": "a2a",
            "agent_name": "test",
            "agent_capabilities": [],
            "api_config": {
                "endpoint": "https://api.example.com/agent",
            },
        }
        config = AgentConfig(**config_data)

        assert isinstance(config.api_config, A2AConfig)
