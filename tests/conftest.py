"""Pytest configuration and fixtures for simulator tests."""

import os
import tempfile
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_dir() -> dict:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def valid_agent_config_a2a_basic() -> dict:
    """Valid agent config for a2a type (basic version for general tests)."""
    return {
        "agent_type": "a2a",
        "agent_name": "test-agent",

        "api_config": {
            "endpoint": "https://api.example.com/agent",
        },
    }


@pytest.fixture
def valid_agent_config_a2a() -> dict:
    """Valid agent config for a2a type."""
    return {
        "agent_type": "a2a",
        "agent_name": "test-a2a-agent",

        "api_config": {
            "endpoint": "https://a2a.example.com/agent",
            "headers": {"Authorization": "Bearer ${API_TOKEN}"},
        },
    }


@pytest.fixture
def valid_agent_config_chat_completions_new() -> dict:
    """Valid agent config for chat_completions type (new format)."""
    return {
        "agent_type": "chat_completions",
        "agent_name": "test-chat-agent-new",

        "api_config": {
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "headers": {
                "Authorization": "Bearer ${OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            "body": {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."}
                ],
            },
        },
    }


@pytest.fixture
def mock_env_vars() -> dict:
    """Mock environment variables for testing."""
    env_vars = {
        "API_TOKEN": "test-token-123",
        "OPENAI_API_KEY": "sk-test-key",
        "CUSTOM_VAR": "custom-value",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars
