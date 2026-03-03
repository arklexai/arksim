# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM provider abstraction."""

import os
from unittest.mock import patch

import pytest

from arksim.llms.chat import LLM
from arksim.llms.utils import check_azure_env_vars


class TestLLM:
    """Tests for LLM factory class."""

    def test_requires_model_name(self) -> None:
        """Test LLM requires a model name."""
        with pytest.raises(ValueError, match="Model name is required"):
            LLM(model=None)

    def test_requires_string_model_name(self) -> None:
        """Test LLM requires model name to be a string."""
        with pytest.raises(ValueError, match="Model name is required"):
            LLM(model=123)

    def test_empty_model_name_raises(self) -> None:
        """Test empty model name raises error."""
        with pytest.raises(ValueError, match="Model name is required"):
            LLM(model="")

    def test_unsupported_model_raises(self) -> None:
        """Test unsupported model raises error."""
        with pytest.raises(ValueError, match="is not supported"):
            LLM(model="unknown-model")

    def test_unsupported_provider_raises(self) -> None:
        """Test unsupported provider raises error."""
        with pytest.raises(ValueError, match="Provider .* is not supported"):
            LLM(model="gpt-4", provider="unsupported")


class TestCheckAzureEnvVars:
    """Tests for check_azure_env_vars function."""

    def test_raises_when_client_id_missing(self) -> None:
        """Test raises error when AZURE_CLIENT_ID is missing."""
        with (
            patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_API_VERSION": "2024-02-01",
                    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4",
                },
                clear=True,
            ),
            pytest.raises(ValueError, match="AZURE_CLIENT_ID"),
        ):
            check_azure_env_vars()

    def test_raises_when_api_version_missing(self) -> None:
        """Test raises error when AZURE_OPENAI_API_VERSION is missing."""
        with (
            patch.dict(
                os.environ,
                {
                    "AZURE_CLIENT_ID": "client-123",
                    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4",
                },
                clear=True,
            ),
            pytest.raises(ValueError, match="AZURE_OPENAI_API_VERSION"),
        ):
            check_azure_env_vars()

    def test_raises_when_deployment_name_missing(self) -> None:
        """Test raises error when AZURE_OPENAI_DEPLOYMENT_NAME is missing."""
        with (
            patch.dict(
                os.environ,
                {
                    "AZURE_CLIENT_ID": "client-123",
                    "AZURE_OPENAI_API_VERSION": "2024-02-01",
                },
                clear=True,
            ),
            pytest.raises(ValueError, match="AZURE_OPENAI_DEPLOYMENT_NAME"),
        ):
            check_azure_env_vars()

    def test_passes_when_all_vars_set(self) -> None:
        """Test passes when all required env vars are set."""
        with patch.dict(
            os.environ,
            {
                "AZURE_CLIENT_ID": "client-123",
                "AZURE_OPENAI_API_VERSION": "2024-02-01",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4",
            },
            clear=True,
        ):
            # Should not raise
            check_azure_env_vars()
