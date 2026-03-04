# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.llms.utils.azure.check_azure_env_vars."""

import os
from unittest.mock import patch

import pytest

from arksim.llms.utils.azure import check_azure_env_vars


class TestCheckAzureEnvVars:
    def test_all_set(self) -> None:
        env = {
            "AZURE_CLIENT_ID": "id",
            "AZURE_OPENAI_API_VERSION": "v1",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "dep",
        }
        with patch.dict(os.environ, env):
            check_azure_env_vars()  # should not raise

    def test_missing_client_id(self) -> None:
        env = {
            "AZURE_OPENAI_API_VERSION": "v1",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "dep",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="AZURE_CLIENT_ID"),
        ):
            check_azure_env_vars()

    def test_missing_api_version(self) -> None:
        env = {
            "AZURE_CLIENT_ID": "id",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "dep",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(ValueError, match="AZURE_OPENAI_API_VERSION"),
        ):
            check_azure_env_vars()

    def test_all_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError):
            check_azure_env_vars()
