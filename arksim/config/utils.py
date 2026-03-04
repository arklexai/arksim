# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def resolve_env_vars(headers: dict[str, str]) -> dict[str, str]:
    """Resolve ${ENV_VAR} patterns in header values with actual env values."""
    env_var_pattern = re.compile(r"\$\{([^}]+)\}")
    resolved_headers = {}

    for key, value in headers.items():

        def replace_env_var(match: re.Match[str]) -> str:
            env_var_name = match.group(1)
            if not os.getenv(env_var_name, ""):
                logger.warning(
                    f"Warning: Environment variable {env_var_name} configured in the agent configuration file is not set."
                )
            # For Azure Agent, the access token is generated at the runtime.
            if env_var_name == "AZURE_ACCESS_TOKEN" and not os.getenv(env_var_name, ""):
                from arksim.llms.utils import (
                    check_azure_env_vars,
                    get_azure_token_provider,
                )

                check_azure_env_vars()
                azure_token = get_azure_token_provider(
                    client_id=os.getenv("AZURE_CLIENT_ID")
                )
                return azure_token()
            return os.getenv(env_var_name, "")

        resolved_value = env_var_pattern.sub(replace_env_var, value)
        resolved_headers[key] = resolved_value

    return resolved_headers


def _resolve_config_relative_path(
    path: str,
    config_dir: str,
    cli_overrides: set,
    attr_name: str,
) -> str | None:
    """Return config-relative path for config-sourced paths, None for CLI-sourced.

    Paths from config.yaml are resolved relative to the config file's directory.
    Paths provided via CLI (in cli_overrides) are left as-is (cwd-relative).
    Absolute paths pass through unchanged.
    """
    if attr_name in cli_overrides:
        return None
    return os.path.normpath(os.path.join(config_dir, path))
