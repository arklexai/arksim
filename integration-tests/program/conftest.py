# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for arksim OSS integration tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

OPENAI_MODEL = os.environ.get("TEST_OPENAI_MODEL", "gpt-5.1")
ANTHROPIC_MODEL = os.environ.get("TEST_ANTHROPIC_MODEL", "claude-sonnet-4-6")
GOOGLE_MODEL = os.environ.get("TEST_GOOGLE_MODEL", "gemini-3-flash-preview")


def _check_env(var: str) -> None:
    """Raise immediately if an env var is missing."""
    if not os.environ.get(var):
        pytest.fail(f"{var} environment variable is required but not set")


@pytest.fixture(autouse=False)
def _require_openai_key() -> None:
    _check_env("OPENAI_API_KEY")


@pytest.fixture(autouse=False)
def _require_anthropic_key() -> None:
    _check_env("ANTHROPIC_API_KEY")


@pytest.fixture(autouse=False)
def _require_google_key() -> None:
    _check_env("GOOGLE_API_KEY")


requires_openai = pytest.mark.usefixtures("_require_openai_key")
requires_anthropic = pytest.mark.usefixtures("_require_anthropic_key")
requires_google = pytest.mark.usefixtures("_require_google_key")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> str:
    """Create output subdirectories for simulation and evaluation."""
    (tmp_path / "simulation").mkdir()
    (tmp_path / "evaluation").mkdir()
    return str(tmp_path)


@pytest.fixture
def minimal_scenarios(tmp_path: Path) -> str:
    """Write a minimal scenarios.json with 3 scenarios."""
    data = {
        "schema_version": "v1",
        "scenarios": [
            {
                "scenario_id": "test_general_inquiry",
                "user_id": "user_001",
                "goal": (
                    "You want to learn about the different "
                    "types of home insurance coverage available "
                    "and which one would be best for a new "
                    "homeowner."
                ),
                "agent_context": (
                    "XYZ Insurance provides home, auto, and "
                    "travel insurance in Canada. We offer "
                    "comprehensive and basic coverage plans."
                ),
                "user_profile": (
                    "You are Alex, a 28-year-old first-time "
                    "homeowner. You are friendly and curious "
                    "but have no prior insurance knowledge."
                ),
                "knowledge": [
                    {
                        "content": (
                            "Home insurance covers damage to "
                            "your property from fire, theft, "
                            "and natural disasters. Basic "
                            "coverage starts at $500/year."
                        )
                    }
                ],
                "origin": {
                    "goal_raw": "Learn about home insurance",
                    "target_agent_capability": ("explain insurance products"),
                },
            },
            {
                "scenario_id": "test_claim_filing",
                "user_id": "user_002",
                "goal": (
                    "You had a pipe burst in your basement "
                    "last night and need to file an insurance "
                    "claim. You want step-by-step guidance on "
                    "the claims process."
                ),
                "agent_context": (
                    "XYZ Insurance provides home, auto, and "
                    "travel insurance in Canada. We offer "
                    "comprehensive and basic coverage plans."
                ),
                "user_profile": (
                    "You are Maria, a 45-year-old homeowner "
                    "who is stressed and worried about the "
                    "water damage. You want clear, direct "
                    "answers."
                ),
                "knowledge": [
                    {
                        "content": (
                            "To file a claim: 1) Turn off the "
                            "main water valve. 2) Document all "
                            "damage with photos. 3) Call our "
                            "claims line at 1-800-555-0123. "
                            "4) A claims adjuster will contact "
                            "you within 24 hours."
                        )
                    }
                ],
                "origin": {
                    "goal_raw": "File an insurance claim",
                    "target_agent_capability": ("guide claims process"),
                },
            },
            {
                "scenario_id": "test_policy_renewal",
                "user_id": "user_003",
                "goal": (
                    "Your home insurance policy is up for "
                    "renewal and the premium increased by 15%. "
                    "You want to understand why and negotiate "
                    "a better rate."
                ),
                "agent_context": (
                    "XYZ Insurance provides home, auto, and "
                    "travel insurance in Canada. We offer "
                    "comprehensive and basic coverage plans."
                ),
                "user_profile": (
                    "You are James, a 55-year-old long-time "
                    "customer who is budget-conscious and "
                    "direct. You have been a loyal customer "
                    "for 10 years with no claims."
                ),
                "knowledge": [
                    {
                        "content": (
                            "Premium increases can be caused "
                            "by: increased rebuilding costs, "
                            "regional risk changes, or "
                            "industry-wide adjustments. "
                            "Customers can lower premiums by "
                            "bundling policies, increasing "
                            "deductibles, or installing "
                            "security systems."
                        )
                    }
                ],
                "origin": {
                    "goal_raw": "Negotiate policy renewal",
                    "target_agent_capability": ("handle policy renewal discussions"),
                },
            },
        ],
    }
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


@pytest.fixture
def agent_config_openai() -> dict:
    """Return an agent config dict for chat_completions
    using OpenAI.
    """
    return {
        "agent_type": "chat_completions",
        "agent_name": "test-insurance-agent",
        "api_config": {
            "endpoint": ("https://api.openai.com/v1/chat/completions"),
            "headers": {
                "Content-Type": "application/json",
                "Authorization": (f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"),
            },
            "body": {
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a customer service chatbot "
                            "for XYZ Insurance. Answer questions "
                            "about insurance products helpfully. "
                            "Never exceed 80 words."
                        ),
                    }
                ],
            },
        },
    }
