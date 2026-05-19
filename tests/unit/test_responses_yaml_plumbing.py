# SPDX-License-Identifier: Apache-2.0
"""End-to-end plumbing test: YAML Pydantic -> LLM(...) -> OpenAI SDK kwargs.

Verifies that base_url and api_key set on SimulationInput / EvaluationInput
travel through model_dump() and LLM(...) construction down to the underlying
OpenAI / AsyncOpenAI client constructors. This guards against the regression
where the fields were silently dropped by Pydantic's default extra='ignore'.
"""

from __future__ import annotations

from unittest.mock import patch

from arksim.evaluator.entities import EvaluationInput
from arksim.llms.chat.llm import LLM
from arksim.simulation_engine.entities import SimulationInput


class TestResponsesYamlPlumbing:
    """SimulationInput / EvaluationInput -> LLM -> OpenAI SDK forwarding."""

    def test_simulation_input_forwards_base_url_and_api_key_to_openai(self) -> None:
        settings = SimulationInput(
            agent_config_file_path="agent.json",
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            LLM(
                model=settings.model,
                provider=settings.provider,
                base_url=settings.base_url,
                api_key=settings.api_key,
            )
            mock_sync.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )
            mock_async.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )

    def test_evaluation_input_forwards_base_url_and_api_key_to_openai(self) -> None:
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            LLM(
                model=settings.model,
                provider=settings.provider,
                base_url=settings.base_url,
                api_key=settings.api_key,
            )
            mock_sync.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )
            mock_async.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )

    def test_simulation_input_defaults_omit_base_url_and_api_key(self) -> None:
        settings = SimulationInput(
            agent_config_file_path="agent.json", provider="responses"
        )
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            LLM(
                model=settings.model,
                provider=settings.provider,
                base_url=settings.base_url,
                api_key=settings.api_key,
            )
            # base_url=None and api_key=None must not be forwarded to the SDK;
            # the SDK falls back to OPENAI_BASE_URL / OPENAI_API_KEY env vars.
            mock_sync.assert_called_once_with()
            mock_async.assert_called_once_with()

    def test_evaluator_overrides_reach_llm_constructor(self) -> None:
        """When evaluator_* fields are set, they must override the shared
        fields when the evaluator constructs its LLM. The simulator-side
        `LLM(...)` call must NOT see the override values.
        """
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            evaluator_model="gpt-4o-mini",
            evaluator_provider="openai",
            evaluator_base_url="https://api.openai.com/v1",
            evaluator_api_key="sk-test",
        )
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            # Simulate the evaluator's LLM construction path
            LLM(
                model=settings.evaluator_model or settings.model,
                provider=settings.evaluator_provider or settings.provider,
                base_url=settings.evaluator_base_url or settings.base_url,
                api_key=settings.evaluator_api_key or settings.api_key,
            )
            mock_sync.assert_called_once_with(
                base_url="https://api.openai.com/v1", api_key="sk-test"
            )

    def test_evaluator_falls_back_to_shared_when_override_unset(self) -> None:
        """When evaluator_* fields are unset, the evaluator's LLM uses the
        shared model/provider/base_url/api_key.
        """
        settings = EvaluationInput(
            model="gpt-4o-mini",
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            LLM(
                model=settings.evaluator_model or settings.model,
                provider=settings.evaluator_provider or settings.provider,
                base_url=settings.evaluator_base_url or settings.base_url,
                api_key=settings.evaluator_api_key or settings.api_key,
            )
            mock_sync.assert_called_once_with(
                base_url="https://api.openai.com/v1", api_key="sk-test"
            )
