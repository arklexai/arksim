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

    def test_evaluator_kwargs_inherit_when_no_overrides(self) -> None:
        """When no evaluator_* fields are set, evaluator inherits all
        simulator-side LLM keys.
        """
        settings = EvaluationInput(
            model="gpt-4o-mini",
            provider="openai",
            base_url=None,
            api_key="sk-shared",
        )
        kwargs = settings.evaluator_llm_kwargs()
        assert kwargs == {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "base_url": None,
            "api_key": "sk-shared",
        }

    def test_evaluator_kwargs_model_only_override_keeps_shared_endpoint(
        self,
    ) -> None:
        """evaluator_model alone overrides only the model; other shared
        keys still apply because the endpoint did not change.
        """
        settings = EvaluationInput(
            model="gpt-4o-mini",
            provider="openai",
            api_key="sk-shared",
            evaluator_model="gpt-4o",
        )
        kwargs = settings.evaluator_llm_kwargs()
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["provider"] == "openai"
        assert kwargs["api_key"] == "sk-shared"

    def test_evaluator_kwargs_provider_split_does_not_inherit_credentials(
        self,
    ) -> None:
        """When evaluator_provider differs from provider, the shared
        api_key and base_url MUST NOT be forwarded. This is the bug fix
        for the cross-endpoint credential leak.
        """
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            evaluator_provider="openai",
            evaluator_model="gpt-4o-mini",
        )
        kwargs = settings.evaluator_llm_kwargs()
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["provider"] == "openai"
        assert kwargs["base_url"] is None, (
            "Shared base_url must not leak into evaluator when provider differs"
        )
        assert kwargs["api_key"] is None, (
            "Shared api_key must not leak into evaluator when provider differs"
        )

    def test_evaluator_kwargs_base_url_split_does_not_inherit_credentials(
        self,
    ) -> None:
        """When evaluator_base_url differs from base_url (endpoint split),
        api_key must NOT inherit from shared. Catches the credential-leak
        scenario where a live OpenAI key would otherwise be sent to a
        self-hosted endpoint.
        """
        settings = EvaluationInput(
            model="gpt-4o-mini",
            provider="openai",
            api_key="sk-LIVE-OPENAI-KEY",
            evaluator_base_url="http://localhost:11434/v1",
        )
        kwargs = settings.evaluator_llm_kwargs()
        assert kwargs["api_key"] is None, (
            "Live api_key must not be forwarded to a different base_url"
        )
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_evaluator_kwargs_explicit_overrides_used_verbatim(self) -> None:
        """When evaluator_* fields are fully specified, they pass through
        verbatim and shared keys are ignored.
        """
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            evaluator_model="gpt-4o-mini",
            evaluator_provider="openai",
            evaluator_base_url="https://api.openai.com/v1",
            evaluator_api_key="sk-eval-explicit",
        )
        kwargs = settings.evaluator_llm_kwargs()
        assert kwargs == {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-eval-explicit",
        }

    def test_evaluator_split_reaches_openai_sdk_without_shared_credentials(
        self,
    ) -> None:
        """End-to-end: simulator on Ollama, evaluator on OpenAI. The OpenAI
        SDK constructor must not see Ollama's base_url or api_key.
        """
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            evaluator_provider="openai",
            evaluator_model="gpt-4o-mini",
        )

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            LLM(**settings.evaluator_llm_kwargs())
            # Critical: NO base_url, NO api_key kwargs. SDK must fall back
            # to OPENAI_BASE_URL / OPENAI_API_KEY env vars.
            mock_sync.assert_called_once_with()
            mock_async.assert_called_once_with()
