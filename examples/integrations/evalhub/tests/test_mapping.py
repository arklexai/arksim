# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure EvalHub <-> arksim mapping (no LLM, no credentials)."""

from __future__ import annotations

import pytest
from arksim_evalhub.mapping import (
    ArksimJobParameters,
    aggregate_metrics,
    build_agent_config,
    compute_overall_score,
)
from evalhub.adapter import ModelConfig
from pydantic import ValidationError

from arksim import A2AConfig, ChatCompletionsConfig
from arksim.evaluator.entities import ConversationEvaluation, Evaluation

_SCENARIO = {
    "scenario_id": "s1",
    "user_id": "u1",
    "goal": "g",
    "agent_context": "c",
    "user_profile": "p",
}


def _params(**overrides: object) -> ArksimJobParameters:
    data: dict[str, object] = {"scenarios": [_SCENARIO]}
    data.update(overrides)
    return ArksimJobParameters.model_validate(data)


def _convo(score: float) -> ConversationEvaluation:
    return ConversationEvaluation(
        conversation_id="c",
        goal_completion_score=score,
        goal_completion_reason="",
        turn_success_ratio=score,
        overall_agent_score=score,
        evaluation_status="Done",
        turn_scores=[],
    )


def _evaluation(scores: list[float]) -> Evaluation:
    return Evaluation(
        schema_version="v1.1",
        generated_at="2026-01-01T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="e1",
        simulation_id="sim1",
        conversations=[_convo(s) for s in scores],
        unique_errors=[],
    )


class TestArksimJobParameters:
    def test_wraps_bare_scenario_list(self) -> None:
        params = _params()
        assert params.scenarios.schema_version == "v1"
        assert len(params.scenarios.scenarios) == 1

    def test_accepts_full_scenarios_object(self) -> None:
        params = _params(scenarios={"schema_version": "v2", "scenarios": [_SCENARIO]})
        assert params.scenarios.schema_version == "v2"

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            _params(simulater_model="typo")

    def test_rejects_empty_scenarios(self) -> None:
        with pytest.raises(ValidationError, match="at least one scenario"):
            ArksimJobParameters.model_validate({"scenarios": []})

    def test_rejects_reserved_request_body_keys(self) -> None:
        with pytest.raises(ValidationError, match="reserved keys"):
            _params(request_body={"messages": "nope"})

    def test_defaults(self) -> None:
        params = _params()
        assert params.protocol == "chat_completions"
        assert params.num_conversations_per_scenario == 2

    def test_accepts_auto_num_workers(self) -> None:
        assert _params(num_workers="auto").num_workers == "auto"

    def test_rejects_junk_num_workers(self) -> None:
        with pytest.raises(ValidationError):
            _params(num_workers="fast")

    def test_rejects_nonpositive_num_workers(self) -> None:
        with pytest.raises(ValidationError):
            _params(num_workers=0)


class TestBuildAgentConfig:
    def test_chat_completions_maps_endpoint_and_bearer(self) -> None:
        model = ModelConfig(url="https://api/x", name="gpt-4.1-mini")
        cfg = build_agent_config(model, _params(), api_key="secret")
        assert cfg.agent_type == "chat_completions"
        assert isinstance(cfg.api_config, ChatCompletionsConfig)
        assert cfg.api_config.endpoint == "https://api/x"
        assert cfg.api_config.body["model"] == "gpt-4.1-mini"
        assert cfg.api_config.headers["Authorization"] == "Bearer secret"

    def test_chat_completions_without_api_key_has_no_auth(self) -> None:
        model = ModelConfig(url="https://api/x", name="m")
        cfg = build_agent_config(model, _params(), api_key=None)
        assert cfg.api_config.headers is None

    def test_a2a_protocol(self) -> None:
        model = ModelConfig(url="https://agent/a2a", name="m")
        cfg = build_agent_config(model, _params(protocol="a2a"), api_key="k")
        assert cfg.agent_type == "a2a"
        assert isinstance(cfg.api_config, A2AConfig)
        assert cfg.api_config.endpoint == "https://agent/a2a"
        assert cfg.api_config.headers["Authorization"] == "Bearer k"

    def test_request_body_merged(self) -> None:
        model = ModelConfig(url="u", name="m")
        cfg = build_agent_config(
            model, _params(request_body={"temperature": 0.0}), api_key=None
        )
        assert cfg.api_config.body["temperature"] == 0.0


class TestAggregation:
    def test_aggregate_metrics_means(self) -> None:
        results = aggregate_metrics(_evaluation([0.4, 0.6]))
        by_name = {r.metric_name: r.metric_value for r in results}
        assert by_name["num_conversations"] == 2
        assert by_name["overall_agent_score"] == pytest.approx(0.5)
        assert by_name["goal_completion_score"] == pytest.approx(0.5)
        assert by_name["turn_success_ratio"] == pytest.approx(0.5)

    def test_aggregate_metrics_empty(self) -> None:
        results = aggregate_metrics(_evaluation([]))
        assert len(results) == 1
        assert results[0].metric_name == "num_conversations"
        assert results[0].metric_value == 0

    def test_compute_overall_score(self) -> None:
        assert compute_overall_score(_evaluation([0.2, 0.8])) == pytest.approx(0.5)

    def test_compute_overall_score_empty(self) -> None:
        assert compute_overall_score(_evaluation([])) is None
