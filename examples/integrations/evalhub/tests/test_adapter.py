# SPDX-License-Identifier: Apache-2.0
"""Adapter wiring tests. The LLM-driven sim/eval seam is stubbed, so these run
without API keys or a target endpoint."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import arksim_evalhub.adapter as adapter_mod
import pytest
from arksim_evalhub.adapter import ArksimAdapter, resolve_target_api_key
from arksim_evalhub.mapping import ArksimJobParameters

from arksim.evaluator.entities import ConversationEvaluation, Evaluation
from arksim.simulation_engine.entities import Simulation

_SCENARIO = {
    "scenario_id": "s",
    "user_id": "u",
    "goal": "g",
    "agent_context": "c",
    "user_profile": "p",
}

JOB_SPEC = {
    "id": "job-1",
    "provider_id": "arksim",
    "benchmark_id": "bench",
    "benchmark_index": 0,
    "callback_url": "http://localhost:8080",
    "model": {"url": "https://api/x", "name": "gpt-4.1-mini"},
    "parameters": {
        "target_api_key_env": "FAKE_KEY",
        "scenarios": [
            {**_SCENARIO, "scenario_id": "s1"},
            {**_SCENARIO, "scenario_id": "s2"},
        ],
    },
}


class _RecordingCallbacks:
    def __init__(self) -> None:
        self.statuses: list[object] = []
        self.results: object | None = None

    def report_status(self, update: object) -> None:
        self.statuses.append(update)

    def report_results(self, results: object) -> None:
        self.results = results

    def create_oci_artifact(self, spec: object) -> object:
        raise NotImplementedError


def _write_job_spec(tmp_path: Path, spec: dict) -> Path:
    path = (
        tmp_path
        / spec["id"]
        / str(spec["benchmark_index"])
        / spec["provider_id"]
        / spec["benchmark_id"]
        / "meta"
        / "job.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec))
    return path


def _fake_sim_eval(
    scores: list[float], *, write_artifacts: bool = True
) -> Callable[..., Awaitable[tuple[Simulation, Evaluation]]]:
    async def _fake(agent_config, params, scenarios, output_dir):  # noqa: ANN001, ANN202
        output_dir.mkdir(parents=True, exist_ok=True)
        if write_artifacts:
            (output_dir / "simulation.json").write_text("{}")
            (output_dir / "evaluation.json").write_text("{}")
            (output_dir / "final_report.html").write_text("<html></html>")
        convos = [
            ConversationEvaluation(
                conversation_id=f"c{i}",
                goal_completion_score=s,
                goal_completion_reason="",
                turn_success_ratio=s,
                overall_agent_score=s,
                evaluation_status="Done",
                turn_scores=[],
            )
            for i, s in enumerate(scores)
        ]
        evaluation = Evaluation(
            schema_version="v1.1",
            generated_at="2026-01-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="e",
            simulation_id="s",
            conversations=convos,
            unique_errors=[],
        )
        simulation = Simulation(
            schema_version="v1", simulator_version="v1", conversations=[]
        )
        return simulation, evaluation

    return _fake


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArksimAdapter:
    spec_path = _write_job_spec(tmp_path, JOB_SPEC)
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("EVALHUB_MODE", "local")
    # JOB_SPEC names FAKE_KEY as the target api-key env var; provide it so the
    # credential resolution succeeds for the wiring/main tests.
    monkeypatch.setenv("FAKE_KEY", "test-key")
    return ArksimAdapter()


def test_run_benchmark_job_wiring(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adapter_mod, "simulate_and_evaluate", _fake_sim_eval([0.4, 0.6])
    )
    callbacks = _RecordingCallbacks()

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    assert results.model_name == "gpt-4.1-mini"
    assert results.num_examples_evaluated == 2
    assert results.overall_score == pytest.approx(0.5)
    metric_names = {r.metric_name for r in results.results}
    assert {"num_conversations", "overall_agent_score"} <= metric_names
    assert len(adapter.mlflow_artifacts) == 3
    assert callbacks.statuses, "expected progress status updates"


def test_missing_artifacts_are_skipped_not_fatal(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adapter_mod,
        "simulate_and_evaluate",
        _fake_sim_eval([0.5], write_artifacts=False),
    )
    results = adapter.run_benchmark_job(adapter.job_spec, _RecordingCallbacks())
    assert results.num_examples_evaluated == 1
    assert adapter.mlflow_artifacts == []


def test_num_examples_caps_scenarios(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}

    async def _fake(agent_config, params, scenarios, output_dir):  # noqa: ANN001, ANN202
        captured["n"] = len(scenarios.scenarios)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation = Evaluation(
            schema_version="v1.1",
            generated_at="2026-01-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="e",
            simulation_id="s",
            conversations=[],
            unique_errors=[],
        )
        return (
            Simulation(schema_version="v1", simulator_version="v1", conversations=[]),
            evaluation,
        )

    monkeypatch.setattr(adapter_mod, "simulate_and_evaluate", _fake)
    spec = adapter.job_spec
    spec.num_examples = 1

    adapter.run_benchmark_job(spec, _RecordingCallbacks())
    assert captured["n"] == 1


def _params_with_key_env() -> ArksimJobParameters:
    return ArksimJobParameters.model_validate(
        {"target_api_key_env": "FAKE_KEY", "scenarios": [_SCENARIO]}
    )


def test_resolve_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_mod, "read_model_auth_key", lambda key: None)
    monkeypatch.setenv("FAKE_KEY", "from-env")
    assert resolve_target_api_key(_params_with_key_env()) == "from-env"


def test_resolve_api_key_prefers_mounted_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_mod, "read_model_auth_key", lambda key: "mounted")
    monkeypatch.setenv("FAKE_KEY", "from-env")
    assert resolve_target_api_key(_params_with_key_env()) == "mounted"


class _RecordingMlflow:
    def __init__(self) -> None:
        self.saved: list[object] = []

    def save(self, results: object, job_spec: object, artifacts: object = None) -> str:
        self.saved.append((results, artifacts))
        return "run-123"


def _stub_callbacks(monkeypatch: pytest.MonkeyPatch) -> _RecordingCallbacks:
    """Make main() use a recording callbacks double instead of the real SDK."""
    cb = _RecordingCallbacks()
    cb.mlflow = _RecordingMlflow()  # type: ignore[attr-defined]
    monkeypatch.setattr(
        adapter_mod.DefaultCallbacks, "from_adapter", staticmethod(lambda a: cb)
    )
    return cb


def test_main_skips_mlflow_without_uri(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_mod, "simulate_and_evaluate", _fake_sim_eval([0.7]))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cb = _stub_callbacks(monkeypatch)

    adapter_mod.main()

    assert cb.results is not None
    assert cb.results.num_examples_evaluated == 1
    assert cb.mlflow.saved == []  # type: ignore[attr-defined]


def test_main_logs_mlflow_when_uri_set(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_mod, "simulate_and_evaluate", _fake_sim_eval([0.7]))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    cb = _stub_callbacks(monkeypatch)

    adapter_mod.main()

    assert len(cb.mlflow.saved) == 1  # type: ignore[attr-defined]
    assert cb.results.mlflow_run_id == "run-123"


def test_main_success_with_real_callbacks_no_sidecar(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real DefaultCallbacks must degrade gracefully (log, not raise) when
    no sidecar is reachable, so a local run completes instead of crashing."""
    monkeypatch.setattr(adapter_mod, "simulate_and_evaluate", _fake_sim_eval([0.7]))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    adapter_mod.main()  # must not raise


def test_main_reports_failed_and_reraises_on_error(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(*args: object, **kwargs: object) -> tuple[object, object]:
        raise RuntimeError("sim exploded")

    monkeypatch.setattr(adapter_mod, "simulate_and_evaluate", _boom)
    cb = _stub_callbacks(monkeypatch)

    with pytest.raises(RuntimeError, match="sim exploded"):
        adapter_mod.main()

    assert any(
        getattr(s, "status", None) == adapter_mod.JobStatus.FAILED for s in cb.statuses
    )


def test_resolve_api_key_raises_when_named_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_mod, "read_model_auth_key", lambda key: None)
    monkeypatch.delenv("FAKE_KEY", raising=False)
    with pytest.raises(ValueError, match="target_api_key_env"):
        resolve_target_api_key(_params_with_key_env())


def test_resolve_api_key_none_when_no_secret_and_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_mod, "read_model_auth_key", lambda key: None)
    params = ArksimJobParameters.model_validate({"scenarios": [_SCENARIO]})
    assert resolve_target_api_key(params) is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com/v1/chat", "https://api.example.com/v1/chat"),
        ("https://user:secret@api.example.com/v1", "https://api.example.com/v1"),
        ("https://api.example.com/v1?api_key=SECRET", "https://api.example.com/v1"),
        ("https://host:8443/v1", "https://host:8443/v1"),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert adapter_mod._redact_url(url) == expected


def test_redact_url_schemeless_does_not_leak() -> None:
    assert "secret" not in adapter_mod._redact_url("user:secret@host.com/v1")


def test_scrub_redacts_secrets() -> None:
    scrubbed = adapter_mod._scrub(
        "fail: Bearer sk-abc123DEF at https://u:p@host/x key sk-ZZZ12345abc"
    )
    assert "sk-abc123DEF" not in scrubbed
    assert "sk-ZZZ12345abc" not in scrubbed
    assert "u:p@" not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_select_scenarios_rejects_nonpositive() -> None:
    scenarios = ArksimJobParameters.model_validate(
        {"scenarios": [_SCENARIO, {**_SCENARIO, "scenario_id": "s2"}]}
    ).scenarios
    for bad in (0, -1):
        with pytest.raises(ValueError, match="num_examples must be positive"):
            ArksimAdapter._select_scenarios(scenarios, bad)


def test_run_benchmark_job_propagates_sim_failure(
    adapter: ArksimAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(*args: object, **kwargs: object) -> tuple[object, object]:
        raise RuntimeError("sim exploded")

    monkeypatch.setattr(adapter_mod, "simulate_and_evaluate", _boom)
    with pytest.raises(RuntimeError, match="sim exploded"):
        adapter.run_benchmark_job(adapter.job_spec, _RecordingCallbacks())
