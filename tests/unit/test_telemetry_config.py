# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from arksim.telemetry.config import TelemetryConfig


class TestTelemetryConfigDefaults:
    def test_defaults(self) -> None:
        cfg = TelemetryConfig()
        assert cfg.enabled is False
        assert cfg.service_name == "arksim"
        assert cfg.endpoint == "http://localhost:4317"
        assert cfg.protocol == "grpc"
        assert cfg.headers == {}
        assert cfg.insecure is True

    def test_custom_values(self) -> None:
        cfg = TelemetryConfig(
            enabled=True,
            service_name="my-service",
            endpoint="http://collector:4318",
            protocol="http",
            headers={"x-api-key": "secret"},
            insecure=False,
        )
        assert cfg.enabled is True
        assert cfg.service_name == "my-service"
        assert cfg.protocol == "http"
        assert cfg.headers == {"x-api-key": "secret"}

    def test_invalid_protocol_rejected(self) -> None:
        with pytest.raises(ValueError):
            TelemetryConfig(protocol="websocket")  # type: ignore[arg-type]


class TestTelemetryConfigInSimulationInput:
    def test_simulation_input_accepts_telemetry(self) -> None:
        from arksim.simulation_engine.entities import SimulationInput

        data = {
            "agent_config_file_path": "agent.json",
            "scenario_file_path": "scenarios.json",
            "telemetry": {"enabled": True, "endpoint": "http://jaeger:4317"},
        }
        sim_input = SimulationInput.model_validate(data)
        assert sim_input.telemetry is not None
        assert sim_input.telemetry.enabled is True
        assert sim_input.telemetry.endpoint == "http://jaeger:4317"

    def test_simulation_input_telemetry_none_by_default(self) -> None:
        from arksim.simulation_engine.entities import SimulationInput

        data = {
            "agent_config_file_path": "agent.json",
            "scenario_file_path": "scenarios.json",
        }
        sim_input = SimulationInput.model_validate(data)
        assert sim_input.telemetry is None


class TestTelemetryConfigInEvaluationInput:
    def test_evaluation_input_accepts_telemetry(self) -> None:
        from arksim.evaluator.entities import EvaluationInput

        data = {
            "telemetry": {"enabled": True, "protocol": "http"},
        }
        eval_input = EvaluationInput.model_validate(data)
        assert eval_input.telemetry is not None
        assert eval_input.telemetry.protocol == "http"

    def test_evaluation_input_telemetry_none_by_default(self) -> None:
        from arksim.evaluator.entities import EvaluationInput

        eval_input = EvaluationInput.model_validate({})
        assert eval_input.telemetry is None
