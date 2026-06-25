# SPDX-License-Identifier: Apache-2.0
"""EvalHub ``FrameworkAdapter`` that runs an arksim simulation + evaluation.

Lifecycle (driven by ``eval-hub-sdk``):

1. EvalHub mounts the JobSpec at ``$EVALHUB_JOB_SPEC_PATH`` (default
   ``/meta/job.json``) and the model secret at ``/var/run/secrets/model``.
2. :class:`ArksimAdapter` is constructed; the SDK loads settings + job spec.
3. :meth:`ArksimAdapter.run_benchmark_job` builds an arksim ``AgentConfig`` from
   the target model, runs simulation then evaluation against it, and returns
   aggregated scalar metrics as ``JobResults``.
4. :func:`main` logs the transcripts + HTML report to MLflow (when configured)
   and reports the terminal result.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from evalhub.adapter import (
    DefaultCallbacks,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    read_model_auth_key,
)
from evalhub.adapter.mlflow import MlflowArtifact

from arksim import AgentConfig, Scenarios
from arksim.evaluator import EvaluationInput, run_evaluation

# Result envelopes are not re-exported at the arksim top level; import them from
# their defining modules. Used only for type hints here.
from arksim.evaluator.entities import Evaluation
from arksim.simulation_engine import SimulationInput, run_simulation
from arksim.simulation_engine.entities import Simulation
from arksim_evalhub.mapping import (
    ArksimJobParameters,
    aggregate_metrics,
    build_agent_config,
    compute_overall_score,
)

logger = logging.getLogger("arksim_evalhub")

# Artifact filenames. simulation.json is set by us (below); evaluation.json and
# final_report.html are arksim's own run_evaluation outputs.
_SIMULATION_FILE = "simulation.json"
_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (_SIMULATION_FILE, "application/json"),
    ("evaluation.json", "application/json"),
    ("final_report.html", "text/html"),
)


def resolve_target_api_key(params: ArksimJobParameters) -> str | None:
    """Resolve the target agent's API key.

    Prefers the EvalHub-mounted model secret
    (``/var/run/secrets/model/api-key``). Falls back to the env var named by
    ``target_api_key_env`` for local runs where no secret is mounted.
    """
    key = read_model_auth_key("api-key")
    if key:
        return key
    if params.target_api_key_env:
        env_key = os.environ.get(params.target_api_key_env)
        if not env_key:
            raise ValueError(
                f"target_api_key_env='{params.target_api_key_env}' is set but that "
                "environment variable is empty or unset. Export it, or remove "
                "target_api_key_env if the target agent needs no credentials."
            )
        return env_key
    return None


# Best-effort secret patterns scrubbed from strings before they are logged or
# sent to EvalHub: bearer tokens, URL userinfo, OpenAI-style keys.
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[\w.\-+/=]+"),
    re.compile(r"//[^/\s@]+@"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
)


def _scrub(text: str) -> str:
    """Redact likely secrets from a free-text string (defense in depth)."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _redact_url(url: str) -> str:
    """Reduce a target URL to scheme://host[:port]/path for logging.

    Drops any userinfo and query string, which can carry credentials. Falls
    back to generic scrubbing for malformed or scheme-less URLs.
    """
    parts = urlsplit(url)
    if not parts.netloc:
        # Malformed/scheme-less: userinfo can hide in the path, so don't risk
        # logging it. ModelConfig requires a non-empty url, not a well-formed one.
        return "[redacted: malformed target url]"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


async def simulate_and_evaluate(
    agent_config: AgentConfig,
    params: ArksimJobParameters,
    scenarios: Scenarios,
    output_dir: Path,
) -> tuple[Simulation, Evaluation]:
    """Run the arksim simulation then evaluation against the target agent.

    Isolated as a module-level function so it can be substituted in tests
    without making real model calls. Writes ``simulation.json``,
    ``evaluation.json``, and ``final_report.html`` into ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation = await run_simulation(
        SimulationInput(
            agent_config=agent_config,
            model=params.simulator_model,
            provider=params.simulator_provider,
            num_conversations_per_scenario=params.num_conversations_per_scenario,
            max_turns=params.max_turns,
            num_workers=params.num_workers,
            output_file_path=str(output_dir / _SIMULATION_FILE),
        ),
        scenarios=scenarios,
    )

    # Pass optional fields only when set: EvaluationInput supplies its own
    # defaults (e.g. metrics_to_run defaults to arksim's standard metric set),
    # which passing None would override.
    eval_kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "model": params.evaluator_model,
        "provider": params.evaluator_provider,
        "generate_html_report": True,
    }
    if params.metrics_to_run is not None:
        eval_kwargs["metrics_to_run"] = params.metrics_to_run
    if params.numeric_thresholds is not None:
        eval_kwargs["numeric_thresholds"] = params.numeric_thresholds
    if params.qualitative_failure_labels is not None:
        eval_kwargs["qualitative_failure_labels"] = params.qualitative_failure_labels

    # run_evaluation is synchronous and CPU/IO bound; keep the event loop free.
    evaluation = await asyncio.to_thread(
        run_evaluation,
        EvaluationInput(**eval_kwargs),
        simulation,
        scenarios,
    )
    return simulation, evaluation


class ArksimAdapter(FrameworkAdapter):
    """Runs arksim multi-turn agent evaluation as an EvalHub benchmark."""

    def __init__(
        self,
        settings: object | None = None,
        job_spec_path: str | None = None,
    ) -> None:
        super().__init__(settings=settings, job_spec_path=job_spec_path)
        # Populated by run_benchmark_job; logged to MLflow by main().
        self.mlflow_artifacts: list[MlflowArtifact] = []

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        started = time.monotonic()

        self._report(callbacks, JobPhase.INITIALIZING, "Parsing job parameters")
        params = ArksimJobParameters.model_validate(config.parameters)
        scenarios = self._select_scenarios(params.scenarios, config.num_examples)

        api_key = resolve_target_api_key(params)
        if api_key and any(h.lower() == "authorization" for h in params.extra_headers):
            logger.warning(
                "extra_headers includes 'Authorization'; it overrides the "
                "resolved api-key for the target agent."
            )
        # Trust model: the JobSpec is operator-authored and the adapter runs
        # single-tenant per job, so config.model.url is trusted (no SSRF
        # allowlist). Revisit if EvalHub ever accepts cross-tenant submissions.
        agent_config = build_agent_config(config.model, params, api_key)
        output_dir = self._output_dir()

        self._report(
            callbacks,
            JobPhase.RUNNING_EVALUATION,
            f"Simulating {len(scenarios.scenarios)} scenario(s) against "
            f"{_redact_url(config.model.url)}",
        )
        # run_benchmark_job is synchronous per the SDK contract (no enclosing
        # event loop), so asyncio.run owns the loop for the duration of the job.
        _simulation, evaluation = asyncio.run(
            simulate_and_evaluate(agent_config, params, scenarios, output_dir)
        )

        self._report(callbacks, JobPhase.POST_PROCESSING, "Aggregating scores")
        results = JobResults(
            id=config.id,
            benchmark_id=config.benchmark_id,
            benchmark_index=config.benchmark_index,
            model_name=config.model.name,
            results=aggregate_metrics(evaluation),
            overall_score=compute_overall_score(evaluation),
            num_examples_evaluated=len(evaluation.conversations),
            duration_seconds=time.monotonic() - started,
        )
        self.mlflow_artifacts = self._collect_artifacts(output_dir)
        return results

    @staticmethod
    def _select_scenarios(scenarios: Scenarios, num_examples: int | None) -> Scenarios:
        """Cap the scenario count when EvalHub requests a subset.

        ``JobSpec.num_examples`` has no lower bound, so reject non-positive
        values explicitly rather than producing an empty or negatively-sliced
        scenario set that fails confusingly downstream.
        """
        if num_examples is None or num_examples >= len(scenarios.scenarios):
            return scenarios
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}")
        return Scenarios(
            schema_version=scenarios.schema_version,
            scenarios=scenarios.scenarios[:num_examples],
        )

    def _output_dir(self) -> Path:
        """Per-run output directory.

        In local mode the SDK scopes ``local_jobs_base_path`` per job and
        benchmark, so outputs never collide across runs. Otherwise (e.g. in
        cluster mode, where the property is None) fall back to a writable temp
        dir keyed by job id; the installed package directory may be read-only or
        root-owned.
        """
        base = self.local_jobs_base_path
        if base is None:
            base = Path(tempfile.gettempdir()) / "arksim-evalhub" / self.job_spec.id
        return base / "results"

    def _collect_artifacts(self, output_dir: Path) -> list[MlflowArtifact]:
        artifacts: list[MlflowArtifact] = []
        for name, content_type in _ARTIFACTS:
            path = output_dir / name
            if path.is_file():
                artifacts.append(
                    MlflowArtifact(
                        path=name,
                        content=path.read_bytes(),
                        content_type=content_type,
                    )
                )
            else:
                logger.warning(
                    "Job %s: expected artifact %s not found at %s",
                    self.job_spec.id,
                    name,
                    path,
                )
        return artifacts

    @staticmethod
    def _report(callbacks: JobCallbacks, phase: JobPhase, message: str) -> None:
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=phase,
                message=MessageInfo(message=message, message_code=phase.value),
            )
        )


def main() -> None:
    """Adapter entrypoint. EvalHub runs ``python main.py`` inside the job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    adapter = ArksimAdapter()
    callbacks = DefaultCallbacks.from_adapter(adapter)

    try:
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
    except Exception as exc:
        # logger.exception emits the traceback to local logs only; avoid log
        # handlers that serialize frame locals, which could capture the api-key.
        logger.exception("arksim benchmark failed")
        # The message is forwarded off-box to EvalHub, so scrub likely secrets
        # (an upstream error may echo a credentialed URL or token).
        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.FAILED,
                phase=JobPhase.COMPLETED,
                message=MessageInfo(
                    message=_scrub(str(exc)), message_code="job_failed"
                ),
            )
        )
        raise

    # MLflow is gated on the tracking URI (injected by EvalHub in-cluster).
    # Skip it locally so the adapter runs without an MLflow server.
    if os.environ.get("MLFLOW_TRACKING_URI"):
        run_id = callbacks.mlflow.save(
            results, adapter.job_spec, artifacts=adapter.mlflow_artifacts
        )
        if run_id:
            results.mlflow_run_id = run_id
    else:
        logger.info(
            "MLFLOW_TRACKING_URI not set; skipping MLflow logging "
            "(%d artifact(s) available)",
            len(adapter.mlflow_artifacts),
        )

    callbacks.report_results(results)


if __name__ == "__main__":
    main()
