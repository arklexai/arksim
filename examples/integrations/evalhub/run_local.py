# SPDX-License-Identifier: Apache-2.0
"""Run the arksim EvalHub adapter locally, without Kubernetes or Docker.

This uses the SDK-only local path: write a JobSpec to the conventional local
job-spec location, point ``EVALHUB_JOB_SPEC_PATH`` at it, then invoke the
adapter entrypoint.

Prerequisites:
  * An LLM key for the simulator/judge (e.g. ``OPENAI_API_KEY``), since arksim
    generates the simulated user and judges transcripts with real models.
  * A reachable target endpoint at the JobSpec's ``model.url``. The sample spec
    points at OpenAI chat-completions and reuses ``OPENAI_API_KEY`` as the
    target key via ``parameters.target_api_key_env``.

Optional:
  * ``MLFLOW_TRACKING_URI`` to log transcripts + report as MLflow artifacts.

Usage:
    python run_local.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    spec = json.loads((HERE / "job.example.json").read_text())

    # EvalHub local layout the SDK expects, under a private (0700) temp dir:
    #   .../{job_id}/{benchmark_index}/{provider_id}/{benchmark_id}/meta/job.json
    spec_path = (
        Path(tempfile.mkdtemp(prefix="evalhub-jobs-"))
        / spec["id"]
        / str(spec["benchmark_index"])
        / spec["provider_id"]
        / spec["benchmark_id"]
        / "meta"
        / "job.json"
    )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2))

    os.environ["EVALHUB_JOB_SPEC_PATH"] = str(spec_path)
    os.environ.setdefault("EVALHUB_MODE", "local")

    print(f"Job spec written to: {spec_path}")

    from arksim_evalhub import main as adapter_main

    adapter_main()


if __name__ == "__main__":
    main()
