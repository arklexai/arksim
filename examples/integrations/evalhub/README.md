# arksim on EvalHub

Run arksim's multi-turn agent simulation and evaluation as an
[EvalHub](https://eval-hub.github.io/) benchmark provider.

EvalHub points at a target agent endpoint; this adapter drives that endpoint
with simulated users across your scenarios, scores the transcripts, and reports
aggregate metrics to EvalHub plus transcripts and an HTML report to MLflow.

## How it maps

| EvalHub | arksim |
|---|---|
| `JobSpec.model.url` / `.name` | target agent endpoint (`chat_completions` or `a2a`) |
| `JobSpec.model.auth.secret_ref` → `/var/run/secrets/model/api-key` | bearer token on the agent's requests |
| `JobSpec.parameters` (freeform JSON) | scenarios + simulator/judge config (see `ArksimJobParameters`) |
| scalar metrics | `overall_agent_score`, `goal_completion_score`, `turn_success_ratio`, `num_conversations` |
| MLflow artifacts | `simulation.json`, `evaluation.json`, `final_report.html` |

Three distinct models are in play: the **target agent** (`JobSpec.model`), the
**simulator** LLM that plays the user (`parameters.simulator_model`), and the
**judge** LLM that scores transcripts (`parameters.evaluator_model`).

## Files

| File | Description |
|------|-------------|
| `arksim_evalhub/mapping.py` | Pure EvalHub<->arksim transforms (no I/O), unit-tested |
| `arksim_evalhub/adapter.py` | `FrameworkAdapter`, credential resolution, the run, and `main()` |
| `main.py` | Container entrypoint (`python main.py`) |
| `run_local.py` | Run locally without Kubernetes or Docker |
| `job.example.json` | Sample EvalHub JobSpec |
| `provider.yaml` | Provider registration template for `eval-hub-server --local` |
| `Containerfile` | Container image build |
| `requirements.txt` | `arksim` + `eval-hub-sdk[adapter]` |
| `tests/` | Mapping + adapter wiring tests (no LLM calls) |

## Run locally (no Kubernetes)

```bash
pip install -r requirements.txt           # arksim + eval-hub-sdk[adapter]
export OPENAI_API_KEY=sk-...              # drives the simulator + judge, and
                                          # the sample target endpoint
python run_local.py
```

`run_local.py` writes `job.example.json` to the local job-spec path the SDK
expects, sets `EVALHUB_JOB_SPEC_PATH`, and invokes the adapter. Edit
`job.example.json` to point `model.url` at your own agent and to define your
scenarios under `parameters.scenarios`.

To log transcripts and the report to MLflow, also set
`MLFLOW_TRACKING_URI=http://localhost:5000` (otherwise MLflow logging is
skipped). Either way the three artifacts (`simulation.json`, `evaluation.json`,
`final_report.html`) are written to a `results/` directory under the run's
output path, which `run_local.py` prints; open `final_report.html` to view the
run. Transcripts can contain PII from your scenarios and the agent's responses,
so scope MLflow access accordingly.

The `Connection refused` warnings for the callback URL are expected in local
mode: EvalHub status callbacks are a no-op without a running server.

## Configuring a run

Everything benchmark-specific lives in `JobSpec.parameters`, validated by
`ArksimJobParameters` (see `arksim_evalhub/mapping.py` for the full field set and
descriptions). To point at your own agent, edit `job.example.json`:

- `model.url` / `model.name`: your target agent endpoint and model id.
- `parameters.protocol`: `chat_completions` (OpenAI-compatible) or `a2a`.
- `parameters.target_api_key_env`: env var holding the target's key (local runs).
  If set but unset at runtime, the job fails fast with a clear message.
- `parameters.simulator_model` / `evaluator_model`: arksim's own LLMs (the
  simulated user and the judge), independent of the target agent. Omitted, they
  fall back to arksim's defaults (currently `gpt-5.1`), not the example's values.
- `parameters.metrics_to_run`: any subset of `helpfulness`, `coherence`,
  `verbosity`, `relevance`, `faithfulness`, `goal_completion`,
  `agent_behavior_failure`. Omit to use arksim's defaults.
- `parameters.num_workers`: a positive integer, or `"auto"`.
- `parameters.scenarios`: at least one scenario (`scenario_id`, `user_id`,
  `goal`, `agent_context`, `user_profile`).

Only `scenarios` is required; everything else has defaults. Pass/fail is decided
by EvalHub from `provider.yaml` (`primary_score` + `pass_criteria`), not here.

## Tests

The pure mapping and the adapter wiring are covered without any LLM calls (the
sim/eval step is stubbed):

```bash
pip install pytest
python -m pytest tests/
```

## Credentials note for the EvalHub team

The model-authentication docs say EvalHub "applies credentials automatically,"
but the SDK only auto-attaches the ServiceAccount token to control-plane
callbacks. The adapter reads `read_model_auth_key("api-key")` itself and puts it
on the agent's outbound requests. TLS `ca_cert` from the mounted secret is not
yet wired into arksim's HTTP client; flag if you need custom-CA target
endpoints.

## Status

Proof of concept for local mode. The `provider.yaml` field names are
illustrative; confirm them against your EvalHub version before wiring CI or the
k8s runtime.
