# Export arksim runs to Langfuse

Push arksim simulations and evaluations to a [Langfuse](https://langfuse.com)
instance (self-hosted or cloud) using the public Langfuse Python SDK (v4).

The exporter maps:

| arksim                     | Langfuse                          |
| -------------------------- | --------------------------------- |
| `Scenario`                 | dataset item                      |
| `Conversation`             | trace (turns -> generations, tool calls -> tool spans) |
| the simulation batch       | dataset run (`dataset.run_experiment`) |
| `ConversationEvaluation`   | trace-level scores                |
| per-turn metric results    | scores on each turn's span        |

## 1. Install

```bash
pip install 'arksim[langfuse]'
```

## 2. Start a local (self-hosted) Langfuse

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d          # serves http://localhost:3000
```

Open http://localhost:3000, create an account and a project, then copy the
project's API keys from Settings.

## 3. Point the example at it

```bash
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
export LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx

python examples/langfuse/export_to_langfuse.py
```

You should see `Export complete.` followed by a run summary. In the Langfuse
UI, check:

- **Datasets -> arksim-demo**: 2 items (`cancel-order`, `store-hours`), each
  with `expected_output` where the scenario declared a tool-call assertion.
- **Datasets -> arksim-demo -> Runs -> arksim-demo-run**: 2 linked traces.
- **Each trace**: nested `scenario -> conversation -> turn -> tool` spans, with
  trace-level scores (`overall_agent_score`, `goal_completion`,
  `turn_success_ratio`, `evaluation_status`) and per-turn scores
  (`helpfulness`, `coherence`, `tone`).

## 4. Use it with a real run

`export_to_langfuse.py` builds sample data so it runs without LLM keys. For a
real simulate -> evaluate -> export against a live agent, use the ready-made
`run_and_export.py` (self-contained OpenAI setup: the agent-under-test is a
second OpenAI model driven by the same key):

```bash
export OPENAI_API_KEY=sk-...            # sim/eval LLM + agent-under-test
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

python examples/langfuse/run_and_export.py
```

Optional overrides: `ARKSIM_MODEL`, `ARKSIM_PROVIDER`, `AGENT_MODEL`,
`AGENT_ENDPOINT`, `AGENT_API_KEY` (see the script's docstring).

Or wire it yourself by passing arksim objects straight through:

```python
import asyncio
from arksim import (
    Scenarios, SimulationInput, EvaluationInput, run_simulation, run_evaluation,
)
from arksim.integrations.langfuse import export_to_langfuse

scenarios = Scenarios.load("scenarios.json")
simulation = asyncio.run(run_simulation(sim_settings, scenarios=scenarios))
evaluation = run_evaluation(eval_settings, simulation=simulation, scenarios=scenarios)

export_to_langfuse(
    scenarios, simulation, evaluation,
    dataset_name="my-agent-regression",
    run_name="v1.2.3",
)
```

### Programmatic use

```python
from arksim.integrations.langfuse import LangfuseExporter

exporter = LangfuseExporter(
    public_key="pk-lf-...", secret_key="sk-lf-...", host="http://localhost:3000",
)
result = exporter.export(
    scenarios, simulation, evaluation,
    dataset_name="my-agent-regression",
    run_name="nightly",
)
print(result.format())
```

You can also pass an already-configured client:
`LangfuseExporter(client=Langfuse(...))`.

## Notes

- A Langfuse dataset run links **one trace per dataset item**. When a scenario
  has a single conversation (the common case), the mapping is a clean 1:1 and
  conversation scores are attached at the trace level so they drive the
  run-comparison view. When `num_conversations_per_scenario > 1`, each
  conversation is a nested span subtree under the item's trace, per-conversation
  scores go on those spans, and the mean `overall_agent_score` is attached at
  the trace level.
- `create_dataset` and `create_dataset_item` are upserts (keyed by name / id),
  so re-running against the same dataset is safe. Pass `create_items=False` to
  reuse existing items.
