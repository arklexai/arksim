# /arksim evaluate

Re-evaluate simulation results with different settings without re-running the agent.

## When to use

- Trying different evaluation metrics (add `faithfulness`, remove `verbosity`)
- Adjusting pass/fail thresholds (raise `overall_score` from 0.6 to 0.8)
- Switching the judge model (e.g. from `gpt-4.1-mini` to `gpt-4.1`)
- Running custom metrics you just wrote

Re-evaluation is cheaper than re-simulation because it only runs the judge LLM against existing conversation transcripts. The agent is not invoked again.

## Flow

### 1. Find the simulation output

Look for the most recent simulation output file. The default location is `./results/simulation/simulation.json`, but check `config.yaml` for a custom `output_file_path`.

If no simulation output exists, suggest running `/arksim test` first.

### 2. Ask what to change

Ask the user what they want to evaluate differently. Common changes:

| Change | Config field |
|---|---|
| Different metrics | `metrics_to_run` |
| Stricter pass/fail | `numeric_thresholds` |
| Fail on specific labels | `qualitative_failure_labels` |
| Different judge model | `model` and `provider` |
| Custom metric files | `custom_metrics_file_paths` |

### 3. Run evaluation

Call the `evaluate` MCP tool with the simulation file path and any changed settings:

```
evaluate(config_path="config.yaml")
```

### 4. Format results

Present results in the same table format as `/arksim test`, but highlight what changed compared to the previous evaluation:

- If thresholds changed, note which scenarios flipped from PASSED to FAILED or vice versa
- If metrics changed, show only the newly added metrics alongside the overall score
- If the judge model changed, note this so the user understands scores may shift

## Available built-in metrics

| Metric | Type | Scale | What it measures |
|---|---|---|---|
| `helpfulness` | quantitative | 1-5 | Whether the agent's response is useful to the user |
| `faithfulness` | quantitative | 1-5 | Whether the response is grounded in provided knowledge |
| `coherence` | quantitative | 1-5 | Logical consistency across turns |
| `relevance` | quantitative | 1-5 | Whether the response addresses the user's question |
| `verbosity` | quantitative | 1-5 | Appropriate response length (not too short or long) |
| `goal_completion` | quantitative | 0-1 | Whether the user's goal was achieved |
| `agent_behavior_failure` | qualitative | label | Detects harmful agent behaviors (false information, disobey user request, etc.) |
| `tool_call_behavior_failure` | qualitative | label | Detects incorrect tool usage patterns |
