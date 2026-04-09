# /arksim results

Inspect, analyze, and compare evaluation results.

## When to use

- Understanding why specific scenarios failed
- Debugging agent behavior turn by turn
- Comparing results across two runs to measure improvement

## Modes

### Summary view

Present an overview of the most recent evaluation. Call the `read_result` MCP tool to load the evaluation output, then display:

```
| Scenario | Status | Helpfulness | Goal | Failures |
|---|---|---|---|---|
| order_status_check | PASSED | 4.2/5 | 0.85 | none |
| product_search | FAILED | 2.1/5 | 0.30 | false information |
| cancel_order | PASSED | 3.8/5 | 0.70 | none |

Overall: 2/3 passed | 1 unique error | mean goal completion: 0.62
```

Include:
- Total pass/fail count
- Number of unique errors detected
- Mean goal completion score across all conversations

### Deep-dive (turn by turn)

When the user asks about a specific scenario or conversation, show the full conversation with per-turn annotations:

```
Turn 1:
  User: "I'd like to check my order status"
  Agent: "I'd be happy to help. Could you provide your email for verification?"
  Helpfulness: 4/5 | Behavior: no failure

Turn 2:
  User: "alice@example.com"
  Agent: "I've sent a verification code to your email. Please share it."
  Helpfulness: 4/5 | Behavior: no failure
  Tool calls: send_verification_code(email="alice@example.com")

Turn 3:
  User: "The code is 123456"
  Agent: "Your order ORD-1001 shipped yesterday and is in transit."
  Helpfulness: 5/5 | Behavior: no failure
  Tool calls: verify_customer(code="123456") -> get_order(order_id="ORD-1001")

Goal completion: 0.90 | Overall: PASSED
```

For failed turns, highlight the failure:

```
Turn 2:
  User: "What laptops do you have under $1000?"
  Agent: "We have the MacBook Air M4 at $899, which is a great choice!"
  Helpfulness: 1/5 | Behavior: FALSE INFORMATION
  Failure reason: Agent stated MacBook Air M4 costs $899 when the actual price is $1199.
```

### Compare runs

Compare two evaluation runs side by side to show what improved or regressed. This mode is skill-driven: call `read_result` twice (once for each run) and compute the diff.

Present changes as:

```
Comparing run abc123 vs run def456:

| Scenario | Goal (before) | Goal (after) | Delta |
|---|---|---|---|
| order_status_check | 0.70 | 0.85 | +0.15 |
| product_search | 0.30 | 0.65 | +0.35 |
| cancel_order | 0.80 | 0.75 | -0.05 |

Unique errors: 3 -> 1 (2 resolved, 0 new)
Mean goal completion: 0.60 -> 0.75 (+0.15)
```

Highlight:
- Scenarios that flipped from FAILED to PASSED (improvements)
- Scenarios that flipped from PASSED to FAILED (regressions)
- Net change in unique error count

## Finding result files

Call the `list_results` MCP tool to discover all evaluation runs under a directory:

```
list_results(output_dir="./results")
```

This returns a summary of every `evaluation.json` found, including pass/fail counts and timestamps. Use this to identify which run to inspect or compare.

Evaluation results are written to the `output_dir` specified in `config.yaml` (default: `./results/evaluation/`). The main file is `evaluation.json`.

Simulation results are at the `output_file_path` (default: `./results/simulation/simulation.json`).

If the user does not specify which run to inspect, use the most recent files found by `list_results`.

## Next steps

Based on findings, suggest:

- If failures are due to the agent: "The agent gave false information in 2 scenarios. Fix the agent and re-run with `/arksim test`."
- If failures are due to scenarios: "The scenario knowledge seems incomplete. Update scenarios with `/arksim scenarios`."
- If scores are borderline: "Scores are close to the threshold. Consider adding more conversations per scenario (`num_conversations_per_scenario`) to reduce variance."
- If comparing and regressions exist: "Scenario `cancel_order` regressed. Investigate the turn-by-turn diff before merging."
