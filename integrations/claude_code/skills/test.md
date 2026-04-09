# /arksim test

Simulate conversations against an agent and evaluate the results.

## When to use

- First time testing an agent (no config.yaml or scenarios.json yet)
- Re-running after code changes to check for regressions
- Validating agent behavior against specific user scenarios

## Flow: first time (no config.yaml in project root)

### 1. Detect the agent

Scan the project for agent files. Look for:

- Python files that import `openai`, `langchain`, `crewai`, `pydantic_ai`, `smolagents`, `autogen`, `google.adk`, `claude_agent_sdk`, `llamaindex`, `langgraph`, `rasa`, `dify`
- Classes that subclass `BaseAgent`
- Functions decorated with `@tool`
- Chat Completions endpoints (HTTP servers exposing `/v1/chat/completions`)
- A2A agent cards (`.well-known/agent.json`)

Ask the user to confirm which file is the agent entry point.

### 2. Determine agent type

Based on the agent, choose one of:

| Agent type | When to use |
|---|---|
| `custom` | Python agent with a callable entry point (most frameworks) |
| `chat_completions` | Agent exposed as an OpenAI-compatible HTTP endpoint |
| `a2a` | Agent using the Agent-to-Agent protocol |

### 3. Initialize the project

Call the `init_project` MCP tool:

```
init_project(agent_type="custom")
```

This creates `config.yaml` and `scenarios.json` in the working directory. For custom agent type, it also creates `my_agent.py` with a starter agent stub.

### 4. Generate scenarios

Read the agent's source code to understand its domain, tools, and capabilities. Generate 3-5 domain-specific scenarios and present them to the user for review before saving.

Write the approved scenarios to `scenarios.json` using the schema below.

### 5. Run simulation and evaluation

Call the `simulate_evaluate` MCP tool:

```
simulate_evaluate(config_path="config.yaml")
```

## Flow: config exists

When `config.yaml` already exists, skip detection and initialization. Call `simulate_evaluate` directly.

If the user mentions code changes, remind them to update scenarios if the agent's capabilities changed.

## Formatting results

Present results as a markdown table:

```
| Scenario | Status | Helpfulness | Goal | Failures |
|---|---|---|---|---|
| order_status_check | PASSED | 4.2/5 | 0.85 | none |
| product_search | FAILED | 2.1/5 | 0.30 | false information |
```

- **Status**: PASSED or FAILED based on overall_agent_score and thresholds
- **Helpfulness**: mean helpfulness score across turns (1-5 scale)
- **Goal**: goal_completion_score (0-1 scale)
- **Failures**: comma-separated behavior failure labels, or "none"

## Next steps

Always end with 1-2 suggested actions based on the results:

- If all scenarios pass: "Try adding edge-case scenarios with `/arksim scenarios`"
- If failures exist: "Dive into the failures with `/arksim results` to see turn-by-turn details"
- If scores are borderline: "Re-evaluate with stricter thresholds using `/arksim evaluate`"

## Scenario JSON schema

```json
{
  "schema_version": "v1",
  "scenarios": [
    {
      "scenario_id": "string (snake_case, unique within the file)",
      "user_id": "string (identifies the simulated user persona)",
      "goal": "string (what the user wants to accomplish, including any relevant context about the situation)",
      "agent_context": "string (system prompt or description given to the simulated user so it knows what to expect from the agent)",
      "user_profile": "string (demographics, personality traits, communication style of the simulated user)",
      "knowledge": [
        {"content": "string (ground truth the simulated user can reference, e.g. order details, account info)"}
      ],
      "assertions": [
        {
          "type": "tool_calls",
          "expected": [{"name": "tool_name"}],
          "match_mode": "strict | unordered | contains | within"
        }
      ]
    }
  ]
}
```

### Field descriptions

- **scenario_id**: Unique identifier. Use snake_case that describes the behavior being tested (e.g. `cancel_shipped_order`, `out_of_scope_question`).
- **user_id**: Groups scenarios by persona. Reuse the same user_id when the same persona appears in multiple scenarios.
- **goal**: What the simulated user is trying to accomplish. Include any situational context here (not in user_profile). Example: "Cancel order ORD-1002, which was placed yesterday and is still processing."
- **agent_context**: Tells the simulated user what the agent can do, so it sets realistic expectations. Leave empty if not applicable.
- **user_profile**: Demographics and personality only. Example: "You are Alex, a 35-year-old software engineer. You are patient and detail-oriented." Do not put scenario-specific context here.
- **knowledge**: Ground truth facts the simulated user can reference during the conversation. Each item is a self-contained fact.
- **assertions**: Optional tool-call trajectory checks. `match_mode` controls strictness: `strict` (exact order and set), `unordered` (same set, any order), `contains` (expected is a subset of actual calls), `within` (actual calls are a subset of expected tools).

### Best practices

- **user_profile is demographics only**. Scenario-specific context (what the user wants, what happened before) goes in `goal`.
- **Use relative dates**. Write "placed yesterday" or "ordered last week" instead of "placed on 2024-03-15". Absolute dates rot.
- **One behavior per scenario**. A scenario named `cancel_and_refund_and_check_status` is testing three things. Split it.
- **Include negative cases**. Test what happens when the agent cannot help, gets bad input, or encounters an error.
- **Knowledge is ground truth**. Put facts here that the simulated user should know (verification codes, order details), not instructions for the agent.
