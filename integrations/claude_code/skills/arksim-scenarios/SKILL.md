---
name: arksim:scenarios
description: Generate or edit test scenarios for your agent
---
# /arksim scenarios

Generate, edit, or extend scenarios for agent testing.

## When to use

- Generating scenarios for a new agent or new capabilities
- Writing regression tests after discovering a failure
- Editing existing scenarios to improve coverage

## Modes

### Generate new scenarios

Read the agent's source code to understand its domain, tools, and capabilities. Generate scenarios that cover:

1. **Happy path**: straightforward request the agent should handle well
2. **Edge case**: unusual input, boundary condition, or rare but valid request
3. **Out of scope**: request the agent should decline or redirect
4. **Multi-step**: goal that requires several turns to accomplish
5. **Error recovery**: what happens when a tool fails or returns unexpected data

Present generated scenarios to the user for review. Do not write to `scenarios.json` until the user approves.

### Generate from failure

When a previous run produced failures, read the evaluation results to understand what went wrong. Generate targeted scenarios that:

- Reproduce the exact failure condition
- Test slight variations of the failing input
- Verify the fix does not break the happy path

This is useful for building regression test suites after fixing agent bugs.

### Edit existing scenarios

Read the current `scenarios.json` and present the scenarios to the user. Accept edits and write the updated file.

When editing, validate that:
- Every `scenario_id` is unique within the file
- Every scenario has a non-empty `goal` and `user_profile`
- `schema_version` is `"v1"`

## Scenario JSON schema

```json
{
  "schema_version": "v1",
  "scenarios": [
    {
      "scenario_id": "string (snake_case, unique within the file)",
      "user_id": "string (identifies the simulated user persona)",
      "goal": "string (what the user wants to accomplish, including situational context)",
      "agent_context": "string (what the agent does, given to the simulated user)",
      "user_profile": "string (demographics and personality only)",
      "knowledge": [
        {"content": "string (ground truth fact the simulated user knows)"}
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

## Assertion match modes

| Mode | Behavior |
|---|---|
| `strict` | Expected tools must match actual calls in exact order and exact set |
| `unordered` | Same set of tools, any order |
| `contains` | Expected tools are a subset of actual calls |
| `within` | Actual calls are a subset of expected tools (expected is a superset) |

## Examples

### Customer service happy path

```json
{
  "scenario_id": "order_status_check",
  "user_id": "user_patient",
  "goal": "Check the status of order ORD-1001, which you placed recently.",
  "agent_context": "You are talking to a customer service assistant with access to order lookup and identity verification tools. The agent will verify your identity before sharing order details.",
  "user_profile": "You are Alice, a 32-year-old premium customer. You are polite and expect accurate information. You communicate clearly.",
  "knowledge": [
    {"content": "Your email is alice@example.com. Your verification code is 123456. Order ORD-1001 contains Wireless Headphones x1, totaling $249.99. The order status is shipped."}
  ],
  "assertions": [
    {"type": "tool_calls", "expected": [{"name": "verify_customer"}, {"name": "get_order"}], "match_mode": "contains"}
  ]
}
```

### Out-of-scope request

```json
{
  "scenario_id": "medical_advice_declined",
  "user_id": "user_confused",
  "goal": "Ask the agent for medical advice about a headache. The agent should politely decline since this is outside its domain.",
  "agent_context": "You are talking to a customer service assistant for an online store. It cannot provide medical, legal, or financial advice.",
  "user_profile": "You are Jordan, a 28-year-old marketing manager. You are not very technical and sometimes ask agents for help with things outside their scope.",
  "knowledge": []
}
```

## Best practices

- **user_profile is demographics only**. Scenario-specific context (what the user ordered, what happened before) goes in `goal`.
- **Use relative dates**. Write "placed yesterday" or "ordered last week" instead of absolute dates that become stale.
- **One behavior per scenario**. If you need "and" to describe what a scenario tests, split it into two scenarios.
- **Include negative cases**. Test what happens when the agent cannot help, gets bad input, or encounters a tool error.
- **Knowledge is ground truth**. Put verifiable facts here (order details, verification codes). Do not put agent instructions in knowledge.
- **Name scenarios descriptively**. `cancel_shipped_order` is clear. `test_case_3` is not.
