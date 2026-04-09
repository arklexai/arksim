# arksim for Claude Code

Test your AI agents inside Claude Code. Generate scenarios from your agent's code, run simulated conversations, and see results inline.

## Install

```bash
pip install "arksim[claude]"
arksim setup-claude
```

`setup-claude` writes skill files to `.claude/skills/arksim-*/SKILL.md` and registers the MCP server in `.mcp.json` at the project root. No global configuration is modified.

## What it does

ArkSim generates synthetic users that hold realistic multi-turn conversations with your agent. Each user has a distinct profile, goal, and knowledge level. They push back, ask follow-ups, and behave like real users would.

After simulation, ArkSim evaluates every turn across metrics like helpfulness, faithfulness, and goal completion. You get a report showing exactly where your agent broke and why.

### Skills

| Skill | Description |
|-------|-------------|
| `/arksim:test` | Test your agent. First time: guided setup (discovers your agent, generates scenarios, runs simulations). After that: re-runs the existing test suite. |
| `/arksim:evaluate` | Re-evaluate a previous run with different metrics or thresholds without re-simulating. |
| `/arksim:scenarios` | Generate new test scenarios from your agent's code, or edit existing ones. |
| `/arksim:results` | Explore results from previous runs. Drill into failures, compare runs side by side. |
| `/arksim:ui` | Open the web dashboard for browsing results interactively. |

## User journey: first time

You have a customer service agent. You want to know if it handles refund requests correctly.

```
You:    /arksim:test

Claude: I found your agent in src/agent.py. It's a customer service bot using
        OpenAI Agents SDK with 4 tools (lookup_customer, get_order,
        cancel_order, process_refund).

        I'll generate test scenarios based on your agent's tools and domain.

        Generated 6 scenarios:
        1. order_status_check - User asks about a recent order
        2. refund_eligible_order - User requests refund for a delivered item
        3. refund_ineligible_order - User requests refund outside return window
        4. cancel_processing_order - User wants to cancel before shipping
        5. authentication_flow - User must verify identity first
        6. out_of_scope_question - User asks about something the agent can't help with

        Running simulations... (6 scenarios, 3 turns each)

        Results:

        Scenario                  Helpful  Faithful  Goal  Pass
        ───────────────────────── ──────── ───────── ───── ────
        order_status_check           4.8      5.0     ✓     ✓
        refund_eligible_order        4.5      4.8     ✓     ✓
        refund_ineligible_order      3.1      4.2     ✗     ✗
        cancel_processing_order      4.6      5.0     ✓     ✓
        authentication_flow          4.4      4.9     ✓     ✓
        out_of_scope_question        2.8      3.5     ✗     ✗

        4/6 scenarios passed. 2 failures:

        - refund_ineligible_order: The agent approved the refund instead of
          explaining the return window policy. See turn 2.
        - out_of_scope_question: The agent hallucinated a product feature
          that does not exist. See turn 1.

        Saved to .arksim/results/evaluation.json.

        Next steps:
        - Run /arksim:results to inspect the failing conversations turn by turn
        - Run /arksim:scenarios to add edge cases for the failure patterns
        - Run /arksim:evaluate to try stricter thresholds
```

## User journey: iterating after code changes

You fixed the refund logic. Now you want to verify the fix and add a stricter metric.

```
You:    /arksim:test

Claude: Found existing config at .arksim/config.yaml with 6 scenarios.
        Running simulations...

        Results:

        Scenario                  Helpful  Faithful  Goal  Pass
        ───────────────────────── ──────── ───────── ───── ────
        order_status_check           4.8      5.0     ✓     ✓
        refund_eligible_order        4.5      4.8     ✓     ✓
        refund_ineligible_order      4.3      4.7     ✓     ✓  ← fixed
        cancel_processing_order      4.6      5.0     ✓     ✓
        authentication_flow          4.4      4.9     ✓     ✓
        out_of_scope_question        2.9      3.6     ✗     ✗

        5/6 passed (was 4/6). refund_ineligible_order is now passing.

        Remaining failure:
        - out_of_scope_question: Agent still hallucinating. See turn 1.

You:    Show me the out_of_scope_question conversation turn by turn.

Claude: Conversation: out_of_scope_question
        User profile: Confused shopper, low technical knowledge

        Turn 1 (User): "Can your product do real-time translation?"
        Turn 1 (Agent): "Yes, we support real-time translation in 40 languages."
        ↳ Faithfulness: 1.2 - Agent fabricated a feature that does not exist.

        Turn 2 (User): "Great, how do I enable it?"
        Turn 2 (Agent): "Go to Settings > Languages > Enable Translation."
        ↳ Faithfulness: 1.0 - Agent invented a settings path.

        Turn 3 (User): "I don't see that option anywhere."
        Turn 3 (Agent): "I apologize for the confusion. Let me check on that."
        ↳ Faithfulness: 3.8 - Agent acknowledged the mistake but too late.

You:    re-evaluate with just helpfulness and goal_completion

Claude: Re-evaluating with helpfulness and goal_completion only...

        Results:

        Scenario                  Helpfulness  Goal   Status
        ───────────────────────── ─────────── ─────  ────────────────
        order_status_check           4.8       1.0   Done
        refund_eligible_order        4.5       1.0   Done
        refund_ineligible_order      4.3       1.0   Done
        cancel_processing_order      4.6       1.0   Done
        authentication_flow          4.4       1.0   Done
        out_of_scope_question        2.9       0.0   Partial Failure

        Saved to results/evaluation.json.
```

## Uninstall

```bash
arksim setup-claude --uninstall
```

This removes the skill directories (`.claude/skills/arksim-*`) and the MCP server entry from `.mcp.json`. Your test results and config are not deleted.

## How it works

The integration adds two components to your project:

**Skills** are SKILL.md files installed to `.claude/skills/arksim-*/`. Each skill directory tells Claude Code what the command does, what arguments it accepts, and what MCP tools to call. Claude reads these files when you type a slash command.

**MCP Server** is a local process configured in `.mcp.json`. It exposes arksim operations (simulate, evaluate, read results) as tool calls that Claude Code can invoke. The server wraps the arksim CLI, so all work happens on your machine.

Everything runs locally. No data is sent to external services beyond the LLM API calls that arksim already makes for simulation and evaluation.

## Requirements

- Python 3.10+
- arksim with Claude extra (`pip install "arksim[claude]"`)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- An LLM API key (OpenAI, Anthropic, or Google) for simulation and evaluation
