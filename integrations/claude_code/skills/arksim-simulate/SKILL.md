---
name: arksim-simulate
description: Use when the user wants to simulate multi-turn conversations against an AI agent. Alias for the arksim-test skill; the canonical flow lives there.
allowed-tools: ["mcp__arksim__init_project", "mcp__arksim__simulate_evaluate", "mcp__arksim__read_result", "Read", "Write", "Edit", "Glob", "Grep"]
---
# arksim-simulate

This skill is an alias for `arksim-test`. The canonical multi-turn
simulation + evaluation flow is documented there. Both names exist so
users can ask for "test" or "simulate" interchangeably.

When invoked, follow the instructions in the `arksim-test` skill.
