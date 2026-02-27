# Arksim

**Test and evaluate your AI agents with realistic, user-driven simulations.**

Arksim lets you define who your users are, what they want, and what they know, then runs them as live multi-turn conversations against your agent. The result is structured performance data you can actually act on.

<img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python" /> <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License" /> <img src="https://img.shields.io/pypi/v/arksim.svg" alt="PyPI" />

---

## Why Arksim?

Most agent testing is either manual (slow, inconsistent) or based on static Q&A pairs (shallow, unrealistic). Arksim sits in between — giving you repeatable, persona-driven simulations that behave like real users and evaluate like a structured test suite.

- **Scenario-driven** — define user goals, personas, and background knowledge
- **Multi-turn** — simulated users drive full conversations, not single prompts
- **Structured evaluation** — every turn is scored across helpfulness, coherence, relevance, verbosity, and faithfulness
- **Cross-conversation insights** — unique errors are deduplicated and surfaced across all conversations so you know exactly where your agent breaks down

---

## How It Works

```
Scenarios → Simulation → Evaluation
```

1. **Scenarios** — define who the simulated user is, what they want, and what they know
2. **Simulation** — Arksim runs each scenario as a live conversation against your agent
3. **Evaluation** — every agent response is scored per turn, with failures categorized and deduplicated across conversations

---

## Installation

```bash
pip install arksim
```

---

## Quickstart

**1. Define a scenario**

```json
{
  "schema_version": "v1",
  "scenarios": [
    {
      "scenario_id": "demo-001",
      "goal": "You want to find out whether your home insurance covers water damage from a burst pipe.",
      "user_attribute": {
        "customer_type": "existing customer",
        "location": "Toronto, ON, Canada",
        "decision_making_style": "analytical"
      }
    }
  ]
}
```

**2. Run a simulation**

```bash
arksim simulate config.yaml
```

Or in Python:

```python
from arksim.simulation_engine import Simulator, SimulationParams
from arksim.config import AgentConfig

scenarios = Simulator.load_scenarios("scenario.json")
agent_config = AgentConfig.load("agent_config.json")

engine = Simulator(
    agent_config=agent_config,
    scenarios=scenarios,
    params=SimulationParams(num_conversations=10, max_turns=8),
)

conversations = engine.simulate()
```

**3. Evaluate your agent**

```bash
arksim evaluate config.yaml
```

Or in Python:

```python
import arksim

results = arksim.evaluate(
    conversation_file_path="./conversations.json",
    output_dir="./evaluation",
    model="gpt-5.1",
    provider="open-ai",
    generate_html_report=True,
)
```

---

## Output

Simulation produces `conversations.json` — full transcripts of every run.

Evaluation produces:

| File                                     | What's in it                                                          |
| ---------------------------------------- | --------------------------------------------------------------------- |
| `agent_performance_per_turn.csv`         | Per-turn metric scores, reasoning, and failure labels                 |
| `agent_performance_per_conversation.csv` | Rolled-up scores and status per conversation                          |
| `unique_errors.csv`                      | Deduplicated failure patterns with descriptions and occurrence traces |
| `final_report.md` / `.html`              | High-level summary with top errors and overall assessment             |

---

## Documentation

Full docs at [**docs.arksim**](#)— including schema references, agent compatibility, configuration options, and evaluation deep-dives.

---

## Contributing

Contributions are welcome! Here's how to get started:

```bash
git clone https://github.com/arklexai/arksim.git
cd arksim
pip install -e ".[dev]"
```

Please open an issue before submitting a large PR so we can discuss the change first. For smaller fixes and improvements, PRs are welcome directly.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## License

MIT — see [LICENSE](LICENSE) for details.