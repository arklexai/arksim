<p align="center">
  <h1 align="center">⛵️ ArkSim</h1>
  <p align="center">
    Simulate multi-turn conversations with your AI agent. Find failures before production.
  </p>
  <p align="center">
    <a href="https://github.com/arklexai/arksim/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/arklexai/arksim/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://github.com/arklexai/arksim/actions/workflows/integration-tests.yml"><img alt="Integration Tests" src="https://github.com/arklexai/arksim/actions/workflows/integration-tests.yml/badge.svg"></a>
    <a href="https://app.codecov.io/gh/arklexai/arksim"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/arklexai/arksim"></a>
    <a href="https://pypi.org/project/arksim/"><img alt="PyPI" src="https://img.shields.io/pypi/v/arksim.svg?cacheSeconds=300"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/arksim.svg?cacheSeconds=300"></a>
    <a href="https://github.com/arklexai/arksim/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
    <a href="https://docs.arklex.ai/main/overview"><img alt="Docs" src="https://img.shields.io/badge/docs-arklex.ai-brightgreen.svg"></a>
    <a href="https://github.com/arklexai/arksim/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/arklexai/arksim.svg?style=social"></a>
    <a href="https://github.com/arklexai/arksim/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/arklexai/arksim.svg"></a>
    <a href="https://github.com/arklexai/arksim/pulls"><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
    <a href="https://arxiv.org/abs/2510.11997"><img alt="2510.11997" src="https://img.shields.io/badge/arXiv-2510.11997-b31b1b.svg"></a>
  </p>
  <p align="center">
    <a href="https://docs.arklex.ai/main/overview">Documentation</a> · <a href="examples/">Examples</a> · <a href="https://github.com/arklexai/arksim/issues">Report a Bug</a>
  </p>
</p>




https://github.com/user-attachments/assets/78706f27-cf49-41c1-8019-9dcbb8abc625




## What is ArkSim?

Agents fail in ways that only show up mid-conversation. They misinterpret intent three turns in, call the wrong tool, or hallucinate a policy that does not exist. Single-turn testing misses all of this.

ArkSim generates LLM-powered synthetic users that hold realistic multi-turn conversations with your agent. Each user has a distinct profile, goal, and knowledge level. They push back, ask follow-ups, and behave like real users would.

You define scenarios, ArkSim simulates conversations, then evaluates every turn across metrics like helpfulness, faithfulness, and goal completion. The output is an interactive report showing exactly where your agent broke and why.

<p align="center">
  <img src="https://raw.githubusercontent.com/arklexai/arksim/main/docs/assets/arksim-flow.svg" alt="ArkSim flow: Scenarios → Simulation → Evaluation → Reports" width="100%">
</p>

## Quickstart

### Have an agent? Test it in 3 commands:

```bash
pip install arksim
export OPENAI_API_KEY="your-key"
arksim init
# Edit my_agent.py with your agent logic, then run:
arksim simulate-evaluate config.yaml
```

This generates `config.yaml`, `scenarios.json`, and a starter `my_agent.py`. No server needed.

For HTTP or A2A agents: `arksim init --agent-type chat_completions` or `arksim init --agent-type a2a`.
For Anthropic or Google as the evaluation LLM: `pip install "arksim[anthropic]"` or `pip install "arksim[google]"`.

### Just exploring? Try an example:

```bash
pip install arksim
export OPENAI_API_KEY="your-key"
arksim examples
cd examples/e-commerce
arksim simulate-evaluate config.yaml
```

### What you'll see

<p align="center">
  <img src="https://raw.githubusercontent.com/arklexai/arksim/main/docs/assets/report-screenshot.png" alt="ArkSim evaluation report showing scores, failure categories, and conversation viewer" width="100%">
</p>

ArkSim generated synthetic users with different profiles and goals, ran multi-turn conversations with the agent, and scored every turn. The report breaks down what went well and what failed, with full conversation transcripts so you can see exactly what happened.

## Test Your Own Agent

### Python class (default, no server needed)

`arksim init` generates a `my_agent.py` with a BaseAgent subclass. Replace the `execute()` body with your agent logic:

```python
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import AgentResponse

class MyAgent(BaseAgent):
    async def get_chat_id(self) -> str:
        return "unique-id"

    async def execute(self, user_query: str, **kwargs: object) -> str | AgentResponse:
        # Replace with your agent logic
        return "agent response"
```

### HTTP endpoint

```bash
arksim init --agent-type chat_completions
```

Edit `config.yaml` with your agent's Chat Completions endpoint:

```yaml
agent_config:
  agent_type: chat_completions
  agent_name: my-agent
  api_config:
    endpoint: http://localhost:8000/v1/chat/completions
```

### A2A protocol

```bash
arksim init --agent-type a2a
```

Edit `config.yaml` with your agent's A2A endpoint:

```yaml
agent_config:
  agent_type: a2a
  agent_name: my-agent
  api_config:
    endpoint: http://localhost:9999/agent
```

Write scenarios that match your agent's domain. See the [Scenarios documentation](https://docs.arklex.ai/main/build-scenario) for how to define goals, user profiles, and knowledge.

## Why ArkSim?

- **Simulation, not just evaluation.** Most tools score conversations you already have. ArkSim generates them with synthetic users who push back, ask follow-ups, and behave unpredictably.
- **Multi-turn by default.** Every test is a full conversation, not a single prompt. Context loss, tool misuse, and contradictions only show up across turns.
- **Any agent, any framework.** Works with [14+ frameworks](#integrations) through Chat Completions, A2A, or direct Python import.
- **Runs in CI.** Add it as a quality gate on every PR. Exits non-zero when your agent drops below threshold.
- **Fully open source.** Runs on your infrastructure. Your data never leaves.

## Integrations

| Integration | Description |
|-------------|-------------|
| [bank-insurance](examples/bank-insurance/) | Financial services agent with custom compliance metrics, adversarial scenarios, and a Chat Completions server |
| [e-commerce](examples/e-commerce/) | E-commerce product recommendation agent with custom metrics |
| [openclaw](examples/openclaw/) | Integration with the OpenClaw agent framework |
| [claude-agent-sdk](examples/integrations/claude-agent-sdk/) | Integration with the Claude Agent SDK |
| [google-adk](examples/integrations/google-adk/) | Integration with Google ADK |
| [openai-agents-sdk](examples/integrations/openai-agents-sdk/) | Integration with the OpenAI Agents SDK |
| [langchain](examples/integrations/langchain/) | Integration with LangChain |
| [langgraph](examples/integrations/langgraph/) | Integration with LangGraph |
| [crewai](examples/integrations/crewai/) | Integration with CrewAI |
| [dify](examples/integrations/dify/) | Integration with Dify |
| [autogen](examples/integrations/autogen/) | Integration with Microsoft AutoGen |
| [llamaindex](examples/integrations/llamaindex/) | Integration with LlamaIndex |
| [pydantic-ai](examples/integrations/pydantic-ai/) | Integration with Pydantic AI |
| [rasa](examples/integrations/rasa/) | Integration with Rasa |
| [smolagents](examples/integrations/smolagents/) | Integration with Hugging Face Smolagents |
| [mastra](examples/integrations/mastra/) | Integration with Mastra (TypeScript) |
| [vercel-ai-sdk](examples/integrations/vercel-ai-sdk/) | Integration with Vercel AI SDK (TypeScript) |

## Learn More

| Topic | |
|-------|---|
| Evaluation metrics (built-in and custom) | [Metrics guide](https://docs.arklex.ai/main/evaluate-conversation) |
| CI integration (pytest and GitHub Actions) | [CI setup guide](https://docs.arklex.ai/main/ci-integration) |
| Configuration reference (all YAML settings) | [Schema reference](https://docs.arklex.ai/main/schema-reference) |
| Simulation and CLI usage | [Simulation guide](https://docs.arklex.ai/main/simulate-conversation) |
| Web UI for browsing results | [Overview](https://docs.arklex.ai/main/overview) |

## Development

```bash
git clone https://github.com/arklexai/arksim.git
cd arksim
pip install -e ".[dev]"
pytest tests/
```

Linting and formatting:

```bash
ruff check .
ruff format .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation
```bibtex
@misc{shea2026sage,
      title={SAGE: A Top-Down Bottom-Up Knowledge-Grounded User Simulator for Multi-turn AGent Evaluation},
      author={Ryan Shea and Yunan Lu and Liang Qiu and Zhou Yu},
      year={2026},
      eprint={2510.11997},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.11997},
}
```
