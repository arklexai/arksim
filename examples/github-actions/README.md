# GitHub Actions Integration

Run ArkSim as a quality gate in your CI pipeline. The workflow starts your agent as an HTTP server, simulates conversations against it, evaluates the results, and fails the job if configured thresholds are not met.

## Quickstart

### 1. Copy the workflow template into your repo

```bash
mkdir -p .github/workflows
cp arksim.yml .github/workflows/arksim.yml
```

### 2. Create your ArkSim config

Create `arksim/config.yaml` pointing to your agent server. The `endpoint` must match the port you start in the workflow:

```yaml
agent_config:
  agent_type: chat_completions
  agent_name: my-agent
  api_config:
    endpoint: http://localhost:8000/v1/chat/completions
    headers:
      Content-Type: application/json
    body:
      model: my-model
      messages: []

scenario_file_path: ./arksim/scenarios.json
num_conversations_per_scenario: 3
max_turns: 5
output_file_path: ./arksim/results/simulation.json

output_dir: ./arksim/results/evaluation
metrics_to_run:
  - faithfulness
  - helpfulness
  - coherence
  - relevance
  - goal_completion
  - agent_behavior_failure
generate_html_report: true

score_threshold: 0.7
numeric_thresholds:
  goal_completion: 0.8

model: gpt-4o
provider: openai
num_workers: 10
```

### 3. Create your scenarios

Create `arksim/scenarios.json` with test cases specific to your agent. See [`examples/bank-insurance`](../bank-insurance) for a full example of the scenario format.

### 4. Customize the workflow

Open `.github/workflows/arksim.yml` and update the `TODO` sections:

| Step | What to change |
|------|----------------|
| **Start agent server** | Your framework's startup command and port |
| **Wait for agent** | Health-check URL (default: `http://localhost:8000/health`) |
| **Run ArkSim** | Path to your config file if different from `arksim/config.yaml` |

### 5. Add GitHub secrets

In your repo → **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|--------|---------|
| `OPENAI_API_KEY` | LLM ArkSim uses to evaluate your agent |
| `AGENT_API_KEY` | *(optional)* API key your agent server needs |

### 6. Push

The workflow runs automatically on every push to `main` and on every PR.

---

## How it works

```
GitHub Actions runner
│
├── 1. Checkout + install dependencies
├── 2. pip install arksim
├── 3. Start your agent as an HTTP server (background)
├── 4. Poll /health until HTTP 200
├── 5. arksim simulate-evaluate arksim/config.yaml
│       ├── Simulate N conversations with your agent
│       ├── Evaluate each conversation with an LLM judge
│       └── Check configured quality thresholds
├── 6. Upload HTML evaluation report as artifact  (always runs)
└── 7. Exit code → job passes or fails
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | All evaluations passed |
| `1` | Evaluation failed — a threshold was not met |
| `2` | Config / usage error |
| `3` | Internal / engine error |

## Quality gates

Configure pass/fail thresholds in `arksim/config.yaml`:

```yaml
# Overall: fail if any conversation's overall_agent_score < 0.7
score_threshold: 0.7

# Per-metric: fail if any conversation's score falls below these minimums.
# Built-in turn-level metrics (faithfulness, helpfulness, coherence, relevance)
# use a 1–5 scale; goal_completion uses 0–1.
numeric_thresholds:
  goal_completion: 0.8
  faithfulness: 3.5

# Qualitative: fail if any evaluated turn returns one of these labels
qualitative_failure_labels:
  agent_behavior_failure: ["false information", "disobey user request"]
```

## Viewing the evaluation report

After every run (pass or fail) the full HTML report is uploaded as a build artifact:

1. Open the workflow run in GitHub Actions.
2. Scroll to **Artifacts** at the bottom.
3. Download **arksim-evaluation-report**.

## Agent framework examples

**FastAPI / uvicorn:**
```yaml
- name: Start agent server
  run: uvicorn my_agent:app --host 0.0.0.0 --port 8000 &
```

**LangChain + LangServe:**
```yaml
- name: Start agent server
  run: python -m uvicorn my_chain_server:app --port 8000 &
```

**OpenAI Agents SDK:**
```yaml
- name: Start agent server
  run: python my_agent_server.py --port 8000 &
```

**Google ADK:**
```yaml
- name: Start agent server
  run: adk api_server --port 8000 &
```

## Files

| File | Description |
|------|-------------|
| `arksim.yml` | GitHub Actions workflow template — copy to `.github/workflows/` |
