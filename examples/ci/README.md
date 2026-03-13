# CI Integration

Run ArkSim as a quality gate in your CI pipeline. Choose the approach that matches how your agent is built.

## Approaches

### Approach 1: pytest (custom agent)

Your agent is a Python class subclassing `BaseAgent`. ArkSim loads it in-process — no HTTP server needed. CI simply runs `pytest`.

**Copy the templates:**

```bash
mkdir -p .github/workflows tests
cp pytest/arksim-pytest.yml .github/workflows/arksim-pytest.yml
cp pytest/test_agent_quality.py tests/test_agent_quality.py
```

**Then:**
1. Implement `BaseAgent` in your agent module (see comments in `test_agent_quality.py`)
2. Update the import and thresholds in `tests/test_agent_quality.py`
3. Create `arksim/scenarios.json` with your test cases
4. Add `OPENAI_API_KEY` to GitHub secrets

### Approach 2: HTTP server (GitHub Actions)

Your agent runs as an HTTP server. ArkSim calls it over HTTP during CI — works with any language or framework.

**Copy the template:**

```bash
mkdir -p .github/workflows
cp github-actions/arksim.yml .github/workflows/arksim.yml
```

**Then:**
1. Update the `TODO` sections in `arksim.yml` (startup command and health-check URL)
2. Create `tests/arksim/config.yaml` pointing to your server endpoint
3. Create `tests/arksim/scenarios.json` with your test cases
4. Add custom metrics to `tests/arksim/custom_metrics/` if needed and reference them in `tests/arksim/config.yaml`
5. Add `OPENAI_API_KEY` (and optionally `AGENT_API_KEY`) to GitHub secrets

See [`examples/bank-insurance`](../bank-insurance) for a full example of the scenario format.

---

## Files

| File | Description |
|------|-------------|
| `github-actions/arksim.yml` | HTTP server workflow — copy to `.github/workflows/` |
| `pytest/arksim-pytest.yml` | pytest workflow — copy to `.github/workflows/` |
| `pytest/test_agent_quality.py` | pytest test template — copy to `tests/` |
