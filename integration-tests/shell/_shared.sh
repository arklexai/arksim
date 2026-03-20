#!/usr/bin/env bash
# Shared utilities for provider integration tests.
# Source this file from each provider test script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# require_env VAR
# Exits 1 with a FAIL message if the env var is not set
# ---------------------------------------------------------------------------
require_env() {
    local var="$1"
    if [[ -z "${!var:-}" ]]; then
        log "FAIL: $var is not set — cannot run test"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Minimal scenarios.json (1 scenario, keeps API calls low)
# ---------------------------------------------------------------------------
write_scenarios() {
    local dest="$1"
    cat > "$dest" << 'EOF'
{
  "schema_version": "v1",
  "scenarios": [
    {
      "scenario_id": "provider_smoke_test",
      "user_id": "user_001",
      "goal": "You want to learn what types of home insurance coverage are available for a new homeowner.",
      "agent_context": "XYZ Insurance provides home, auto, and travel insurance. We offer comprehensive and basic coverage plans.",
      "user_profile": "You are Alex, a 28-year-old first-time homeowner. You are friendly and curious but have no prior insurance knowledge.",
      "knowledge": [
        {
          "content": "Home insurance covers damage from fire, theft, and natural disasters. Basic coverage starts at $500/year."
        }
      ],
      "origin": {
        "goal_raw": "Learn about home insurance",
        "target_agent_capability": "explain insurance products"
      }
    }
  ]
}
EOF
}

# ---------------------------------------------------------------------------
# write_config DEST_DIR MODULE_PATH MODEL PROVIDER
# Writes a minimal config.yaml. module_path should be ./custom_agent.py
# when the agent file has been copied alongside config.yaml.
# ---------------------------------------------------------------------------
write_config() {
    local dest_dir="$1"
    local model="$2"
    local provider="$3"
    cat > "$dest_dir/config.yaml" << EOF
agent_config:
  agent_type: custom
  agent_name: provider-smoke-test
  custom_config:
    module_path: ./custom_agent.py

scenario_file_path: ./scenarios.json
num_conversations_per_scenario: 1
max_turns: 5
output_file_path: ./results/simulation/simulation.json
model: $model
provider: $provider
num_workers: 5
EOF
}

# ---------------------------------------------------------------------------
# validate_output PATH
# Asserts that the simulation output is valid JSON with at least 1 conversation.
# ---------------------------------------------------------------------------
validate_output() {
    local sim_file="$1"
    if [[ ! -f "$sim_file" ]]; then
        log "FAIL: output file not found: $sim_file"
        exit 1
    fi
    python3 - "$sim_file" << 'PYEOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
assert data.get("schema_version") == "v1.1", f"Bad schema_version: {data.get('schema_version')}"
convos = data.get("conversations", [])
assert len(convos) >= 1, f"Expected >= 1 conversation, got {len(convos)}"
for c in convos:
    history = c.get("conversation_history", [])
    assert len(history) >= 2, f"Expected >= 2 messages, got {len(history)}"
    assert history[0].get("content"), "First message content is empty"
print(f"OK: {len(convos)} conversation(s), {len(history)} messages in last convo")
PYEOF
}

# ---------------------------------------------------------------------------
# run_shell_test NAME CUSTOM_AGENT_SRC MODEL PROVIDER
# Orchestrates temp dir setup, simulate, and validation.
# ---------------------------------------------------------------------------
run_shell_test() {
    local name="$1"
    local agent_src="$2"
    local model="$3"
    local provider="$4"

    local tmp="$(mktemp -d)"
    trap "rm -rf $tmp" EXIT

    log "Setting up temp dir: $tmp"
    cp "$agent_src" "$tmp/custom_agent.py"
    write_scenarios "$tmp/scenarios.json"
    write_config "$tmp" "$model" "$provider"
    mkdir -p "$tmp/results/simulation"

    log "Running: arksim simulate config.yaml (provider=$name, model=$model)"
    (cd "$tmp" && arksim simulate config.yaml)

    log "Validating output..."
    validate_output "$tmp/results/simulation/simulation.json"
    log "PASS: $name integration test"
}
