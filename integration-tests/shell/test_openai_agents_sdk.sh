#!/usr/bin/env bash
# Integration test for the OpenAI Agents SDK provider example.
#
# Requires: OPENAI_API_KEY
# Install:  pip install openai-agents
# Run:      bash tests/integration/providers/test_openai_agents_sdk.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_shared.sh"

require_env "OPENAI_API_KEY"

log "Installing OpenAI Agents SDK dependencies..."
pip install openai-agents --quiet

run_provider_test \
    "openai-agents-sdk" \
    "$REPO_ROOT/examples/integrations/openai-agents-sdk/custom_agent.py" \
    "gpt-5.1" \
    "openai"
