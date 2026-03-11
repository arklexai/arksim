#!/usr/bin/env bash
# Integration test for the Claude Agent SDK provider example.
#
# Requires: ANTHROPIC_API_KEY
# Install:  pip install claude-agent-sdk
# Run:      bash tests/integration/providers/test_claude_agent_sdk.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_shared.sh"

require_env "ANTHROPIC_API_KEY"

log "Installing Claude Agent SDK dependencies..."
pip install claude-agent-sdk --quiet

run_provider_test \
    "claude-agent-sdk" \
    "$REPO_ROOT/examples/integrations/claude-agent-sdk/custom_agent.py" \
    "claude-sonnet-4-6" \
    "anthropic"
