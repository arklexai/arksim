#!/usr/bin/env bash
# Integration test for the Google ADK agent provider example.
#
# Requires: GOOGLE_API_KEY
# Install:  pip install google-adk
# Run:      bash tests/integration/providers/test_google_adk.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_shared.sh"

require_env "GOOGLE_API_KEY"

log "Installing Google ADK dependencies..."
pip install google-adk --quiet

run_shell_test \
    "google-adk" \
    "$REPO_ROOT/examples/integrations/google-adk/custom_agent.py" \
    "gemini-3-flash-preview" \
    "google"
