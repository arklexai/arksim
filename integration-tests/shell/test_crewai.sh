#!/usr/bin/env bash
# Integration test for the CrewAI agent provider example.
#
# Requires: OPENAI_API_KEY
# Install:  pip install crewai
# Run:      bash tests/integration/providers/test_crewai.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_shared.sh"

require_env "OPENAI_API_KEY"

log "Installing CrewAI dependencies..."
pip install crewai --quiet

run_provider_test \
    "crewai" \
    "$REPO_ROOT/examples/integrations/crewai/custom_agent.py" \
    "gpt-5.1" \
    "openai"
