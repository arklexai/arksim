#!/usr/bin/env bash
# Integration test for the LlamaIndex agent provider example.
#
# Requires: OPENAI_API_KEY
# Install:  pip install llama-index llama-index-llms-openai
# Run:      bash tests/integration/providers/test_llamaindex.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_shared.sh"

require_env "OPENAI_API_KEY"

log "Installing LlamaIndex dependencies..."
pip install llama-index llama-index-llms-openai --quiet

run_provider_test \
    "llamaindex" \
    "$REPO_ROOT/examples/integrations/llamaindex/custom_agent.py" \
    "gpt-5.1" \
    "openai"
