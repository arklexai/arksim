#!/usr/bin/env bash
# Integration test for the LangChain/LangGraph agent provider example.
#
# Requires: OPENAI_API_KEY
# Install:  pip install langgraph langchain-openai
# Run:      bash tests/integration/providers/test_langchain.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_shared.sh"

require_env "OPENAI_API_KEY"

log "Installing LangChain dependencies..."
pip install langgraph langchain-openai --quiet

run_shell_test \
    "langchain" \
    "$REPO_ROOT/examples/integrations/langchain/custom_agent.py" \
    "gpt-5.1" \
    "openai"
