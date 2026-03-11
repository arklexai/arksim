#!/usr/bin/env bash
# Run all shell integration tests and report a summary.
#
# Tests that are missing their required API key are counted as FAIL.
#
# Usage:
#   bash integration-tests/shell/run_all.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    test_langchain.sh
    test_openai_agents_sdk.sh
    test_crewai.sh
    test_llamaindex.sh
    test_claude_agent_sdk.sh
    test_google_adk.sh
)

pass=0
fail=0
failed_names=()

for script in "${SCRIPTS[@]}"; do
    name="${script%.sh}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    set +e
    output=$(bash "$SCRIPT_DIR/$script" 2>&1)
    exit_code=$?
    set -e

    echo "$output"

    if [[ $exit_code -eq 0 ]]; then
        echo "  → PASSED"
        ((pass++))
    else
        echo "  → FAILED (exit code $exit_code)"
        ((fail++))
        failed_names+=("$name")
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: ${pass} passed, ${fail} failed"
if [[ ${#failed_names[@]} -gt 0 ]]; then
    echo "  Failed:  ${failed_names[*]}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ $fail -eq 0 ]]
