#!/usr/bin/env bash
# Pre-commit hook: remind contributors to update CHANGELOG.md
# Skip with: git commit --no-verify

if git diff --cached --name-only | grep -q "^CHANGELOG.md$"; then
    exit 0
fi

echo "CHANGELOG.md not updated in this commit."
echo "Please add an entry under [Unreleased] in CHANGELOG.md."
echo "Skip this check with: git commit --no-verify"
exit 1
