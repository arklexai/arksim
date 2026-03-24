---
name: pre-review
description: Check your branch before opening a PR (lint, test, title validation)
disable-model-invocation: true
context: fork
agent: Explore
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Pre-review

Run local checks on the current branch before opening a pull request.

## Steps

1. **Gather branch context**
   ```bash
   BRANCH=$(git branch --show-current)
   BASE="main"
   git log --oneline "$BASE"..HEAD
   git diff --stat "$BASE"..HEAD
   ```

2. **Group changed files by area**
   - List files changed vs main
   - Categorize: source (`arksim/`), tests (`tests/`), docs (`docs/`), config, other

3. **Draft and validate title**
   - Infer type and scope from changes
   - Propose a title matching: `^(feat|fix|docs|chore|ci|build|refactor|test|perf|style|revert)(\([a-z][a-z0-9_-]*\))?!?: .+$`
   - Verify max 72 characters
   - Recommend a scope if omitted

4. **Run ruff check**
   ```bash
   ruff check . 2>&1
   ```

5. **Run ruff format check**
   ```bash
   ruff format --check . 2>&1
   ```

6. **Run unit tests**
   ```bash
   pytest tests/unit/ -x --tb=short 2>&1
   ```

7. **Check code quality in diff**
   - For new `.py` files in the diff, verify:
     - `# SPDX-License-Identifier: Apache-2.0` header present
     - `from __future__ import annotations` present
     - Type annotations on function signatures
     - Absolute imports only (no relative imports)

8. **Check changelog**
   - If source files changed, check for an entry in CHANGELOG.md under `[Unreleased]`

## Output

Print a readiness checklist:

```
Pre-review checklist
--------------------
[ ] Branch: <branch-name>
[x] Ruff check: PASS / FAIL (N issues)
[x] Ruff format: PASS / FAIL (N files)
[x] Unit tests: PASS / FAIL (N passed, N failed)
[x] License headers: PASS / FAIL / N/A
[x] Future annotations: PASS / FAIL / N/A
[x] Type hints: PASS / WARN / N/A
[x] Absolute imports: PASS / FAIL / N/A
[x] Changelog updated: YES / NO / N/A
[x] Suggested title: <title>

Ready to open PR: YES / NO
```
