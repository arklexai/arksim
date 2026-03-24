---
name: review-pr
description: Review a pull request against arksim project standards
disable-model-invocation: true
effort: high
argument-hint: "[PR-number]"
context: fork
agent: Explore
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Review PR

Review an open pull request for arksim standards compliance.

## Inputs

- PR number (required, passed as argument)

## Steps

1. **Fetch PR metadata**
   ```bash
   gh pr view <number> --json title,body,files,additions,deletions,commits,headRefName,baseRefName
   ```

2. **Validate PR title**
   - Must match: `^(feat|fix|docs|chore|ci|build|refactor|test|perf|style|revert)(\([a-z][a-z0-9_-]*\))?!?: .+$`
   - Max 72 characters
   - Scope is recommended but optional
   - Common mistakes: using a component name as type, including ticket IDs

3. **Validate PR description**
   - Strip HTML comments, markdown headers, empty checkboxes
   - Remaining content must be at least 20 characters

4. **Check scope recommendation**
   - If no scope in title, suggest one based on changed file paths

5. **Review changed Python files**
   - Fetch the diff: `gh pr diff <number>`
   - For each new `.py` file, verify:
     - First line is `# SPDX-License-Identifier: Apache-2.0`
     - Has `from __future__ import annotations`
     - Functions have type annotations
     - Uses absolute imports (no relative imports)

6. **Check test coverage**
   - If source files under `arksim/` changed, check for corresponding test changes
   - Flag if no tests added or modified for new features

7. **Check documentation**
   - If behavior, config, or API changed, verify `docs/` updates
   - Check for CHANGELOG.md entry under `[Unreleased]`

8. **Check branch naming**
   - Should follow `<type>/<short-description>` pattern

## Output

Print a summary table:

| Check | Status | Notes |
|-------|--------|-------|
| Title format | PASS/FAIL | ... |
| Title length | PASS/FAIL | ... |
| Description | PASS/FAIL | ... |
| License headers | PASS/FAIL/N/A | ... |
| Future annotations | PASS/FAIL/N/A | ... |
| Type hints | PASS/WARN/N/A | ... |
| Absolute imports | PASS/FAIL/N/A | ... |
| Tests | PASS/WARN | ... |
| Docs/Changelog | PASS/WARN | ... |
| Branch naming | PASS/WARN | ... |

End with an overall verdict: APPROVED, CHANGES REQUESTED, or NEEDS DISCUSSION.
