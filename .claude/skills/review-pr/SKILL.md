---
name: review-pr
description: Review a pull request against arksim project standards
disable-model-invocation: true
effort: high
argument-hint: "[PR-number]"
context: fork
agent: Explore
allowed-tools: Bash, Read, Grep, Glob
---

# Review PR

Review an open pull request for functional correctness and project conventions. Focus on logic, architecture, and behavior. Formatting issues are caught by CI (ruff, pre-commit) and do not need manual review.

## Steps

1. **Fetch PR metadata and diff**
   ```bash
   gh pr view $ARGUMENTS --json title,body,files,additions,deletions,commits,headRefName
   gh pr diff $ARGUMENTS
   ```

2. **Validate PR title and description**
   - Title matches Conventional Commits regex, under 72 chars
   - Description has 20+ characters of real content
   - These are also CI-enforced, so flag only if CI somehow missed them

3. **Review functional changes**
   Focus on:
   - Logic correctness: does the code do what the PR claims?
   - Edge cases: are error paths handled?
   - API changes: are they backwards compatible or properly marked as breaking?
   - Security: no hardcoded secrets, no unsanitized input in dangerous contexts
   - Performance: any obvious inefficiencies (N+1 queries, unbounded loops)?

4. **Check scope**
   - Single concern per PR
   - Flag unrelated changes bundled together

5. **Check test coverage**
   - New behavior should have tests
   - Flag large additions to `arksim/` with no corresponding test changes

6. **Check documentation**
   - If behavior, config, or API changed, verify `docs/` updates
   - Check for CHANGELOG.md entry under `[Unreleased]`

7. **Check license headers on new files**
   - New `.py` files should start with `# SPDX-License-Identifier: Apache-2.0`

## Output

| Check | Status | Notes |
|-------|--------|-------|
| Title format | PASS/FAIL | ... |
| Description | PASS/FAIL | ... |
| Functional correctness | PASS/WARN | ... |
| Scope | PASS/WARN | ... |
| Tests | PASS/WARN | ... |
| Docs/Changelog | PASS/WARN | ... |
| License headers | PASS/N/A | ... |

End with: APPROVED, CHANGES REQUESTED, or NEEDS DISCUSSION.
