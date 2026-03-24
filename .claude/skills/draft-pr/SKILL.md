---
name: draft-pr
description: Generate a PR title and description from your changes
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Draft PR

Analyze commits and diff to produce a ready-to-use PR title and description.

## Steps

1. **Analyze commits and diff**
   ```bash
   BASE="main"
   git log --oneline "$BASE"..HEAD
   git diff --stat "$BASE"..HEAD
   git diff "$BASE"..HEAD
   ```

2. **Determine type and scope**
   - Infer type from changes (feat, fix, docs, chore, etc.)
   - Infer scope from primary directory affected
   - If multiple areas changed equally, omit scope

3. **Validate title**
   - Must match: `^(feat|fix|docs|chore|ci|build|refactor|test|perf|style|revert)(\([a-z][a-z0-9_-]*\))?!?: .+$`
   - Max 72 characters
   - Use imperative mood, lowercase after colon

4. **Read PR template**
   ```bash
   cat .github/pull_request_template.md
   ```

5. **Fill out the template**
   - Summary: describe what the PR does and why
   - Changes: list key modifications as bullet points
   - Documentation: check the appropriate box
   - How to Test: fill in verification steps
   - Notes: mention trade-offs or follow-up work if relevant

6. **Check changelog**
   - Remind to add entry in CHANGELOG.md under `[Unreleased]`
   - Suggest the entry text based on the changes

7. **Validate description**
   - Ensure at least 20 characters of real content (excluding template boilerplate)

## Output

Print the title and full PR body, ready to copy:

```
Title: <type>(<scope>): <description>

Body:
<filled-out PR template>
```

Remind about CHANGELOG.md if no entry was found.
