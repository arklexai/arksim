---
name: draft-pr
description: Generate a PR title and description from your changes
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

# Draft PR

Analyze commits and diff to produce a ready-to-use PR title and description.

## Steps

1. **Analyze commits and diff**
   ```bash
   git log --oneline main..HEAD
   git diff --stat main..HEAD
   git diff main..HEAD
   ```

2. **Determine type and scope**
   - Infer type from changes (feat, fix, docs, chore, etc.)
   - Infer scope from primary directory affected
   - If multiple areas changed equally, omit scope

3. **Validate title**
   - Must match: `^(feat|fix|docs|chore|ci|build|refactor|test|perf|style|revert)(\([a-z][a-z0-9_-]*\))?!?: .+$`
   - Max 72 characters
   - Use imperative mood, lowercase after colon

4. **Read and fill the PR template**

   Read `.github/pull_request_template.md` and fill in each section with real content from the commits and diff. Strip all HTML comments and placeholder text.

5. **Validate description**
   - Ensure at least 20 characters of real content (excluding template boilerplate)

## Output

Print the title and filled PR template body in raw markdown format, ready to copy:

```
Title: <type>(<scope>): <description>

<filled-out PR template in raw markdown>
```
