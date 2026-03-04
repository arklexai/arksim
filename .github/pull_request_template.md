<!--
  PR title must follow Conventional Commits: <type>(<scope>): <description>

  Type        | When to use                          | Changelog section
  ----------- | ------------------------------------ | -----------------
  feat        | New user-facing feature               | Added
  fix         | Bug fix                              | Fixed
  docs        | Documentation only                   | Documentation
  refactor    | Code restructuring, no behavior change| Changed
  chore       | Maintenance, deps, CI config         | Changed
  ci          | CI/CD pipeline changes               | Changed
  build       | Build system or dependency changes    | Changed
  perf        | Performance improvement              | Changed
  test        | Adding or updating tests             | Changed
  style       | Formatting, whitespace               | Changed
  revert      | Reverting a previous commit          | Reverted

  Append ! for breaking changes: feat!: remove legacy API
  Scope is optional: fix(evaluator): handle empty list
-->

## Summary

<!-- Brief description of what this PR does and why. Link any related issues. -->

Closes #

## Changes

<!-- What changed? List the key modifications. -->

-

## Documentation

<!-- Check every box that applies. -->

- [ ] Updated relevant docs in `docs/` (if behavior, config, or API changed)
- [ ] Updated `README.md` (if installation, quickstart, or usage changed)
- [ ] No docs needed (explain why below)

<!-- If "no docs needed", briefly explain: e.g. "internal refactor, no user-facing change" -->

## How to Test

<!-- Steps a reviewer can follow to verify the changes. -->

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `pytest tests/` passes
- [ ] Manual verification: <!-- describe what you tested -->

## Notes

<!-- Anything reviewers should know: trade-offs, follow-up work, migration steps, etc. -->

## Reviewers

/cc @arklexai/arksim-maintainers
