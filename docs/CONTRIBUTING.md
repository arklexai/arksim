---
title: "Contributing"
description: "How to contribute to ArkSim: bugs, features, pull requests, and code style."
---

### Contributing

## Getting Started

- **Local development:** Follow the Developer Install section in the README to set up a Python 3.10–3.13 environment and install ArkSim in editable mode.
- **Documentation:** For simulator configuration and usage, see the [ArkSim docs](https://doc.arklex.ai/).

## Repository Layout

- `arksim/` Core package: CLI, simulation engine, evaluators, config, LLMs, scenario handling, and Web UI.
- `examples/` Example configs and runnable setups (e.g. openclaw, e-commerce, bank-insurance).
- `tests/` Unit and integration tests; test data lives under `tests/test_data/`.

---

## How to Contribute

### Bugs and Features

We use GitHub issues to track bugs and feature ideas. Before opening a new issue, search existing issues to avoid duplicates.

### Pull Requests

<Steps>
<Step title="Fork and branch">
Fork the repository and create a branch from `main`.
</Step>

<Step title="Set up your environment">
Follow the Developer Install instructions in the README, then install pre-commit hooks:
```bash
make install-dev   # installs editable package + pre-commit hooks
```
</Step>

<Step title="Make your changes">
Scope changes to the area you're modifying. Smaller, focused PRs are easier to review and merge. If your change is large, consider splitting it into logical PRs.
</Step>

<Step title="Add or update tests">
If you fix a bug or add behavior that can be tested, add or update tests in `tests/`.
</Step>

<Step title="Lint and format">
```bash
ruff check .
ruff format .
```
</Step>

<Step title="Run the test suite">
```bash
pytest
```
</Step>

<Step title="Open a pull request">
Open a pull request against `main`. See PR description guidelines below.
</Step>
</Steps>

### PR Description Guidelines

- **Title:** Use a clear, descriptive title. Conventional-style is welcome, e.g. `fix(evaluator): handle empty response`.
- **Issue:** If the PR addresses an issue, reference it in the description, e.g. `Fixes #123`.
- **Summary:** Start with a short summary and add context, rationale, and any caveats below.
- **UI changes:** For changes to the Web UI, include screenshots or a short video when helpful.

---

## Commit Conventions

Format: `<component>: <verb> <description>`
```
evaluator: add score threshold option
simulator: fix config validation
```

For general repo changes (docs, CI, etc.):
```
fix typo in readme
```

- Keep the subject line under 72 characters.
- Use lowercase and imperative mood, e.g. "add" not "added".

---

## Code Style

- **Line length:** Code up to 120 characters; comments and docstrings up to 80 characters.
- **Python:** Follow PEP 8. Use type hints on function signatures where practical.
- **Imports:** Group stdlib, third-party, and local imports separated by a blank line. Prefer absolute imports.
- **Naming:** `PascalCase` for classes, `snake_case` for functions and variables, `SCREAMING_SNAKE_CASE` for constants; prefix private names with `_`.
- **License header:** Every `.py` file must start with `# SPDX-License-Identifier: Apache-2.0`. The `insert-license` pre-commit hook adds it automatically.
- **Future annotations:** Every `.py` file must include `from __future__ import annotations` (enforced by ruff rule FA100).
- **Formatting:** 4 spaces, no tabs; no trailing whitespace; single newline at end of file. Run `ruff format .` and `ruff check .` to stay consistent.

---

## Testing

Tests live under `tests/` and run with `pytest`. Mark slow or integration tests with the markers defined in `pytest.ini` when relevant.

### Quality gates

| Gate | Current | Direction |
|------|---------|-----------|
| Test coverage (line) | 60% minimum | Ratchet upward as modules gain coverage |
| Ruff lint + format | Required in CI and pre-commit | Add rules incrementally |
| Bandit security scan | Required in CI | Expand as surface area grows |

Coverage is enforced in both `pyproject.toml` (`fail_under`) and the CI workflow (`--cov-fail-under`). When you add tests that push coverage above the current threshold, please bump both values so they stay in sync and the bar never drops.

---

## Code Review

Reviews focus on design, correctness, simplicity, tests, naming, and documentation. Please:

- Be kind and constructive.
- Explain your reasoning when requesting changes.
- Prefer suggesting simplifications over only listing problems.

---

## AI-Assisted Contributions

AI-assisted contributions are welcome. Please:

- Run the relevant linters and tests before submitting.
- Review the diff and any generated code for correctness and style.
- Do not include secrets, API keys, or sensitive data in commits or PR descriptions.
- Ensure you have the right to submit the contribution (e.g. licensing, employer policies).

---

Have questions? Open a [GitHub discussion](https://github.com/arklexai/arksim/discussions) or an issue and we'll do our best to help.
