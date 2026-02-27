---
title: "Contributing"
description: "How to contribute to Arksim: bugs, features, pull requests, and code style."
---

## Getting Started

- **Local development:** Follow the Developer Install section in the README to set up a Python 3.10–3.13 environment and install Arksim in editable mode.
- **Documentation:** For simulator configuration and usage, see the [Arksim docs](/quickstart).

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
Follow the Developer Install instructions in the README.
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
- **Formatting:** 4 spaces, no tabs; no trailing whitespace; single newline at end of file. Run `ruff format .` and `ruff check .` to stay consistent.

---

## Testing

Tests live under `tests/` and run with `pytest`. Mark slow or integration tests with the markers defined in `pytest.ini` — `unit`, `integration`, or `slow` — when relevant.

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

Have questions? Open a [GitHub discussion](https://github.com/arklex/arksim/discussions) or an issue and we'll do our best to help.