# arksim

Agent simulation and evaluation framework. Python 3.10+, Apache-2.0.

## Project structure

Key directories: `arksim/` (source), `tests/unit/` (CI tests), `tests/integration/`,
`docs/` (Mintlify), `examples/`, `integration-tests/`.

## Dev setup

```bash
pip install -e ".[dev]"
pre-commit install
```

Makefile: `make install-dev`, `make lint`, `make format`, `make test`, `make clean`.

## Code style

See @.claude/rules/python-style.md for all Python rules (auto-enforced by hooks).

## PR conventions

PRs are squash-merged. The PR title becomes the commit on main.

Title regex (CI-enforced):
```
^(feat|fix|docs|chore|ci|build|refactor|test|perf|style|revert)(\([a-z][a-z0-9_-]*\))?!?: .+$
```

- Max 72 characters. Scope recommended but optional.
- Common mistakes: using a component name (e.g. `evaluator:`) as the type, putting ticket IDs in the title.
- Description body must have 20+ characters of real content.
- Update CHANGELOG.md under `[Unreleased]` (Added, Changed, Fixed, Removed).

## Testing

```bash
pytest tests/unit/          # What CI runs
pytest tests/ --cov=arksim  # Full suite with coverage
```

- pytest with pytest-asyncio (auto mode). Markers: `unit`, `integration`, `slow`.
- Coverage minimum: 60%. CI matrix: Python 3.10-3.13.

## CI checks

PR title format, PR description (20 char min), pre-commit hooks, ruff check + format,
bandit security scan, CodeQL, pytest across 3.10-3.13, Codecov upload.

## Hooks (.claude/settings.json)

- **SessionStart**: prints current branch and any open PR for context.
- **PreToolUse (Edit|Write)**: blocks edits to `.env` files and `secrets/`/`credentials/`/`keys/` dirs.
- **PostToolUse (Edit|Write)**: auto-runs `ruff format` + `ruff check --fix` on saved `.py` files.

## Documentation

Mintlify docs live in `docs/`. Use relative links within version folders.
Do not edit versioned snapshot folders (`docs/v0.x.x/`).

## Writing style

- No em dashes or en dashes in prose. Use hyphens only where grammatically required.
- No AI attribution in commits, PRs, docs, or any artifacts.

## Branch naming

```
<type>/<short-description>
```

Examples: `feat/retry-logic`, `fix/empty-list-handling`, `docs/update-quickstart`.
No ticket IDs in branch names (this is an open-source repo).

## What NOT to do

- Do not commit `.env` files, secrets, or credentials.
- Do not edit versioned doc snapshot folders (`docs/v0.x.x/`).
- Do not bundle unrelated changes in a single PR.
- Do not add files over 500KB without discussion.
