# Contributing to ArkSim

Thank you for your interest in contributing to ArkSim! This guide will help you get started.

## Getting Started

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/arksim.git
   cd arksim
   ```

2. Install in editable mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Create a branch for your change:

   ```bash
   git checkout -b my-feature
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/
```

### Linting and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .    # Lint
ruff format .   # Format
```

Pre-commit hooks run both automatically on commit.

### Code Style

- Follow PEP 8 conventions
- Code lines: max 120 characters
- Comments and docstrings: max 80 characters
- Type hints encouraged for function signatures
- Use absolute imports over relative imports

### Commit Messages

```
<component>: <verb> <description>
```

Examples:
- `evaluator: add custom metric support`
- `simulator: fix profile generation for empty attributes`
- `cli: support verbose flag for streaming output`

Keep the subject line under 72 characters, use lowercase, imperative mood.

## Submitting a Pull Request

1. Make sure all tests pass and linting is clean
2. **Use the same format for your PR title** as commit messages (`<component>: <verb> <description>`, max 72 characters). CI enforces this since squash merge uses the PR title as the commit message.
3. Write a clear PR description explaining what changed and why
4. **Update the changelog**: Add an entry under `## [Unreleased]` in `CHANGELOG.md` describing your change. Use the appropriate heading (`Added`, `Changed`, `Fixed`, `Removed`, etc.). If your change is purely internal with no user-facing impact, note that in the PR instead.
5. **Update docs if needed**: If your PR changes behavior, config options, CLI commands, or the Python API, update the relevant files in `docs/` to match.
6. Link any related issues
7. Keep PRs focused on a single change

## Claude Code (optional)

This repo includes [Claude Code](https://claude.ai/code) skills for contributors:

- `/pre-review` - Check your branch before opening a PR (runs lint, tests, validates title)
- `/draft-pr` - Generate a PR title and description from your changes
- `/review-pr <number>` - Review a PR against project standards

These are optional. You can contribute without Claude Code.

## Reporting Issues

Open an issue at [github.com/arklexai/arksim/issues](https://github.com/arklexai/arksim/issues) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
