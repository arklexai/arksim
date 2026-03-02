# Changelog

All notable changes to Arksim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- Add your changes here under the appropriate heading. -->
<!-- Use: Added, Changed, Deprecated, Removed, Fixed, Security -->

### Fixed

- Trailing whitespace and missing EOF newlines across 81 files (docs, examples, tests, source)

### Changed

- CI restructured: separate Lint job, test matrix on Python 3.10 through 3.13, PR checks (title, description, changelog) in one job

### Added

- Dependabot for automated pip and GitHub Actions dependency updates
- Pre-commit hooks with ruff, trailing whitespace, end-of-file, YAML, and changelog checks
- `py.typed` marker for PEP 561 type checking support
- Top-level lazy exports (`from arksim import run_simulation, Evaluator, ...`)
- `.editorconfig` for consistent editor settings
- `Makefile` with common dev commands
- Issue template chooser linking to docs and discussions
- Documentation and Changelog checklist in PR template

## [0.0.1] - 2026-03-02

### Added

- Multi-turn conversation simulation with configurable scenarios
- Quantitative and qualitative evaluation metrics
- Chat Completions and A2A agent protocol support
- CLI (`arksim simulate`, `arksim evaluate`, `arksim simulate-evaluate`)
- Python SDK with `run_simulation()` and `run_evaluation()`
- Parallel execution with configurable worker count
- Example setups for e-commerce and bank insurance use cases
- OpenAI, Anthropic, Google Gemini, and Azure OpenAI provider support
- HTML report generation for evaluation results

[Unreleased]: https://github.com/arklexai/arksim/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/arklexai/arksim/releases/tag/v0.0.1
