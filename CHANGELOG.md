# Changelog

All notable changes to Arksim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- Add your changes here under the appropriate heading. -->
<!-- Use: Added, Changed, Deprecated, Removed, Fixed, Security -->

### Changed

- Refactored `bank-insurance` and `e-commerce` example agents to use the OpenAI Agents SDK and corrected the example server port from `8080` to `8888`
- Agent configuration is now defined inline in `config.yaml` under the `agent_config` key instead of in separate `agent_config.json` files
- Updated all documentation and examples to use inline agent configuration
- UI now passes `agent_config`, `custom_metrics_file_paths`, and `metrics_to_run` from loaded config YAML

### Removed

- Separate `agent_config.json` files from all examples (replaced by inline YAML config)
- Knowledge Configuration and Data folder sections from agent configuration docs

### Security

- Removed third-party client data from test fixtures and evaluator prompt examples
- Replaced with generic insurance-domain examples consistent with the public bank-insurance example

### Fixed
- Example servers: added logging before HTTP error responses, filtered system messages from agent chat history, fixed wrong config filename in e-commerce README
- Documentation: end-of-file newlines in `docs/insurance-customer-service-agent-evaluation.mdx` and `docs/e-commerce-customer-service-agent-evaluation.mdx`
- Documentation: trailing whitespace and end-of-file in `docs/simulate-conversation.mdx` and `docs/evaluate-conversation.mdx`
- Trailing whitespace and missing EOF newlines across 81 files (docs, examples, tests, source)
- UI file browser: `PROJECT_ROOT` now uses `cwd` for PyPI installs, parent navigation works correctly, YAML files shown in browser
- Lint: replaced `try/except/pass` with `contextlib.suppress` in example loader

### Added

- Dependabot for automated pip and GitHub Actions dependency updates
- Pre-commit hooks with ruff, trailing whitespace, end-of-file, YAML, and changelog checks
- `py.typed` marker for PEP 561 type checking support
- Top-level lazy exports (`from arksim import run_simulation, Evaluator, ...`)
- `.editorconfig` for consistent editor settings
- `Makefile` with common dev commands
- Issue template chooser linking to docs and discussions
- Documentation and Changelog checklist in PR template

- CI restructured: separate Lint job, test matrix on Python 3.10 through 3.13, PR checks (title, description, changelog) in one job
- Dependabot PRs now auto-labeled `skip-changelog` to bypass changelog CI check

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
