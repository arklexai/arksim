# Changelog

All notable changes to Arksim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `arksim examples` CLI command to download example projects from GitHub without cloning
- Bandit security scanning in CI pipeline
- Test coverage threshold (60% minimum) enforced in CI
- 26 new unit test files covering evaluator metrics, CLI utilities, LLM factory, concurrency workers, simulation utilities, error detection, API endpoints, and path validation
- `from __future__ import annotations` to all source files for consistent typing
- Ruff `FA100` rule to enforce `from __future__ import annotations` on every Python file
- `insert-license` pre-commit hook to enforce `SPDX-License-Identifier: Apache-2.0` headers
- Path traversal protection on filesystem and results API endpoints
- Generic error messages in API endpoints to prevent server path leakage
- Shared evaluation constants (`SCORE_NOT_COMPUTED`, `BEHAVIOR_FAILURE_THRESHOLD`)
- Quality gates section in CONTRIBUTING.md documenting coverage ratchet policy
- `codecov.yml` with patch target of 50% to avoid false-positive CI failures on infra changes

### Changed

- `SCORE_NOT_COMPUTED` display label changed from "N/A (Evaluation Failed)" to "N/A (Not computed)"
- Removed stale `{chat_id}` placeholder reference from `ChatCompletionsConfig.body` field description

- UI auto-loads scenarios from config on startup instead of always showing the demo
- File browser shows a hint that browsing is scoped to the launch directory
- Coverage badge switched from Codecov-hosted to shields.io for reliability
- Expanded SECURITY.md with response process, scope, and disclosure guidelines
- Azure OpenAI provider now raises `ValueError` instead of silently returning a raw string when structured output parsing fails
- `validate_num_workers` rejects zero and negative values

### Removed

- Project root path from the UI sidebar header
- Dead code: unused error message constants, `METRIC_THRESHOLD`, `UNIQUE_BUGS` enum, `flip_hist_content_only`, `LLMConfig`
- Hidden Unicode characters (U+200C zero-width non-joiner) from e-commerce example data files

## [0.0.4] - 2026-03-03

### Added

- SPDX license headers (`Apache-2.0`) to all Python source files
- Test coverage reporting with `pytest-cov` and Codecov integration (with token authentication)
- Coverage badge in README
- CodeQL security scanning workflow (on push, PR, and weekly schedule)
- Apache `NOTICE` file for third-party dependency attributions

### Changed

- PyPI publish workflow now shows a human-readable run name (e.g. "Publish v1.2.3 to PyPI") in GitHub Actions
- PyPI version and Python version README badges now use a 5-minute cache TTL (`cacheSeconds=300`) so they update promptly after a release
- `user_profile` is now the primary persona field on `Scenario` (replaces `user_attributes`)
- Simulator reads `user_profile` directly instead of generating profiles via LLM
- Jinja template variable renamed from `simulation.profile` to `scenario.user_profile`
- HTML report shows user profile in Scenario section and prompt template in Simulation Prompt section

### Removed

- LLM-based profile generation (`profile.py`, `prompts.py`, `schema.py` under `arksim/scenario/`)
- `user_attributes` removed from `Scenario` model root (moved into `origin` as metadata)
- `generate_profiles` method from simulator

## [0.0.3] - 2026-03-03

### Removed

- Azure references from documentation

## [0.0.2] - 2026-03-03

### Added

- GitHub Actions workflow to build, release, and publish `arksim` to PyPI on `v*` tag pushes, with dynamic VCS-based versioning via `hatch-vcs`
- Dependabot for automated pip and GitHub Actions dependency updates
- Pre-commit hooks with ruff, trailing whitespace, end-of-file, YAML, and changelog checks
- `py.typed` marker for PEP 561 type checking support
- Top-level lazy exports (`from arksim import run_simulation, Evaluator, ...`)
- `.editorconfig` for consistent editor settings
- `Makefile` with common dev commands
- Issue template chooser linking to docs and discussions
- Documentation and Changelog checklist in PR template

### Changed

- Refactored `bank-insurance` and `e-commerce` example agents to use the OpenAI Agents SDK and corrected the example server port from `8080` to `8888`
- Agent configuration is now defined inline in `config.yaml` under the `agent_config` key instead of in separate `agent_config.json` files
- Updated all documentation and examples to use inline agent configuration
- UI now passes `agent_config`, `custom_metrics_file_paths`, and `metrics_to_run` from loaded config YAML
- CI restructured: separate Lint job, test matrix on Python 3.10 through 3.13, PR checks (title, description, changelog) in one job
- Dependabot PRs now auto-labeled `skip-changelog` to bypass changelog CI check

### Removed

- Separate `agent_config.json` files from all examples (replaced by inline YAML config)
- Knowledge Configuration and Data folder sections from agent configuration docs

### Fixed

- Example servers: added logging before HTTP error responses, filtered system messages from agent chat history, fixed wrong config filename in e-commerce README
- Documentation: end-of-file newlines in `docs/insurance-customer-service-agent-evaluation.mdx` and `docs/e-commerce-customer-service-agent-evaluation.mdx`
- Documentation: trailing whitespace and end-of-file in `docs/simulate-conversation.mdx` and `docs/evaluate-conversation.mdx`
- Trailing whitespace and missing EOF newlines across 81 files (docs, examples, tests, source)
- UI file browser: `PROJECT_ROOT` now uses `cwd` for PyPI installs, parent navigation works correctly, YAML files shown in browser
- Lint: replaced `try/except/pass` with `contextlib.suppress` in example loader

### Security

- Removed third-party client data from test fixtures and evaluator prompt examples
- Replaced with generic insurance-domain examples consistent with the public bank-insurance example

## [0.0.1] - 2026-03-02

### Added

- Multi-turn conversation simulation with configurable scenarios
- Quantitative and qualitative evaluation metrics
- Chat Completions and A2A agent protocol support
- CLI (`arksim simulate`, `arksim evaluate`, `arksim simulate-evaluate`)
- Python SDK with `run_simulation()` and `run_evaluation()`
- Parallel execution with configurable worker count
- Example setups for e-commerce and bank insurance use cases
- OpenAI, Anthropic, and Google Gemini provider support
- HTML report generation for evaluation results

[Unreleased]: https://github.com/arklexai/arksim/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/arklexai/arksim/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/arklexai/arksim/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/arklexai/arksim/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/arklexai/arksim/releases/tag/v0.0.1
