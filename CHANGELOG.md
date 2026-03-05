# Changelog

All notable changes to ArkSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0](https://github.com/arklexai/arksim/compare/v0.0.6...v0.1.0) (2026-03-05)


### ⚠ BREAKING CHANGES

* prepare for 0.1.0 release ([#66](https://github.com/arklexai/arksim/issues/66))
* update llm provider name ([#67](https://github.com/arklexai/arksim/issues/67))

### Added

* add version display to CLI and web UI ([#50](https://github.com/arklexai/arksim/issues/50)) ([7e0ec82](https://github.com/arklexai/arksim/commit/7e0ec82660ffa266b1caba1a663674b3f6b9899a))
* prepare for 0.1.0 release ([#66](https://github.com/arklexai/arksim/issues/66)) ([2e919ce](https://github.com/arklexai/arksim/commit/2e919ce44f23029c09d4bf969d0cd5c65fcca31d))
* support in-memory evaluation flow and refresh docs ([#63](https://github.com/arklexai/arksim/issues/63)) ([547546f](https://github.com/arklexai/arksim/commit/547546f1f587c3d1ebb0fd3c2143bd32c009e9da))
* update llm provider name ([#67](https://github.com/arklexai/arksim/issues/67)) ([e949e58](https://github.com/arklexai/arksim/commit/e949e58bdbe449dbbe17b30b63d287d74f98a3c2))


### Fixed

* **examples:** improve sample scenario files ([#53](https://github.com/arklexai/arksim/issues/53)) ([c0448a3](https://github.com/arklexai/arksim/commit/c0448a36cfdcd37b4f78334407116e6746661d0a))
* update installation and quickstart guides ([#65](https://github.com/arklexai/arksim/issues/65)) ([589ada7](https://github.com/arklexai/arksim/commit/589ada70ffcf31bb5d40571600139f49f997ca84))
* update logic for resolving file paths ([#54](https://github.com/arklexai/arksim/issues/54)) ([029eda8](https://github.com/arklexai/arksim/commit/029eda8f347c45d3bab8a8f6021b65e367d1d044))
* use absolute URL for flow diagram in README ([#69](https://github.com/arklexai/arksim/issues/69)) ([f09056f](https://github.com/arklexai/arksim/commit/f09056f4c4ded3395668983aeb982769129ca0cc))


### Changed

* add commit type reference to PR template ([#68](https://github.com/arklexai/arksim/issues/68)) ([f5974e7](https://github.com/arklexai/arksim/commit/f5974e701f1dd61b30e1996fd49c61bcf57e389f))
* enforce SPDX headers and future annotations ([#56](https://github.com/arklexai/arksim/issues/56)) ([edb0310](https://github.com/arklexai/arksim/commit/edb03101677ff6b6592cf44679fe47485357afbe))
* rebrand to ⛵️ ArkSim with slogan ([#57](https://github.com/arklexai/arksim/issues/57)) ([71c01ed](https://github.com/arklexai/arksim/commit/71c01ed832dd7f20acd5a9078e391eac744a1a49))

## [0.0.6](https://github.com/arklexai/arksim/compare/v0.0.5...v0.0.6) (2026-03-04)


### Fixed

* multi-provider chat completions and config-driven LLM ([#60](https://github.com/arklexai/arksim/issues/60)) ([4b8ce23](https://github.com/arklexai/arksim/commit/4b8ce23f6aaefaea6b8f6c87735f408f35ecd01b))
* respect num_conversations_per_scenario in simulator ([#55](https://github.com/arklexai/arksim/issues/55)) ([28b65b3](https://github.com/arklexai/arksim/commit/28b65b35ffc554ceb22b3ee4db2f9726a80039db))


### Documentation

* update quickstart to use examples command, fix examples ([#44](https://github.com/arklexai/arksim/issues/44)) ([97a27db](https://github.com/arklexai/arksim/commit/97a27db762c57d8eb7152de66b856d9dd0802efe))


### Changed

* trigger PyPI publish on release instead of tag push ([#62](https://github.com/arklexai/arksim/issues/62)) ([c086aea](https://github.com/arklexai/arksim/commit/c086aea0ac8a66d11c7adfb82cd79183f2e7d747))

## [0.0.5](https://github.com/arklexai/arksim/compare/v0.0.4...v0.0.5) (2026-03-04)


### Fixed

* remove stale token from Codecov coverage badge ([#49](https://github.com/arklexai/arksim/issues/49)) ([b2ac7fd](https://github.com/arklexai/arksim/commit/b2ac7fddf9dfac77e2bd4b4466e654cdf29e9bde))


### Changed

* add Codecov token to coverage upload ([#45](https://github.com/arklexai/arksim/issues/45)) ([0d25132](https://github.com/arklexai/arksim/commit/0d251324b2b498bfba0d853eaa75a70bf0bf2859))
* add OSS hardening for security, coverage, and annotations ([#46](https://github.com/arklexai/arksim/issues/46)) ([aa7fa77](https://github.com/arklexai/arksim/commit/aa7fa77b4c0db302cf8bd0104dcd7b076d134d0b))
* add release-please for automated releases and changelog ([#47](https://github.com/arklexai/arksim/issues/47)) ([92f48cc](https://github.com/arklexai/arksim/commit/92f48cc05241f7c741d97c3a7fa48eb7da6996a4))
* code quality audit fixes ([#52](https://github.com/arklexai/arksim/issues/52)) ([b7295c2](https://github.com/arklexai/arksim/commit/b7295c218637a4579d8a1f190b4460039259b86c))
* improve unit test coverage from 44% to 62% ([#51](https://github.com/arklexai/arksim/issues/51)) ([a3f8f42](https://github.com/arklexai/arksim/commit/a3f8f4213637939f0377b963fba632a2cc024f04))
* trigger CI on release-please branches ([#59](https://github.com/arklexai/arksim/issues/59)) ([71ba01b](https://github.com/arklexai/arksim/commit/71ba01b1c6611dc5763093ff69c2c958b385e5a8))

## [Unreleased]

### Fixed

- Simulator now respects `num_conversations_per_scenario`, previously only 1 conversation was generated per scenario

### Added

- Support for agents that use tool/function calling in simulations
- `-v` / `--version` CLI flag to show arksim version and exit
- Version display in web UI sidebar (next to "Arksim" title)
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

- Project display name from "Arksim" to "ArkSim" with ⛵️ emoji and slogan across all docs, CLI, and metadata
- `SCORE_NOT_COMPUTED` display label changed from "N/A (Evaluation Failed)" to "N/A (Not computed)"
- Removed stale `{chat_id}` placeholder reference from `ChatCompletionsConfig.body` field description

- UI auto-loads scenarios from config on startup instead of always showing the demo
- File browser shows a hint that browsing is scoped to the launch directory
- Coverage badge switched from Codecov-hosted to shields.io for reliability
- Expanded SECURITY.md with response process, scope, and disclosure guidelines
- Azure OpenAI provider now raises `ValueError` instead of silently returning a raw string when structured output parsing fails
- `validate_num_workers` rejects zero and negative values
- `run_evaluation` now accepts optional in-memory `simulation` and `scenarios` inputs, while still supporting file-based loading
- Evaluation docs and quickstart examples now show direct Python usage of `run_simulation` and `run_evaluation` with in-memory handoff between steps
- Resolve input file paths to config-relative if read from config.yaml and resolve as cwd-relative if cli overriden.

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
