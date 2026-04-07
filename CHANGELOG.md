# Changelog

All notable changes to ArkSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* **tracing:** add automatic tool call capture for agents that handle tools internally
* **tracing:** add `ArksimTracingProcessor` for OpenAI Agents SDK (register once, zero per-turn wrapping)
* **tracing:** add OTLP/HTTP trace receiver with protobuf and JSON support (`arksim[otel]`)
* **tracing:** add dual attribute convention support (OTel GenAI semconv and OpenInference)
* **examples:** add Dify chatbot integration example
* **evaluator:** focus file generation after evaluation for targeted reruns of failing scenarios
* **evaluator:** scenario IDs shown in CLI error output alongside focus file paths
* **report:** scenario IDs displayed in HTML report error cards
* **evaluator:** error-to-scenario mappings included in `evaluation.json` output

### Changed

* **simulation:** bump output schema version to v1.1 (additive `tool_calls` field on Message)
* **evaluator:** bump evaluation output schema version to v1.1 (additive `error_scenario_mappings` field)

## [0.3.3](https://github.com/arklexai/arksim/compare/v0.3.2...v0.3.3) (2026-03-27)


### Added

* **eval:** add deterministic trajectory matching for tool calls ([#111](https://github.com/arklexai/arksim/issues/111)) ([1d0e81a](https://github.com/arklexai/arksim/commit/1d0e81a0a3632ecd6f31d4dca0856df6d2c7ff0e))
* **examples:** upgrade Rasa integration to Rasa Pro with CALM ([#123](https://github.com/arklexai/arksim/issues/123)) ([0f5ccfa](https://github.com/arklexai/arksim/commit/0f5ccfa28207e35d014d99c276c6a74c0a755e5d))


### Fixed

* **evaluator:** tighten agent behavior failure prompts and add e2e test ([#125](https://github.com/arklexai/arksim/issues/125)) ([a77a413](https://github.com/arklexai/arksim/commit/a77a41334141459179a3eb313812a145f23ce27f))


### Documentation

* remove unrelated image in ci-integration page ([#120](https://github.com/arklexai/arksim/issues/120)) ([8bdf05b](https://github.com/arklexai/arksim/commit/8bdf05b010808cfe0d7408bc7108d4705802a1a1))
* snapshot v0.3.2 from main ([#116](https://github.com/arklexai/arksim/issues/116)) ([a04bcef](https://github.com/arklexai/arksim/commit/a04bcef290f21a545651501ac33de66ce0565101))
* update docs links to use main branch docs ([#117](https://github.com/arklexai/arksim/issues/117)) ([bd133b3](https://github.com/arklexai/arksim/commit/bd133b33e4287f5d33ee53054adc0c73aca86c48))


### Changed

* add Claude Code standards, skills, and hooks ([#122](https://github.com/arklexai/arksim/issues/122)) ([f52cbfc](https://github.com/arklexai/arksim/commit/f52cbfc3e156100d0fc15a2c5bd8a8030788e11b))
* improve evaluation metrics display in html report ([#119](https://github.com/arklexai/arksim/issues/119)) ([a05a964](https://github.com/arklexai/arksim/commit/a05a96484cab92d03f4e3d02cb6ab275a1bc7b9c))

## [0.3.2](https://github.com/arklexai/arksim/compare/v0.3.1...v0.3.2) (2026-03-20)


### Added

* **examples:** add Rasa integration example ([#99](https://github.com/arklexai/arksim/issues/99)) ([bfe200d](https://github.com/arklexai/arksim/commit/bfe200d261acaf3fa260d433da7c1101f6cb12c1))
* **examples:** update integration examples and add docs page ([#110](https://github.com/arklexai/arksim/issues/110)) ([0102ea2](https://github.com/arklexai/arksim/commit/0102ea28df229977f600dc5ce40394a4f2032114))


### Fixed

* **ui:** handle missing output path when no sidebar config selected ([#106](https://github.com/arklexai/arksim/issues/106)) ([9a84334](https://github.com/arklexai/arksim/commit/9a843346a981a14c639c659ebbb5f0f657547da3))


### Documentation

* add versioned documentation structure with automated snapshots ([#104](https://github.com/arklexai/arksim/issues/104)) ([17d6aed](https://github.com/arklexai/arksim/commit/17d6aede141544732a49db2de52ae3a77c620ce4))
* update readme quickstart section with documentation ([#112](https://github.com/arklexai/arksim/issues/112)) ([d0d63d9](https://github.com/arklexai/arksim/commit/d0d63d91eb6c37364bde5d482fccd416c95f5bc4))


### Changed

* add colors for qualitative metrics ([#109](https://github.com/arklexai/arksim/issues/109)) ([42abaf1](https://github.com/arklexai/arksim/commit/42abaf17099af3ad68818888f90c7359f86913fa))
* fix tests for simulation schema version update ([#113](https://github.com/arklexai/arksim/issues/113)) ([641fd5e](https://github.com/arklexai/arksim/commit/641fd5e42779de499c31dcafd62dcf7018df0951))

## [0.3.1](https://github.com/arklexai/arksim/compare/v0.3.0...v0.3.1) (2026-03-17)


### Added

* **evaluator:** add tool call evaluation with working example ([#95](https://github.com/arklexai/arksim/issues/95)) ([761a750](https://github.com/arklexai/arksim/commit/761a7500101eb324435168a15709f400468d32d0))


### Fixed

* **eval:** correctly provide conversation history to eval prompts ([#107](https://github.com/arklexai/arksim/issues/107)) ([2211112](https://github.com/arklexai/arksim/commit/22111124b0b011e628ba932309c26fdba1c3870f))
* **ui:** fix scenario save and add agent_context field ([#105](https://github.com/arklexai/arksim/issues/105)) ([77e0d83](https://github.com/arklexai/arksim/commit/77e0d8349181418894a5b49a2c3cc507e1fd9226))

## [0.3.0](https://github.com/arklexai/arksim/compare/v0.2.0...v0.3.0) (2026-03-17)


### ⚠ BREAKING CHANGES

* add CI integration templates, docs, and example updates ([#92](https://github.com/arklexai/arksim/issues/92))

### Added

* add CI integration templates, docs, and example updates ([#92](https://github.com/arklexai/arksim/issues/92)) ([09a3674](https://github.com/arklexai/arksim/commit/09a367402a849a4fb89845ab5e940e3756b56157))
* add per-metric thresholds and structured exit codes ([#89](https://github.com/arklexai/arksim/issues/89)) ([5f0f93a](https://github.com/arklexai/arksim/commit/5f0f93aebbd03fd8bad624b7b4f707a01ab5d756))
* **examples:** add 6 new SDK/framework integration examples ([#97](https://github.com/arklexai/arksim/issues/97)) ([883ff9a](https://github.com/arklexai/arksim/commit/883ff9a930167bb94440a5bf9c3eaae9cef9b511))


### Documentation

* update readme to include paper information ([#90](https://github.com/arklexai/arksim/issues/90)) ([8e18e5c](https://github.com/arklexai/arksim/commit/8e18e5c05c046fb71926450eaaf8517c6ecc0ea1))


### Changed

* add integration tests ([#91](https://github.com/arklexai/arksim/issues/91)) ([b047397](https://github.com/arklexai/arksim/commit/b047397bf0284bbf10925c74d380deca90be6efc))
* bump actions/create-github-app-token from 2 to 3 ([#98](https://github.com/arklexai/arksim/issues/98)) ([724a779](https://github.com/arklexai/arksim/commit/724a77994037ce84178c54841750c88617fb1972))
* update SECURITY.md supported versions ([#94](https://github.com/arklexai/arksim/issues/94)) ([2220870](https://github.com/arklexai/arksim/commit/22208706365d63e22d08743fa06d0e888a2b9d7e))
* update vercel ai sdk deps for security updates ([#100](https://github.com/arklexai/arksim/issues/100)) ([412ad4c](https://github.com/arklexai/arksim/commit/412ad4cc84db932c9866c901d4d07e7c6d3f29df))

## [0.2.0](https://github.com/arklexai/arksim/compare/v0.1.0...v0.2.0) (2026-03-10)


### ⚠ BREAKING CHANGES

* support custom Python agent connector ([#76](https://github.com/arklexai/arksim/issues/76))
* fail early if error loading custom metrics or file not found ([#78](https://github.com/arklexai/arksim/issues/78))

### Added

* **examples:** add SDK/framework integration examples ([#87](https://github.com/arklexai/arksim/issues/87)) ([a56ad00](https://github.com/arklexai/arksim/commit/a56ad00cf027b9d97bd269fb58107588d537ae51))
* support custom Python agent connector ([#76](https://github.com/arklexai/arksim/issues/76)) ([eeaf33f](https://github.com/arklexai/arksim/commit/eeaf33fdef0fbc45e5c8d89d18083446150507c0))


### Fixed

* fail early if error loading custom metrics or file not found ([#78](https://github.com/arklexai/arksim/issues/78)) ([b7a4bde](https://github.com/arklexai/arksim/commit/b7a4bdea8fbe024b3ef9c5651754f23ec6332df6))


### Documentation

* add demo video ([#77](https://github.com/arklexai/arksim/issues/77)) ([8a6d2e6](https://github.com/arklexai/arksim/commit/8a6d2e6e39a2ade7ea4c7b446c198698c02e80f2))
* update slogan and video ([#80](https://github.com/arklexai/arksim/issues/80)) ([0679f9c](https://github.com/arklexai/arksim/commit/0679f9cb3f598fa44e5dc9ffe6e6dafdfc66554b))


### Changed

* bump actions/checkout from 4 to 6 ([#81](https://github.com/arklexai/arksim/issues/81)) ([11cc630](https://github.com/arklexai/arksim/commit/11cc63018b56b6c2116c283337eca6cc17bb3c35))
* bump actions/create-github-app-token from 1 to 2 ([#84](https://github.com/arklexai/arksim/issues/84)) ([529c6a6](https://github.com/arklexai/arksim/commit/529c6a697ff7085887831c40792ab0071f9cc396))
* bump actions/download-artifact from 4 to 8 ([#82](https://github.com/arklexai/arksim/issues/82)) ([ff969bf](https://github.com/arklexai/arksim/commit/ff969bf28fff99f9e0fe9f5e5e19cc000ccc93ec))
* bump actions/upload-artifact from 4 to 7 ([#83](https://github.com/arklexai/arksim/issues/83)) ([b9fa245](https://github.com/arklexai/arksim/commit/b9fa24522299423b52abb3cff3346ccef3d1985c))
* bump github/codeql-action from 3 to 4 ([#85](https://github.com/arklexai/arksim/issues/85)) ([3189b9b](https://github.com/arklexai/arksim/commit/3189b9b8dcf9bbe1943532c71e96afdd8538df95))
* **ui:** remove demo scenario loading ([#71](https://github.com/arklexai/arksim/issues/71)) ([709e6ab](https://github.com/arklexai/arksim/commit/709e6ab953021a24f42b02aeb60f02447b782ca9))

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
