---
paths:
  - "**/*.py"
---

# Python style rules

- Ruff format + ruff check enforced by hooks (auto-runs on every edit)
- Line length: 88 characters (ruff format default). E501 disabled.
- Ruff rules: E, F, FA, I, UP, ANN, B, SIM, C4, TID. Ignored: E501, UP045.
- Every .py file starts with `# SPDX-License-Identifier: Apache-2.0`
- Every module has `from __future__ import annotations` (after the license header)
- Absolute imports preferred. Single-level relative imports (`from .module import ...`) are allowed. Multi-level relative imports (`from ..module import ...`) are not.
- Type annotations on all function signatures
- Both `str | None` and `Optional[str]` accepted (UP045 ignored); prefer `str | None` in new code
