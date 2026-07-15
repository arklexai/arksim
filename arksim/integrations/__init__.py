# SPDX-License-Identifier: Apache-2.0
"""Optional, third-party integrations for arksim.

Each subpackage here depends on an optional dependency and is import-safe:
importing ``arksim.integrations`` never pulls in a heavy optional package.
Install extras to enable one, e.g. ``pip install arksim[langfuse]``.
"""

from __future__ import annotations
