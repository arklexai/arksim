# SPDX-License-Identifier: Apache-2.0
"""Per-SDK tracing adapters for arksim's Python connector.

Each adapter ships as an optional install (``pip install 'arksim[<sdk>]'``)
and is imported explicitly by the user; this package does NOT eagerly
import any SDK at package import time.
"""

from __future__ import annotations
