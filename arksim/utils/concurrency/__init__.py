# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from .workers import resolve_num_workers, validate_num_workers

__all__ = [
    "validate_num_workers",
    "resolve_num_workers",
]
