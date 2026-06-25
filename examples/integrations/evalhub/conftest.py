# SPDX-License-Identifier: Apache-2.0
"""Put the example directory on sys.path so tests can import ``arksim_evalhub``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
