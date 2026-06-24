# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioBuffer:
    """PCM audio interchange type for the voice loop.

    ``samples`` is mono float32 in [-1, 1] unless ``num_channels`` > 1.
    """

    samples: np.ndarray
    sample_rate: int
    num_channels: int = 1
