# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum


class OutputDir(str, Enum):
    SCENARIO = "scenario"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    RESULTS = "results"
