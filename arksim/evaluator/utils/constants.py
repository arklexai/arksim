# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Quantitative Metric Threshold (1-5 scale)
METRIC_THRESHOLD = 3

# Turn Success Ratio Score Weight
TURN_SUCCESS_RATIO_SCORE_WEIGHT = 0.75
# Goal Completion Score Weight
GOAL_COMPLETION_SCORE_WEIGHT = 0.25

# Evaluation Partial Failure Threshold
EVALUATION_PARTIAL_FAILURE_THRESHOLD = 0.6


# Score interpretation labels (1-5 scale)
SCORE_LABELS = {
    (1.0, 2.0): "Poor",
    (2.0, 3.0): "Needs Improvement",
    (3.0, 4.0): "Good",
    (4.0, 5.0): "Excellent",
}


def score_label(score: float) -> str:
    """Map a 1-5 score to a human-readable label."""
    if score < 1.0:
        return "Poor"
    for (low, high), label in SCORE_LABELS.items():
        if low <= score < high:
            return label
    return "Excellent"  # score == 5.0
