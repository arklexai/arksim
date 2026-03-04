# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Upper bound for 'auto' worker counts.  High enough to keep I/O-bound LLM
# calls well-utilised, low enough to avoid saturating API rate-limits or
# spawning hundreds of threads that stall the progress bar.
MAX_WORKERS: int = 8


def validate_num_workers(num_workers: int | str) -> None:
    """Validate that num_workers is an integer or the string 'auto'."""
    if isinstance(num_workers, str):
        if num_workers != "auto":
            raise ValueError(
                f"num_workers must be an integer or 'auto', got: {num_workers!r}"
            )
    elif not isinstance(num_workers, int):
        raise ValueError(
            f"num_workers must be an integer or 'auto', got: {type(num_workers).__name__}"
        )


def resolve_num_workers(num_workers: int | str, auto_value: int) -> int:
    """Resolve num_workers, converting 'auto' to a concrete value.

    When *num_workers* is ``"auto"`` the result is capped at
    :data:`MAX_WORKERS` to prevent excessive thread/task counts that can
    overwhelm API rate-limits or stall the event loop.
    """
    if num_workers == "auto":
        return min(auto_value, MAX_WORKERS)
    return num_workers
