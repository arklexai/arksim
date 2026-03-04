# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


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
    """Resolve num_workers, converting 'auto' to a concrete value."""
    if num_workers == "auto":
        return auto_value
    return num_workers
