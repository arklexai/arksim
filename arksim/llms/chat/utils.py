# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])

_BASE_DELAY = 1.0
_MAX_DELAY = 30.0


def _is_non_retryable(e: Exception) -> bool:
    """Return True for 4xx errors (except 429) that will never succeed on retry.

    Checks both ``.status_code`` (OpenAI, Anthropic) and ``.code`` (Google GenAI)
    so that non-retryable errors are detected regardless of the underlying library.
    """
    status_code = getattr(e, "status_code", None)
    if status_code is None:
        status_code = getattr(e, "code", None)
    return (
        isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429
    )


def retry(max_retries: int = 5) -> Callable[[F], F]:
    """Retry decorator for LLM calls with exponential backoff."""

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: object, **kwargs: object) -> object:
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if _is_non_retryable(e) or attempt >= max_retries - 1:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded. Error: {e}",
                            )
                            raise
                        delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: object, **kwargs: object) -> object:
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if _is_non_retryable(e) or attempt >= max_retries - 1:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded. Error: {e}",
                            )
                            raise
                        delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            return sync_wrapper

    return decorator
