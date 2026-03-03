# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps

logger = logging.getLogger(__name__)

MAX_RETRIES = 5


def _parse_retry_after(raw: str, default: int = 5) -> int:
    """Parse a Retry-After header value, falling back to default on HTTP-date strings."""
    try:
        return int(raw)
    except ValueError:
        return default


def rate_limit_handler(func: Callable) -> Callable:
    """
    A decorator to handle 429 Too Many Requests errors with
    Retry-After header support. Works with both sync and async
    functions that return an httpx/requests Response object.
    Retries up to MAX_RETRIES times before propagating the error.
    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            for attempt in range(MAX_RETRIES):
                response = await func(*args, **kwargs)
                if response.status_code == 429:
                    raw = response.headers.get("Retry-After", "5")
                    retry_after = _parse_retry_after(raw)
                    logger.warning(
                        f"Rate limited (429). Waiting for {retry_after} seconds... "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(retry_after)
                    continue
                else:
                    response.raise_for_status()
                    return response
            logger.error(f"Rate limit retries exhausted after {MAX_RETRIES} attempts.")
            response.raise_for_status()

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args: object, **kwargs: object) -> object:
            for attempt in range(MAX_RETRIES):
                response = func(*args, **kwargs)
                if response.status_code == 429:
                    raw = response.headers.get("Retry-After", "5")
                    retry_after = _parse_retry_after(raw)
                    logger.warning(
                        f"Rate limited (429). Waiting for {retry_after} seconds... "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(retry_after)
                    continue
                else:
                    response.raise_for_status()
                    return response
            logger.error(f"Rate limit retries exhausted after {MAX_RETRIES} attempts.")
            response.raise_for_status()

        return sync_wrapper
