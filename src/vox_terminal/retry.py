"""Simple async retry utility with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default set of exceptions considered transient / retryable.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    TimeoutError,
    ConnectionError,
)


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: object,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: tuple[type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
    **kwargs: object,
) -> T:
    """Call *fn* with retry and exponential backoff.

    Parameters
    ----------
    fn:
        The async callable to invoke.
    max_attempts:
        Total number of attempts (including the first call).
    base_delay:
        Initial delay in seconds; doubles after each retry.
    retryable:
        Tuple of exception types that trigger a retry.  Any other
        exception propagates immediately.
    """
    last_exc: BaseException | None = None
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except retryable as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            logger.warning(
                "Attempt %d/%d failed (%s), retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2
    raise last_exc  # type: ignore[misc]
