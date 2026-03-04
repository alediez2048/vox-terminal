"""Tests for the retry utility."""

from __future__ import annotations

import asyncio

import pytest

from vox_terminal.retry import retry_async


class TestRetryAsync:
    async def test_success_on_first_attempt(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_async(fn, max_attempts=3, base_delay=0.01)
        assert result == "ok"
        assert calls == 1

    async def test_retries_on_transient_error(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise OSError("transient")
            return "recovered"

        result = await retry_async(fn, max_attempts=3, base_delay=0.01)
        assert result == "recovered"
        assert calls == 3

    async def test_exhaustion_raises_last_error(self) -> None:
        async def fn() -> str:
            raise TimeoutError("always fails")

        with pytest.raises(TimeoutError, match="always fails"):
            await retry_async(fn, max_attempts=2, base_delay=0.01)

    async def test_non_retryable_propagates_immediately(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await retry_async(fn, max_attempts=3, base_delay=0.01)
        assert calls == 1

    async def test_custom_retryable_exceptions(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise ValueError("custom retryable")
            return "ok"

        result = await retry_async(
            fn, max_attempts=3, base_delay=0.01, retryable=(ValueError,)
        )
        assert result == "ok"
        assert calls == 2

    async def test_passes_args_and_kwargs(self) -> None:
        async def fn(a: int, b: int, mul: int = 1) -> int:
            return (a + b) * mul

        result = await retry_async(fn, 2, 3, max_attempts=1, base_delay=0.01, mul=10)
        assert result == 50
