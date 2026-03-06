"""Tests for observability helpers."""

from __future__ import annotations

import asyncio

import pytest

from vox_terminal.observability import (
    TurnContext,
    TurnWaterfall,
    generate_turn_id,
    get_current_turn_id,
)


class TestGenerateTurnID:
    def test_generate_turn_id_returns_hex_uuid(self) -> None:
        turn_id = generate_turn_id()
        assert isinstance(turn_id, str)
        assert len(turn_id) == 32
        int(turn_id, 16)  # validate hex

    def test_generate_turn_id_is_unique(self) -> None:
        first = generate_turn_id()
        second = generate_turn_id()
        assert first != second


class TestTurnContext:
    def test_turn_context_sets_and_resets(self) -> None:
        assert get_current_turn_id() is None
        with TurnContext("abc123"):
            assert get_current_turn_id() == "abc123"
        assert get_current_turn_id() is None

    async def test_turn_context_propagates_to_async_tasks(self) -> None:
        async def _read_turn_id() -> str | None:
            await asyncio.sleep(0)
            return get_current_turn_id()

        with TurnContext("task-turn-id"):
            task = asyncio.create_task(_read_turn_id())
            assert await task == "task-turn-id"


class TestTurnWaterfall:
    def test_mark_and_snapshot(self) -> None:
        waterfall = TurnWaterfall(start_monotonic=100.0)
        waterfall.mark("stt_start_ms", at=100.05)
        waterfall.mark("stt_end_ms", at=100.2)
        snapshot = waterfall.snapshot_ms()
        assert snapshot["stt_start_ms"] == 50.0
        assert snapshot["stt_end_ms"] == 200.0

    def test_first_value_wins_without_overwrite(self) -> None:
        waterfall = TurnWaterfall(start_monotonic=1.0)
        waterfall.mark("llm_first_token_ms", at=1.2)
        waterfall.mark("llm_first_token_ms", at=1.4)
        assert waterfall.elapsed_ms("llm_first_token_ms") == pytest.approx(200.0)
