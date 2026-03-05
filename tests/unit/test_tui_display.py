"""Tests for SessionDisplay lifecycle and state transitions."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from vox_terminal.tui.display import SessionDisplay
from vox_terminal.tui.state import DisplayState


@pytest.fixture
def display_state() -> DisplayState:
    return DisplayState(model_name="test-model")


@pytest.fixture
def display(display_state: DisplayState) -> SessionDisplay:
    console = Console(file=None, force_terminal=True, width=80)
    return SessionDisplay(display_state, console=console)


class TestSessionDisplayLifecycle:
    async def test_start_and_stop(self, display: SessionDisplay) -> None:
        await display.start()
        assert display._live is not None
        assert display._running is True
        assert display._refresh_task is not None

        await display.stop()
        assert display._live is None
        assert display._running is False

    async def test_stop_without_start(self, display: SessionDisplay) -> None:
        # Should not raise
        await display.stop()

    async def test_pause_resume(self, display: SessionDisplay) -> None:
        await display.start()

        display.pause()
        # Live should be stopped but not None
        assert display._live is not None

        display.resume()
        assert display._live is not None

        await display.stop()

    async def test_print_static(self, display: SessionDisplay) -> None:
        # Should not raise whether live is active or not
        display.print_static("test message")

        await display.start()
        display.print_static("test during live")
        await display.stop()


class TestStateTransitions:
    async def test_phase_changes_reflected(
        self, display: SessionDisplay, display_state: DisplayState
    ) -> None:
        await display.start()

        display_state.phase = "listening"
        # Let refresh loop run once
        await asyncio.sleep(0.1)

        display_state.phase = "thinking"
        display_state.question = "Hello?"
        await asyncio.sleep(0.1)

        display_state.phase = "speaking"
        display_state.response_chunks.append("Hi there!")
        await asyncio.sleep(0.1)

        display_state.phase = "idle"
        await asyncio.sleep(0.1)

        await display.stop()

    async def test_audio_level_from_capture(
        self, display: SessionDisplay, display_state: DisplayState
    ) -> None:
        mock_capture = MagicMock()
        mock_capture.audio_level = 0.75

        await display.start(capture=mock_capture)
        await asyncio.sleep(0.1)  # let refresh loop read level

        assert display_state.audio_level == pytest.approx(0.75, abs=0.01)
        assert len(display._levels) > 0

        await display.stop()

    async def test_no_capture_defaults_zero(
        self, display: SessionDisplay, display_state: DisplayState
    ) -> None:
        await display.start(capture=None)
        await asyncio.sleep(0.1)

        assert display_state.audio_level == 0.0

        await display.stop()
