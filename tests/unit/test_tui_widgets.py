"""Tests for TUI widgets."""

from __future__ import annotations

from collections import deque

from rich.console import Console

from vox_terminal.tui.state import DisplayState
from vox_terminal.tui.widgets import ResponsePanel, SoundBar, StateSpinner, StatusBar


def _render(widget: object) -> str:
    """Render a Rich renderable to a plain string."""
    console = Console(file=None, force_terminal=True, width=80)
    with console.capture() as capture:
        console.print(widget, end="")
    return capture.get()


class TestSoundBar:
    def test_empty_levels(self) -> None:
        levels: deque[float] = deque(maxlen=10)
        bar = SoundBar(levels, width=10)
        text = _render(bar)
        # Should render without crashing; only contains spaces and ANSI codes
        assert "▇" not in text

    def test_full_levels(self) -> None:
        levels: deque[float] = deque([1.0] * 5, maxlen=10)
        bar = SoundBar(levels, width=5)
        text = _render(bar)
        assert "▇" in text

    def test_mixed_levels(self) -> None:
        levels: deque[float] = deque([0.0, 0.5, 1.0], maxlen=10)
        bar = SoundBar(levels, width=5)
        text = _render(bar)
        assert "▇" in text  # max level present

    def test_clamps_values(self) -> None:
        levels: deque[float] = deque([2.0, -1.0], maxlen=10)
        bar = SoundBar(levels, width=5)
        text = _render(bar)
        # Should not crash, values clamped
        assert isinstance(text, str)


class TestStateSpinner:
    def test_listening_renders(self) -> None:
        spinner = StateSpinner("listening")
        text = _render(spinner)
        assert "Listening" in text

    def test_transcribing_renders(self) -> None:
        spinner = StateSpinner("transcribing")
        text = _render(spinner)
        assert "Transcribing" in text

    def test_thinking_renders(self) -> None:
        spinner = StateSpinner("thinking")
        text = _render(spinner)
        assert "Thinking" in text

    def test_speaking_renders(self) -> None:
        spinner = StateSpinner("speaking")
        text = _render(spinner)
        assert "Speaking" in text


class TestResponsePanel:
    def test_renders_question_and_response(self) -> None:
        panel = ResponsePanel(["Hello ", "world"], question="What is this?")
        text = _render(panel)
        assert "What is this?" in text
        assert "Hello world" in text
        assert "Vox-Terminal" in text  # panel title

    def test_empty_response_shows_ellipsis(self) -> None:
        panel = ResponsePanel([], question="test")
        text = _render(panel)
        assert "…" in text

    def test_no_question(self) -> None:
        panel = ResponsePanel(["answer"])
        text = _render(panel)
        assert "answer" in text


class TestStatusBar:
    def test_renders_phase_and_model(self) -> None:
        state = DisplayState(
            phase="listening",
            model_name="claude-sonnet",
            history_count=3,
        )
        bar = StatusBar(state)
        text = _render(bar)
        assert "Listening" in text
        assert "claude-sonnet" in text
        assert "3" in text

    def test_renders_elapsed_time(self) -> None:
        state = DisplayState(phase="idle")
        bar = StatusBar(state)
        text = _render(bar)
        # Should contain a time like "0:00"
        assert "0:" in text
