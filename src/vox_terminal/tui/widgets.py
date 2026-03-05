"""Rich renderables for the TUI."""

from __future__ import annotations

import time as _time
from collections import deque

from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from vox_terminal.tui.state import DisplayState, Phase

# Unicode block characters for the soundbar (8 levels)
_BLOCKS = " ▁▂▃▄▅▆▇"


class SoundBar:
    """Renders a horizontal audio-level bar from a deque of recent levels."""

    def __init__(self, levels: deque[float], width: int = 40) -> None:
        self._levels = levels
        self._width = width

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        width = min(self._width, options.max_width)
        # Take the most recent `width` samples, pad with 0 if fewer
        samples = list(self._levels)[-width:]
        if len(samples) < width:
            pad = width - len(samples)
            samples = [0.0] * pad + samples

        chars: list[str] = []
        for level in samples:
            clamped = max(0.0, min(1.0, level))
            idx = int(clamped * (len(_BLOCKS) - 1))
            chars.append(_BLOCKS[idx])

        yield Text("".join(chars), style="green")


# Spinner configs per phase
_SPINNER_MAP: dict[Phase, str] = {
    "listening": "point",
    "transcribing": "dots",
    "thinking": "moon",
    "speaking": "speaker",  # custom frames below
}

_SPEAKER_FRAMES = ["🔈", "🔉", "🔊", "🔉"]

_PHASE_LABELS: dict[Phase, str] = {
    "listening": "Listening…",
    "transcribing": "Transcribing…",
    "thinking": "Thinking…",
    "speaking": "Speaking…",
    "idle": "",
}

_PHASE_STYLES: dict[Phase, str] = {
    "listening": "green",
    "transcribing": "dim",
    "thinking": "bold green",
    "speaking": "bold green",
    "idle": "dim",
}


class StateSpinner:
    """Phase-aware animated spinner with label."""

    def __init__(self, phase: Phase) -> None:
        self._phase = phase
        if phase == "speaking":
            self._spinner = Spinner("dots", text=_PHASE_LABELS[phase], style=_PHASE_STYLES[phase])
            self._speaker_start = _time.monotonic()
        else:
            name = _SPINNER_MAP.get(phase, "dots")
            self._spinner = Spinner(name, text=_PHASE_LABELS[phase], style=_PHASE_STYLES[phase])
            self._speaker_start = 0.0

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self._phase == "speaking":
            idx = int((_time.monotonic() - self._speaker_start) * 3) % len(_SPEAKER_FRAMES)
            frame = _SPEAKER_FRAMES[idx]
            yield Text(f"{frame} {_PHASE_LABELS[self._phase]}", style=_PHASE_STYLES[self._phase])
        else:
            yield self._spinner


class ResponsePanel:
    """Rich Panel wrapping the streamed LLM response."""

    def __init__(self, chunks: list[str], question: str = "") -> None:
        self._chunks = chunks
        self._question = question

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        body_parts: list[Text] = []
        if self._question:
            body_parts.append(Text(f"You: {self._question}", style="bold cyan"))
            body_parts.append(Text())

        response_text = "".join(self._chunks)
        if response_text:
            body_parts.append(Text(response_text, style="white"))
        else:
            body_parts.append(Text("…", style="dim"))

        body = Text("\n").join(body_parts)
        yield Panel(
            body,
            title="Vox-Terminal",
            border_style="green",
            expand=True,
        )


class StatusBar:
    """Persistent footer with session info."""

    def __init__(self, state: DisplayState) -> None:
        self._state = state

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        s = self._state

        # Phase icon
        phase_icons: dict[Phase, str] = {
            "idle": "⏸",
            "listening": "🎙",
            "transcribing": "📝",
            "thinking": "🧠",
            "speaking": "🔊",
        }
        icon = phase_icons.get(s.phase, "?")

        # Elapsed time
        elapsed = _time.monotonic() - s.session_start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}:{seconds:02d}"

        # Model name (truncate if long)
        model = s.model_name or "—"
        if len(model) > 25:
            model = model[:22] + "…"

        # Build grid row
        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(justify="left", ratio=1)
        table.add_column(justify="center", ratio=1)
        table.add_column(justify="center", ratio=1)
        table.add_column(justify="right", ratio=1)
        table.add_row(
            Text(f"{icon} {s.phase.capitalize()}", style=_PHASE_STYLES.get(s.phase, "dim")),
            Text(f"⏱ {time_str}", style="dim"),
            Text(f"🤖 {model}", style="dim"),
            Text(f"💬 {s.history_count}", style="dim"),
        )
        yield table
