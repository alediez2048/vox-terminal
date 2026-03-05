"""Shared display state for the TUI."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["idle", "listening", "transcribing", "thinking", "speaking"]


@dataclass
class DisplayState:
    """Mutable state read by the display refresh loop."""

    phase: Phase = "idle"
    audio_level: float = 0.0
    response_chunks: list[str] = field(default_factory=list)
    question: str = ""
    model_name: str = ""
    history_count: int = 0
    session_start: float = field(default_factory=_time.monotonic)
    turn_start: float = 0.0
    error_message: str = ""
