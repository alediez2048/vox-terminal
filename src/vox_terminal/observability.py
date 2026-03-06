"""Observability utilities for per-turn tracing."""

from __future__ import annotations

import time as _time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

_TURN_ID: ContextVar[str | None] = ContextVar("vox_terminal_turn_id", default=None)


def generate_turn_id() -> str:
    """Generate a unique trace identifier for a conversation turn."""
    return uuid.uuid4().hex


def get_current_turn_id() -> str | None:
    """Return the current turn ID from context-local storage."""
    return _TURN_ID.get()


def set_turn_id(turn_id: str) -> Token[str | None]:
    """Set turn ID in context-local storage and return reset token."""
    return _TURN_ID.set(turn_id)


def reset_turn_id(token: Token[str | None]) -> None:
    """Reset turn ID to the previous context-local value."""
    _TURN_ID.reset(token)


@dataclass(slots=True)
class TurnContext:
    """Context manager that sets and restores the active turn ID."""

    turn_id: str
    _token: Token[str | None] | None = None

    def __enter__(self) -> str:
        self._token = _TURN_ID.set(self.turn_id)
        return self.turn_id

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._token is not None:
            _TURN_ID.reset(self._token)


@dataclass(slots=True)
class TurnWaterfall:
    """Collect per-turn stage timestamps and expose elapsed milliseconds."""

    start_monotonic: float = field(default_factory=_time.monotonic)
    _marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, *, at: float | None = None, overwrite: bool = False) -> None:
        """Record a stage timestamp.

        By default, the first value wins for a mark name unless ``overwrite`` is
        explicitly set.
        """
        if not overwrite and name in self._marks:
            return
        self._marks[name] = at if at is not None else _time.monotonic()

    def elapsed_ms(self, name: str) -> float | None:
        """Return elapsed milliseconds from turn start for ``name``."""
        mark = self._marks.get(name)
        if mark is None:
            return None
        return (mark - self.start_monotonic) * 1000

    def snapshot_ms(self) -> dict[str, float]:
        """Return all recorded marks as elapsed milliseconds."""
        return {
            name: round((mark - self.start_monotonic) * 1000, 1)
            for name, mark in self._marks.items()
        }
