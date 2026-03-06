"""Observability utilities for per-turn tracing."""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass

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
