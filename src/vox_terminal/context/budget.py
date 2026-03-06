"""Context budget helpers for assembling prompt context."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ContextFragment:
    """A context fragment with priority metadata."""

    name: str
    content: str
    priority: int
    token_estimate: int = 0
    requires_network: bool = False


class ContextBudget:
    """Character budget manager for context assembly."""

    def __init__(self, total_chars: int) -> None:
        self._total_chars = max(0, total_chars)
        self._remaining_chars = self._total_chars

    @property
    def total_chars(self) -> int:
        return self._total_chars

    @property
    def remaining_chars(self) -> int:
        return self._remaining_chars

    def allocate(self, fragment: ContextFragment) -> str:
        """Allocate budget to *fragment* and return included content."""
        if self._remaining_chars <= 0:
            return ""
        if not fragment.content:
            return ""
        if len(fragment.content) <= self._remaining_chars:
            self._remaining_chars -= len(fragment.content)
            return fragment.content

        allocated = fragment.content[: self._remaining_chars]
        self._remaining_chars = 0
        return allocated
