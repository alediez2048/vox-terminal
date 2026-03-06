"""Abstract base classes for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """A single conversation message."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] | None = field(default=None)


class LLMClient(ABC):
    """Abstract base for all LLM provider clients."""

    @abstractmethod
    async def stream(self, prompt: str, history: list[Message] | None = None) -> AsyncIterator[str]:
        """Stream response tokens for *prompt*, yielding text deltas."""
        ...  # pragma: no cover
        # https://github.com/python/mypy/issues/5070 — yield needed for type inference
        if False:
            yield ""  # pragma: no cover

    @abstractmethod
    async def ask(self, prompt: str, history: list[Message] | None = None) -> LLMResponse:
        """Send *prompt* and return the complete response."""
        ...  # pragma: no cover
