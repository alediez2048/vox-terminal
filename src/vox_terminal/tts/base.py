"""Abstract base class for TTS engines with sentence-buffered streaming."""

from __future__ import annotations

import logging
import re
import time as _time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# Patterns that sound bad when read aloud by TTS
_MARKDOWN_RE = re.compile(
    r"\*{1,3}"  # bold / italic markers
    r"|`{1,3}"  # inline code / code fences
    r"|^#{1,6}\s"  # heading markers
    r"|^[-*]\s"  # bullet list markers
    r"|^\d+\.\s"  # numbered list markers
    r"|^>\s"  # blockquote markers
    r"|\[([^\]]*)\]\([^)]*\)",  # markdown links → keep link text
    re.MULTILINE,
)


def _sanitize_for_speech(text: str) -> str:
    """Strip markdown and special characters that TTS engines read aloud."""
    # Replace markdown links with just the link text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove remaining markdown syntax
    text = _MARKDOWN_RE.sub("", text)
    # Strip conventional commit prefixes (feat:, fix:, chore:, etc.)
    text = re.sub(r"\b(feat|fix|chore|docs|style|refactor|perf|test|ci|build|revert):\s*", "", text)
    # Remove content inside parentheses that looks like technical tags (e.g. "(MVP-017)")
    text = re.sub(r"\([A-Z]+-\d+\)", "", text)
    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


class TTSEngine(ABC):
    """Abstract base for all text-to-speech engines."""

    _interrupted: bool = False

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak a complete text string."""
        ...

    def interrupt(self) -> None:
        """Stop speaking as soon as possible.

        Subclasses that spawn a playback process should override this to
        also terminate that process.
        """
        self._interrupted = True

    async def speak_streamed(self, chunks: AsyncIterator[str]) -> None:
        """Buffer LLM streaming chunks and speak at sentence boundaries.

        Accumulates text chunks and flushes to speak() when a sentence
        boundary (.!?\\n) is detected.  Text is sanitised to remove markdown
        and special characters before being sent to the TTS engine.

        Respects :meth:`interrupt` — if called, the current playback is
        killed and no further sentences are spoken.
        """
        self._interrupted = False
        buffer = ""
        spoken_sentences = 0
        spoken_chars = 0
        t0 = _time.monotonic()
        async for chunk in chunks:
            if self._interrupted:
                break
            buffer += chunk
            # Check for sentence boundaries
            while not self._interrupted:
                # Find the earliest sentence boundary
                idx = -1
                for sep in ".!?\n":
                    pos = buffer.find(sep)
                    if pos != -1 and (idx == -1 or pos < idx):
                        idx = pos
                if idx == -1:
                    break
                sentence = buffer[: idx + 1].strip()
                buffer = buffer[idx + 1 :]
                if sentence:
                    sentence = _sanitize_for_speech(sentence)
                    if sentence:
                        await self.speak(sentence)
                        spoken_sentences += 1
                        spoken_chars += len(sentence)
        if not self._interrupted:
            # Flush remaining
            remaining = buffer.strip()
            if remaining:
                remaining = _sanitize_for_speech(remaining)
                if remaining:
                    await self.speak(remaining)
                    spoken_sentences += 1
                    spoken_chars += len(remaining)
        logger.info(
            "TTS streamed output (sentences=%d, chars=%d, interrupted=%s, elapsed_ms=%.0f)",
            spoken_sentences,
            spoken_chars,
            self._interrupted,
            (_time.monotonic() - t0) * 1000,
        )


class FallbackTTSEngine(TTSEngine):
    """Wraps a primary TTS engine with an automatic fallback.

    If the primary engine's ``speak()`` raises, switches to the fallback
    engine for the rest of the session and logs a warning.
    """

    def __init__(self, primary: TTSEngine, fallback: TTSEngine) -> None:
        self._primary = primary
        self._fallback = fallback
        self._using_fallback = False

    @property
    def active_engine(self) -> TTSEngine:
        """The engine currently being used."""
        return self._fallback if self._using_fallback else self._primary

    async def speak(self, text: str) -> None:
        if self._using_fallback:
            await self._fallback.speak(text)
            return
        try:
            await self._primary.speak(text)
        except Exception:
            logger.warning(
                "Primary TTS failed, switching to fallback (%s)",
                type(self._fallback).__name__,
            )
            self._using_fallback = True
            await self._fallback.speak(text)

    def interrupt(self) -> None:
        super().interrupt()
        self.active_engine.interrupt()
