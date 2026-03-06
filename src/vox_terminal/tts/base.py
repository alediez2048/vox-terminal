"""Abstract base class for TTS engines with sentence-buffered streaming."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time as _time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

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

_SENTENCE_BOUNDARIES: tuple[str, ...] = (".", "!", "?", "\n")
_FIRST_FLUSH_BOUNDARIES: tuple[str, ...] = (".", "!", "?", "\n", ",", ";", ":")


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


def _find_boundary(buffer: str, *, first_flush: bool) -> int:
    """Find earliest boundary index in ``buffer`` for streaming flushes."""
    separators = _FIRST_FLUSH_BOUNDARIES if first_flush else _SENTENCE_BOUNDARIES
    idx = -1
    for sep in separators:
        pos = buffer.find(sep)
        if pos != -1 and (idx == -1 or pos < idx):
            idx = pos
    return idx


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

    async def speak_streamed(
        self,
        chunks: AsyncIterator[str],
        *,
        on_event: Callable[[str, float], None] | None = None,
    ) -> None:
        """Stream TTS with first-clause flush and producer/consumer overlap."""
        def _emit(event_name: str) -> None:
            if on_event is not None:
                on_event(event_name, _time.monotonic())

        self._interrupted = False
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        buffer = ""
        spoken_sentences = 0
        spoken_chars = 0
        first_flush_emitted = False
        first_audio_emitted = False
        t0 = _time.monotonic()

        async def _queue_phrase(raw_text: str) -> None:
            nonlocal first_flush_emitted
            phrase = _sanitize_for_speech(raw_text.strip())
            if not phrase:
                return
            if not first_flush_emitted:
                first_flush_emitted = True
                _emit("tts_first_flush_ms")
            await queue.put(phrase)

        async def _producer() -> None:
            nonlocal buffer
            try:
                async for chunk in chunks:
                    if self._interrupted:
                        break
                    buffer += chunk
                    while not self._interrupted:
                        idx = _find_boundary(buffer, first_flush=not first_flush_emitted)
                        if idx == -1:
                            break
                        phrase = buffer[: idx + 1]
                        buffer = buffer[idx + 1 :]
                        await _queue_phrase(phrase)
                if not self._interrupted:
                    await _queue_phrase(buffer)
            finally:
                await queue.put(None)

        async def _consumer() -> None:
            nonlocal first_audio_emitted, spoken_sentences, spoken_chars
            while True:
                phrase = await queue.get()
                if phrase is None:
                    break
                if self._interrupted:
                    continue
                if not first_audio_emitted:
                    first_audio_emitted = True
                    _emit("tts_first_audio_ms")
                await self.speak(phrase)
                spoken_sentences += 1
                spoken_chars += len(phrase)

        producer_task = asyncio.create_task(_producer())
        consumer_task = asyncio.create_task(_consumer())
        try:
            await asyncio.gather(producer_task, consumer_task)
        finally:
            for task in (producer_task, consumer_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            _emit("tts_end_ms")
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
