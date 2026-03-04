"""macOS 'say' command TTS engine."""

from __future__ import annotations

import asyncio
import logging
import time as _time

from vox_terminal.config import TTSSettings
from vox_terminal.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class MacOSSayTTS(TTSEngine):
    """TTS engine using the macOS built-in ``say`` command."""

    def __init__(self, settings: TTSSettings) -> None:
        self._voice = settings.macos_voice
        self._rate = settings.macos_rate
        self._proc: asyncio.subprocess.Process | None = None

    def interrupt(self) -> None:
        """Kill the running ``say`` process and stop further speech."""
        super().interrupt()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()

    async def speak(self, text: str) -> None:
        """Speak *text* using ``say -v <voice> -r <rate>``."""
        if self._interrupted:
            return

        t0 = _time.monotonic()
        self._proc = await asyncio.create_subprocess_exec(
            "say",
            "-v",
            self._voice,
            "-r",
            str(self._rate),
            text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=120.0)
        except TimeoutError:
            logger.warning("Playback timed out — killing process")
            self._proc.kill()
            await self._proc.wait()
        self._proc = None
        logger.debug("macOS say: %.0fms", (_time.monotonic() - t0) * 1000)
