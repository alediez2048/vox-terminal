"""OpenAI TTS engine."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time as _time
from pathlib import Path

from openai import AsyncOpenAI

from vox_terminal.config import TTSSettings
from vox_terminal.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class OpenAITTS(TTSEngine):
    """TTS engine using the OpenAI Audio API."""

    def __init__(self, settings: TTSSettings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._voice = settings.openai_voice
        self._model = settings.openai_model
        self._proc: asyncio.subprocess.Process | None = None

    def interrupt(self) -> None:
        """Kill the running ``afplay`` process and stop further speech."""
        super().interrupt()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()

    async def speak(self, text: str) -> None:
        """Synthesise *text* via OpenAI and play the result with ``afplay``."""
        if self._interrupted:
            return

        t0 = _time.monotonic()
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,  # type: ignore[arg-type]
            input=text,
        )
        logger.debug("OpenAI TTS API: %.0fms", (_time.monotonic() - t0) * 1000)

        # Write audio to a temporary file and play it
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)  # noqa: SIM115
        tmp_path = Path(tmp.name)
        try:
            tmp.write(response.content)
            tmp.close()

            self._proc = await asyncio.create_subprocess_exec(
                "afplay",
                str(tmp_path),
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
        finally:
            tmp_path.unlink(missing_ok=True)
