"""ElevenLabs TTS engine with optional streaming via ffplay."""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import tempfile
import time as _time
from pathlib import Path

from elevenlabs.client import AsyncElevenLabs

from vox_terminal.config import TTSSettings
from vox_terminal.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class ElevenLabsTTS(TTSEngine):
    """TTS engine using the ElevenLabs API.

    If ``ffplay`` is available on PATH, audio chunks are piped to it as
    they arrive for lower first-chunk latency (~200ms).  Otherwise, all
    bytes are collected, written to a temp file, and played with ``afplay``.
    """

    def __init__(self, settings: TTSSettings) -> None:
        self._client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
        self._voice_id = settings.elevenlabs_voice_id
        self._model_id = settings.elevenlabs_model_id
        self._rate = settings.elevenlabs_speed
        self._output_format = settings.elevenlabs_output_format
        self._proc: asyncio.subprocess.Process | None = None
        # Prefer afplay on macOS (reliable); use ffplay only on other platforms
        has_afplay = platform.system() == "Darwin" and shutil.which("afplay") is not None
        self._use_ffplay = not has_afplay and shutil.which("ffplay") is not None
        if has_afplay:
            logger.debug("macOS detected — using afplay for playback")
        elif self._use_ffplay:
            logger.debug("ffplay detected — streaming mode enabled")
        else:
            logger.debug("No audio player found")

    def interrupt(self) -> None:
        """Kill the running playback process and stop further speech."""
        super().interrupt()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()

    async def speak(self, text: str) -> None:
        """Synthesise *text* via ElevenLabs and play it."""
        if self._interrupted:
            return

        if self._use_ffplay:
            await self._speak_streaming(text)
        else:
            await self._speak_buffered(text)

    # ------------------------------------------------------------------
    # Streaming path (ffplay)
    # ------------------------------------------------------------------

    async def _speak_streaming(self, text: str) -> None:
        """Pipe audio chunks to ``ffplay`` stdin as they arrive."""
        t0 = _time.monotonic()

        response = self._client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format=self._output_format,
        )

        self._proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        total_bytes = 0
        first_chunk = True
        async for chunk in response:
            if self._interrupted:
                break
            if first_chunk:
                logger.debug(
                    "ElevenLabs first chunk: %.0fms",
                    (_time.monotonic() - t0) * 1000,
                )
                first_chunk = False
            if self._proc.stdin is not None:
                self._proc.stdin.write(chunk)
                await self._proc.stdin.drain()
            total_bytes += len(chunk)

        # Close stdin to signal EOF to ffplay
        if self._proc.stdin is not None:
            self._proc.stdin.close()

        await self._proc.wait()
        self._proc = None
        logger.debug(
            "ElevenLabs streaming: %.0fms total, %d bytes",
            (_time.monotonic() - t0) * 1000,
            total_bytes,
        )

    # ------------------------------------------------------------------
    # Buffered path (afplay fallback)
    # ------------------------------------------------------------------

    async def _speak_buffered(self, text: str) -> None:
        """Collect all bytes then play via ``afplay``."""
        t0 = _time.monotonic()

        response = self._client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id,
            model_id=self._model_id,
        )

        audio_bytes = b""
        async for chunk in response:
            if self._interrupted:
                return
            audio_bytes += chunk

        api_ms = (_time.monotonic() - t0) * 1000
        logger.debug("ElevenLabs API: %.0fms, %d bytes", api_ms, len(audio_bytes))

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)  # noqa: SIM115
        tmp_path = Path(tmp.name)
        try:
            tmp.write(audio_bytes)
            tmp.close()

            self._proc = await asyncio.create_subprocess_exec(
                "afplay",
                "-r",
                str(self._rate),
                str(tmp_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await self._proc.wait()
            self._proc = None
        finally:
            tmp_path.unlink(missing_ok=True)
