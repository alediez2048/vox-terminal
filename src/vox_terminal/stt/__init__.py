"""Speech-to-text engine package."""

from __future__ import annotations

from vox_terminal.config import STTSettings
from vox_terminal.stt.base import STTEngine, TranscriptionResult
from vox_terminal.stt.openai_stt import OpenAISTT
from vox_terminal.stt.whisper_local import WhisperLocalSTT

__all__ = [
    "OpenAISTT",
    "STTEngine",
    "TranscriptionResult",
    "WhisperLocalSTT",
    "create_stt_engine",
]


def create_stt_engine(settings: STTSettings) -> STTEngine:
    """Factory: create an STT engine from settings."""
    if settings.engine == "whisper_local":
        return WhisperLocalSTT(settings)
    if settings.engine == "openai":
        return OpenAISTT(settings)
    msg = f"Unknown STT engine: {settings.engine!r}"
    raise ValueError(msg)
