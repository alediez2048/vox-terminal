"""Text-to-speech engine package."""

from __future__ import annotations

from vox_terminal.config import TTSSettings
from vox_terminal.tts.base import FallbackTTSEngine, TTSEngine
from vox_terminal.tts.elevenlabs_tts import ElevenLabsTTS
from vox_terminal.tts.macos_say import MacOSSayTTS
from vox_terminal.tts.openai_tts import OpenAITTS
from vox_terminal.tts.piper import PiperTTS

# Cloud engines that should auto-fallback to macOS say.
_CLOUD_ENGINES = {"elevenlabs", "openai"}


def create_tts_engine(settings: TTSSettings) -> TTSEngine:
    """Factory: create a TTS engine from configuration.

    Cloud engines (``elevenlabs``, ``openai``) are automatically wrapped with
    a :class:`FallbackTTSEngine` that falls back to macOS ``say`` on failure.

    Raises ``ValueError`` if ``settings.engine`` is not a recognised engine
    name.
    """
    if settings.engine == "macos_say":
        return MacOSSayTTS(settings)
    if settings.engine == "openai":
        primary: TTSEngine = OpenAITTS(settings)
    elif settings.engine == "piper":
        return PiperTTS(settings)
    elif settings.engine == "elevenlabs":
        primary = ElevenLabsTTS(settings)
    else:
        raise ValueError(f"Unknown TTS engine: {settings.engine!r}")

    # Wrap cloud engines with macOS say fallback
    fallback = MacOSSayTTS(settings)
    return FallbackTTSEngine(primary, fallback)


__all__ = [
    "ElevenLabsTTS",
    "FallbackTTSEngine",
    "MacOSSayTTS",
    "OpenAITTS",
    "PiperTTS",
    "TTSEngine",
    "create_tts_engine",
]
