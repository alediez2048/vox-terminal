"""Piper TTS engine (placeholder)."""

from __future__ import annotations

from vox_terminal.config import TTSSettings
from vox_terminal.tts.base import TTSEngine


class PiperTTS(TTSEngine):
    """Placeholder for the Piper local TTS engine.

    Piper (https://github.com/rhasspy/piper) support is planned but not yet
    implemented.  Instantiating this class is fine, but calling ``speak()``
    will raise ``NotImplementedError``.
    """

    def __init__(self, settings: TTSSettings) -> None:
        self._model_path = settings.piper_model_path

    async def speak(self, text: str) -> None:
        """Not implemented yet."""
        raise NotImplementedError(
            "PiperTTS is not yet implemented. "
            "Use 'macos_say' or 'openai' as your TTS engine instead."
        )
