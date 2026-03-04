"""Local Whisper STT engine using faster-whisper."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from vox_terminal.config import STTSettings
from vox_terminal.stt.base import STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperLocalSTT(STTEngine):
    """Speech-to-text engine backed by faster-whisper running locally.

    The model is loaded lazily on the first call to :meth:`transcribe` so
    that construction is lightweight.
    """

    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the faster-whisper model (called once, on first transcribe)."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute=%s)",
            self._settings.whisper_model,
            self._settings.whisper_device,
            self._settings.whisper_compute_type,
        )
        model = WhisperModel(
            self._settings.whisper_model,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
        )
        return model

    @property
    def model_loaded(self) -> bool:
        """``True`` if the Whisper model has already been loaded."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe *audio* using the local Whisper model.

        The heavy ``model.transcribe()`` call is offloaded to a thread-pool
        executor so the event loop is never blocked.
        """
        if self._model is None:
            self._model = self._load_model()

        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(audio, beam_size=5),
        )

        # Materialise segments (they are a generator)
        segment_list = list(segments)
        text = " ".join(seg.text.strip() for seg in segment_list if seg.text.strip())
        avg_confidence = (
            sum(seg.avg_logprob for seg in segment_list) / len(segment_list)
            if segment_list
            else None
        )

        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=avg_confidence,
        )
