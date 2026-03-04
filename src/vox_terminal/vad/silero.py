"""Silero VAD engine — ML-based voice activity detection."""

from __future__ import annotations

import logging

import numpy as np

from vox_terminal.vad.base import VADEngine, VADResult

logger = logging.getLogger(__name__)


class SileroVAD(VADEngine):
    """Voice activity detection using Silero VAD.

    Requires ``silero-vad`` and ``torch`` to be installed (available via
    the ``vad`` optional dependency group).

    The model is loaded lazily on first call to :meth:`is_speech`.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the Silero VAD model."""
        import torch

        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        logger.info("Silero VAD model loaded")

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> VADResult:
        import torch

        if self._model is None:
            self._load_model()

        # Silero expects float32 tensor, mono, 16kHz or 8kHz
        audio = chunk.flatten().astype(np.float32)
        tensor = torch.from_numpy(audio)

        confidence = float(self._model(tensor, sample_rate))
        return VADResult(
            is_speech=confidence >= self._threshold,
            confidence=confidence,
        )
