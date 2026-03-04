"""Energy-based (RMS) voice activity detection."""

from __future__ import annotations

import numpy as np

from vox_terminal.vad.base import VADEngine, VADResult


class EnergyVAD(VADEngine):
    """Simple RMS energy threshold VAD.

    This extracts the existing logic from ``audio_capture.py`` into a
    pluggable engine.
    """

    def __init__(self, threshold: float = 0.01) -> None:
        self._threshold = threshold

    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> VADResult:
        rms = float(np.sqrt(np.mean(chunk**2)))
        return VADResult(
            is_speech=rms > self._threshold,
            confidence=min(rms / self._threshold, 1.0) if self._threshold > 0 else 1.0,
        )
