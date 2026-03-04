"""Abstract base classes for speech-to-text engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text transcription."""

    text: str
    language: str | None = None
    confidence: float | None = None


class STTEngine(ABC):
    """Abstract base for all STT provider engines."""

    @abstractmethod
    async def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe *audio* (float32 numpy array) and return the result."""
        ...  # pragma: no cover
