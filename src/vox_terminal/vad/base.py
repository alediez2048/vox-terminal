"""Abstract base for voice activity detection engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VADResult:
    """Result of a single VAD check."""

    is_speech: bool
    confidence: float = 1.0


class VADEngine(ABC):
    """Abstract base class for voice activity detection."""

    @abstractmethod
    def is_speech(self, chunk: np.ndarray, sample_rate: int) -> VADResult:
        """Return whether *chunk* contains speech."""
        ...
