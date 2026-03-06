"""Voice activity detection package."""

from __future__ import annotations

import logging
from typing import Literal

from vox_terminal.vad.base import VADEngine, VADResult
from vox_terminal.vad.energy import EnergyVAD

logger = logging.getLogger(__name__)


def create_vad_engine(
    engine: Literal["silero", "energy"] = "silero",
    threshold: float = 0.5,
    energy_threshold: float = 0.01,
) -> VADEngine:
    """Factory: create a VAD engine, falling back to energy if silero unavailable."""
    if engine == "silero":
        try:
            import torch  # noqa: F401

            from vox_terminal.vad.silero import SileroVAD

            return SileroVAD(threshold=threshold)
        except ImportError:
            logger.warning(
                "silero-vad or torch not installed — falling back to energy VAD. "
                "Install with: pip install 'vox-terminal[vad]'"
            )
            return EnergyVAD(threshold=energy_threshold)

    return EnergyVAD(threshold=energy_threshold)


__all__ = [
    "EnergyVAD",
    "VADEngine",
    "VADResult",
    "create_vad_engine",
]
