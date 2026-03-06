"""Tests for the VAD (voice activity detection) module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from vox_terminal.vad import create_vad_engine
from vox_terminal.vad.base import VADEngine, VADResult
from vox_terminal.vad.energy import EnergyVAD

# ---------------------------------------------------------------------------
# EnergyVAD
# ---------------------------------------------------------------------------


class TestEnergyVAD:
    def test_loud_audio_detected_as_speech(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        # Loud sine wave
        chunk = np.sin(np.linspace(0, 2 * np.pi, 1600)).astype(np.float32)
        result = vad.is_speech(chunk, 16000)
        assert result.is_speech is True
        assert result.confidence > 0

    def test_silence_not_detected_as_speech(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        chunk = np.zeros(1600, dtype=np.float32)
        result = vad.is_speech(chunk, 16000)
        assert result.is_speech is False

    def test_threshold_boundary(self) -> None:
        vad = EnergyVAD(threshold=0.5)
        # Very quiet noise — below threshold
        chunk = np.full(1600, 0.001, dtype=np.float32)
        result = vad.is_speech(chunk, 16000)
        assert result.is_speech is False

    def test_is_vad_engine(self) -> None:
        vad = EnergyVAD()
        assert isinstance(vad, VADEngine)

    def test_returns_vad_result(self) -> None:
        vad = EnergyVAD()
        chunk = np.zeros(1600, dtype=np.float32)
        result = vad.is_speech(chunk, 16000)
        assert isinstance(result, VADResult)


# ---------------------------------------------------------------------------
# SileroVAD (mocked — avoids requiring torch)
# ---------------------------------------------------------------------------


class TestSileroVAD:
    def test_speech_detected_with_mocked_model(self) -> None:
        # Create a mock torch module so SileroVAD can import
        mock_torch = MagicMock()
        mock_torch.from_numpy = MagicMock(return_value="fake-tensor")
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from vox_terminal.vad.silero import SileroVAD

            vad = SileroVAD(threshold=0.5)
            mock_model = MagicMock(return_value=0.9)
            vad._model = mock_model

            chunk = np.random.randn(1600).astype(np.float32)
            result = vad.is_speech(chunk, 16000)

        assert result.is_speech is True
        assert result.confidence == 0.9

    def test_silence_with_mocked_model(self) -> None:
        mock_torch = MagicMock()
        mock_torch.from_numpy = MagicMock(return_value="fake-tensor")
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from vox_terminal.vad.silero import SileroVAD

            vad = SileroVAD(threshold=0.5)
            mock_model = MagicMock(return_value=0.1)
            vad._model = mock_model

            chunk = np.zeros(1600, dtype=np.float32)
            result = vad.is_speech(chunk, 16000)

        assert result.is_speech is False
        assert result.confidence == 0.1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateVADEngine:
    def test_energy_engine(self) -> None:
        engine = create_vad_engine(engine="energy", energy_threshold=0.02)
        assert isinstance(engine, EnergyVAD)
        assert engine._threshold == 0.02

    def test_silero_fallback_when_not_installed(self) -> None:
        """When silero import fails, factory should fall back to EnergyVAD."""
        # Temporarily block the silero import inside the factory
        with patch.dict(sys.modules, {"vox_terminal.vad.silero": None}):
            engine = create_vad_engine(engine="silero")
        assert isinstance(engine, EnergyVAD)

    def test_silero_when_available(self) -> None:
        """If silero imports succeed, we should get a SileroVAD instance."""
        mock_torch = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            try:
                from vox_terminal.vad.silero import SileroVAD

                engine = create_vad_engine(engine="silero", threshold=0.6)
                assert isinstance(engine, SileroVAD)
            except ImportError:
                engine = create_vad_engine(engine="silero", threshold=0.6)
                assert isinstance(engine, EnergyVAD)
