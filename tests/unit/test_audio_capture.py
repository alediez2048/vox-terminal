"""Tests for the AudioCapture module."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox_terminal.audio_capture import AudioCapture, AudioCaptureError


@pytest.fixture
def mock_sd() -> MagicMock:
    """Return a mocked sounddevice module-level object."""
    mock = MagicMock()
    mock_stream = MagicMock()
    mock_stream.active = True
    mock.InputStream.return_value = mock_stream
    return mock


class TestAudioCaptureProperties:
    def test_defaults(self) -> None:
        cap = AudioCapture()
        assert cap.sample_rate == 16000
        assert cap.channels == 1

    def test_custom(self) -> None:
        cap = AudioCapture(sample_rate=44100, channels=2)
        assert cap.sample_rate == 44100
        assert cap.channels == 2

    def test_is_recording_false_initially(self) -> None:
        cap = AudioCapture()
        assert cap.is_recording is False


class TestAudioCaptureLifecycle:
    def test_start_stop_returns_array(self, mock_sd: MagicMock) -> None:
        """start() then stop() should return a float32 numpy array."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture(sample_rate=16000)
            cap.start()

            # Verify stream was created and started
            mock_sd.InputStream.assert_called_once()
            mock_sd.InputStream.return_value.start.assert_called_once()

            # Simulate some audio data arriving via the callback
            chunk1 = np.random.default_rng(0).random((160, 1), dtype=np.float32)
            chunk2 = np.random.default_rng(1).random((160, 1), dtype=np.float32)
            cap._audio_callback(chunk1, 160, None, None)
            cap._audio_callback(chunk2, 160, None, None)

            audio = cap.stop()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 320  # 160 + 160, flattened

    def test_is_recording_during_capture(self, mock_sd: MagicMock) -> None:
        """is_recording should be True between start() and stop()."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture()
            assert cap.is_recording is False

            cap.start()
            assert cap.is_recording is True

            cap.stop()
            assert cap.is_recording is False

    def test_stop_without_start_raises(self) -> None:
        """Calling stop() before start() should raise AudioCaptureError."""
        cap = AudioCapture()
        with pytest.raises(AudioCaptureError, match="no active recording"):
            cap.stop()

    def test_stop_with_no_chunks_returns_empty(self, mock_sd: MagicMock) -> None:
        """stop() with no audio data collected returns an empty array."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture()
            cap.start()
            audio = cap.stop()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == 0

    def test_start_when_sd_is_none_raises(self) -> None:
        """start() raises AudioCaptureError when sounddevice is unavailable."""
        with patch("vox_terminal.audio_capture.sd", None):
            cap = AudioCapture()
            with pytest.raises(AudioCaptureError, match="sounddevice is not available"):
                cap.start()

    def test_callback_copies_data(self, mock_sd: MagicMock) -> None:
        """The callback should copy indata so the original buffer can be reused."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture()
            cap.start()

            original = np.ones((160, 1), dtype=np.float32)
            cap._audio_callback(original, 160, None, None)

            # Mutate original — captured chunk should be unaffected
            original[:] = 0.0
            assert cap._chunks[0].sum() > 0

            cap.stop()


class TestRecordUntilSilence:
    async def test_stops_after_silence_following_speech(self, mock_sd: MagicMock) -> None:
        """Should auto-stop after silence_duration of quiet after speech."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture(sample_rate=16000)

            async def _simulate_speech() -> None:
                """Feed loud then quiet audio through the VAD callback."""
                await asyncio.sleep(0.05)
                # Simulate speech (loud)
                loud = np.full((160, 1), 0.5, dtype=np.float32)
                cap._vad_callback(loud, 160, None, None)
                # Simulate silence (quiet) — set silence_start
                quiet = np.full((160, 1), 0.001, dtype=np.float32)
                cap._vad_callback(quiet, 160, None, None)
                await asyncio.sleep(0.05)
                # Exceed silence_duration by manipulating _silence_start
                cap._silence_start = cap._silence_start - 1.0  # type: ignore[operator]
                cap._vad_callback(quiet, 160, None, None)

            task = asyncio.create_task(_simulate_speech())
            audio = await cap.record_until_silence(
                silence_threshold=0.01,
                silence_duration=0.5,
                max_duration=5.0,
            )
            await task
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    async def test_stops_at_max_duration(self, mock_sd: MagicMock) -> None:
        """Should auto-stop when max_duration is exceeded even without speech."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture(sample_rate=16000)

            async def _simulate_timeout() -> None:
                await asyncio.sleep(0.05)
                # Force recording_start into the past to exceed max_duration
                cap._recording_start -= 35.0
                quiet = np.full((160, 1), 0.001, dtype=np.float32)
                cap._vad_callback(quiet, 160, None, None)

            task = asyncio.create_task(_simulate_timeout())
            audio = await cap.record_until_silence(
                silence_threshold=0.01,
                silence_duration=0.5,
                max_duration=30.0,
            )
            await task
        assert isinstance(audio, np.ndarray)

    async def test_raises_when_sd_unavailable(self) -> None:
        """Should raise AudioCaptureError when sounddevice is None."""
        with patch("vox_terminal.audio_capture.sd", None):
            cap = AudioCapture()
            with pytest.raises(AudioCaptureError, match="sounddevice is not available"):
                await cap.record_until_silence()

    async def test_speech_not_detected_keeps_listening(self, mock_sd: MagicMock) -> None:
        """Quiet audio alone should not trigger stop (only max_duration does)."""
        with patch("vox_terminal.audio_capture.sd", mock_sd):
            cap = AudioCapture(sample_rate=16000)

            quiet = np.full((160, 1), 0.001, dtype=np.float32)

            async def _simulate_quiet_then_timeout() -> None:
                await asyncio.sleep(0.05)
                # Send quiet audio — should NOT trigger done
                cap._vad_callback(quiet, 160, None, None)
                assert not cap._speech_detected
                # Force max_duration to trigger
                cap._recording_start -= 35.0
                cap._vad_callback(quiet, 160, None, None)

            task = asyncio.create_task(_simulate_quiet_then_timeout())
            audio = await cap.record_until_silence(
                silence_threshold=0.01,
                silence_duration=0.5,
                max_duration=30.0,
            )
            await task
        assert not cap._speech_detected
