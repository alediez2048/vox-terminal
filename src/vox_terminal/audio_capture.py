"""Audio capture module using sounddevice with optional silence detection."""

from __future__ import annotations

import asyncio
import logging
import threading
import time as _time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except OSError:  # pragma: no cover — missing PortAudio / no mic access
    sd = None  # type: ignore[assignment]
    logger.warning("sounddevice unavailable — audio capture disabled")


class AudioCaptureError(RuntimeError):
    """Raised when audio capture fails."""


class AudioCapture:
    """Audio recorder backed by sounddevice.

    Supports both push-to-talk (``start``/``stop``) and hands-free
    recording with automatic silence detection (``record_until_silence``).

    An optional :class:`~vox_terminal.vad.base.VADEngine` can be injected to
    replace the default RMS energy check with a more sophisticated model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        vad: Any | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunks: list[np.ndarray] = []
        self._stream: Any | None = None
        self._vad = vad  # optional VADEngine

        # VAD state (used by record_until_silence)
        self._speech_detected = False
        self._speech_started_at: float | None = None
        self._silence_start: float | None = None
        self._recording_start: float = 0.0
        self._silence_threshold: float = 0.01
        self._silence_duration: float = 1.5
        self._silence_duration_after_speech: float = 0.5
        self._adaptive_endpointing: bool = True
        self._speech_start_threshold: float | None = None
        self._speech_end_threshold: float | None = None
        self._max_duration: float = 30.0
        self._done_event: asyncio.Event | None = None
        self._speech_started_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_level_lock = threading.Lock()
        self._current_audio_level: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_recording(self) -> bool:
        """``True`` while the stream is open and recording."""
        return self._stream is not None and self._stream.active

    @property
    def speech_started(self) -> asyncio.Event | None:
        """Event that is set when speech is first detected during recording."""
        return self._speech_started_event

    @property
    def last_speech_started_at(self) -> float | None:
        """Monotonic timestamp when speech was first detected for latest recording."""
        return self._speech_started_at

    @property
    def audio_level(self) -> float:
        """Current audio level (0.0-1.0), updated from the VAD callback."""
        with self._audio_level_lock:
            return self._current_audio_level

    # ------------------------------------------------------------------
    # Push-to-talk recording
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Clear the buffer and begin recording from the microphone."""
        if sd is None:
            raise AudioCaptureError("sounddevice is not available — cannot record audio")
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.debug("Audio capture started (rate=%d, ch=%d)", self._sample_rate, self._channels)

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a float32 array."""
        if self._stream is None:
            raise AudioCaptureError("Cannot stop — no active recording")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.debug("Audio capture stopped — %d chunks collected", len(self._chunks))
        if not self._chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._chunks, axis=0).flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Hands-free recording with silence detection
    # ------------------------------------------------------------------

    async def record_until_silence(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.7,
        silence_duration_after_speech: float = 0.5,
        adaptive_endpointing: bool = True,
        speech_start_threshold: float | None = None,
        speech_end_threshold: float | None = None,
        max_duration: float = 30.0,
    ) -> np.ndarray:
        """Record audio and auto-stop after silence following speech.

        Waits for the user to start speaking (energy above *silence_threshold*),
        then stops once energy stays below the threshold for
        *silence_duration* seconds.  Recording also stops after
        *max_duration* seconds regardless.

        Returns the captured audio as a float32 numpy array.
        """
        if sd is None:
            raise AudioCaptureError("sounddevice is not available — cannot record audio")

        self._speech_detected = False
        self._speech_started_at = None
        self._silence_start = None
        self._recording_start = _time.monotonic()
        self._silence_threshold = silence_threshold
        self._silence_duration = silence_duration
        self._silence_duration_after_speech = silence_duration_after_speech
        self._adaptive_endpointing = adaptive_endpointing
        self._speech_start_threshold = speech_start_threshold
        self._speech_end_threshold = speech_end_threshold
        self._max_duration = max_duration
        self._done_event = asyncio.Event()
        self._speech_started_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()

        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            callback=self._vad_callback,
        )
        self._stream.start()
        logger.debug(
            "Hands-free recording started (threshold=%.4f, silence=%.1fs, post_speech_silence=%.1fs, adaptive=%s)",
            silence_threshold,
            silence_duration,
            silence_duration_after_speech,
            adaptive_endpointing,
        )

        await self._done_event.wait()
        return self.stop()

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: Any,
    ) -> None:
        """sounddevice callback for push-to-talk — just accumulates chunks."""
        if status:
            logger.warning("Audio callback status: %s", status)
        self._chunks.append(indata.copy())

    def _vad_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: Any,
    ) -> None:
        """sounddevice callback with voice-activity detection."""
        if status:
            logger.warning("Audio callback status: %s", status)
        self._chunks.append(indata.copy())

        now = _time.monotonic()

        # Hard cap on recording length
        if now - self._recording_start > self._max_duration:
            self._signal_done()
            return

        # Determine speech presence via injected VAD or legacy RMS
        if self._vad is not None:
            try:
                result = self._vad.is_speech(indata, self._sample_rate)
                is_speech = result.is_speech
                level = result.confidence
            except Exception:
                # Silero rejects chunks that are too short — fall back to RMS
                rms = float(np.sqrt(np.mean(indata**2)))
                is_speech = rms > self._silence_threshold
                level = min(1.0, rms * 10.0)
        else:
            rms = float(np.sqrt(np.mean(indata**2)))
            threshold = self._resolve_energy_threshold()
            is_speech = rms > threshold
            level = min(1.0, rms * 10.0)  # scale RMS to 0-1 range

        with self._audio_level_lock:
            self._current_audio_level = level

        if is_speech:
            # User is speaking
            if not self._speech_detected:
                self._speech_detected = True
                self._speech_started_at = now
                if self._loop is not None and self._speech_started_event is not None:
                    self._loop.call_soon_threadsafe(self._speech_started_event.set)
            self._silence_start = None
        elif self._speech_detected:
            # User was speaking but now it's quiet
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self._resolve_required_silence_duration(now):
                self._signal_done()

    def _resolve_energy_threshold(self) -> float:
        """Resolve RMS threshold with optional adaptive start/end values."""
        if not self._adaptive_endpointing:
            return self._silence_threshold
        if self._speech_detected:
            if self._speech_end_threshold is not None:
                return self._speech_end_threshold
            return max(0.001, self._silence_threshold * 0.8)
        if self._speech_start_threshold is not None:
            return self._speech_start_threshold
        return self._silence_threshold

    def _resolve_required_silence_duration(self, now: float) -> float:
        """Resolve post-utterance silence duration with adaptive hangover."""
        if not self._adaptive_endpointing:
            return self._silence_duration
        if self._speech_started_at is None:
            return self._silence_duration
        utterance_duration = now - self._speech_started_at
        if utterance_duration < 0.5:
            return self._silence_duration
        return min(self._silence_duration, self._silence_duration_after_speech)

    def cancel_recording(self) -> None:
        """Cancel an in-progress ``record_until_silence`` call."""
        self._signal_done()

    def _signal_done(self) -> None:
        """Thread-safe: set the done event on the async loop."""
        if self._loop is not None and self._done_event is not None:
            self._loop.call_soon_threadsafe(self._done_event.set)
