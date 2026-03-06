"""OpenAI Whisper API STT engine."""

from __future__ import annotations

import io
import logging
import struct

import numpy as np
from openai import AsyncOpenAI

from vox_terminal.config import STTSettings
from vox_terminal.stt.base import STTEngine, TranscriptionResult

logger = logging.getLogger(__name__)


class OpenAISTT(STTEngine):
    """Speech-to-text engine using the OpenAI Whisper API (``whisper-1``)."""

    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # WAV encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert a float32 numpy array to an in-memory WAV file (PCM 16-bit).

        Manually constructs the RIFF/WAV header using :mod:`struct` so
        that ``scipy`` is not required.
        """
        # Convert float32 [-1.0, 1.0] -> int16
        audio_clipped = np.clip(audio, -1.0, 1.0)
        pcm_data = (audio_clipped * 32767).astype(np.int16).tobytes()

        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)
        chunk_size = 36 + data_size  # 36 = size of header minus RIFF/size fields

        header = struct.pack(
            "<4sI4s"  # RIFF header
            "4sIHHIIHH"  # fmt  sub-chunk
            "4sI",  # data sub-chunk header
            b"RIFF",
            chunk_size,
            b"WAVE",
            b"fmt ",
            16,  # fmt chunk size (PCM)
            1,  # audio format (1 = PCM)
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        return header + pcm_data

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe *audio* via the OpenAI ``whisper-1`` API."""
        wav_bytes = self._numpy_to_wav_bytes(audio, sample_rate)

        wav_file = io.BytesIO(wav_bytes)
        wav_file.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_file,
        )

        return TranscriptionResult(
            text=response.text,
            language=None,
            confidence=None,
        )
