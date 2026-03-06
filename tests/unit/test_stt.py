"""Tests for the STT engine modules."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vox_terminal.config import STTSettings
from vox_terminal.stt import TranscriptionResult, create_stt_engine
from vox_terminal.stt.openai_stt import OpenAISTT
from vox_terminal.stt.whisper_local import WhisperLocalSTT

# ------------------------------------------------------------------
# TranscriptionResult dataclass
# ------------------------------------------------------------------


class TestTranscriptionResult:
    def test_defaults(self) -> None:
        r = TranscriptionResult(text="hello world")
        assert r.text == "hello world"
        assert r.language is None
        assert r.confidence is None

    def test_all_fields(self) -> None:
        r = TranscriptionResult(text="hi", language="en", confidence=0.95)
        assert r.text == "hi"
        assert r.language == "en"
        assert r.confidence == 0.95


# ------------------------------------------------------------------
# WhisperLocalSTT
# ------------------------------------------------------------------


class TestWhisperLocalSTT:
    def test_lazy_loading(self) -> None:
        """Model should NOT be loaded at construction time."""
        settings = STTSettings()
        engine = WhisperLocalSTT(settings)
        assert engine.model_loaded is False

    async def test_transcribe_loads_model_and_returns_result(self) -> None:
        """On first transcribe, the model is loaded, then transcription runs."""
        settings = STTSettings()
        engine = WhisperLocalSTT(settings)

        # Mock segment objects
        seg = MagicMock()
        seg.text = " hello world "
        seg.avg_logprob = -0.3

        info = MagicMock()
        info.language = "en"

        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([seg], info)

        with patch.object(engine, "_load_model", return_value=fake_model):
            audio = np.zeros(16000, dtype=np.float32)
            result = await engine.transcribe(audio, sample_rate=16000)

        assert engine.model_loaded is True
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.confidence is not None

    async def test_transcribe_empty_segments(self) -> None:
        """Transcription with no segments returns empty text."""
        settings = STTSettings()
        engine = WhisperLocalSTT(settings)

        info = MagicMock()
        info.language = "en"

        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([], info)

        with patch.object(engine, "_load_model", return_value=fake_model):
            audio = np.zeros(16000, dtype=np.float32)
            result = await engine.transcribe(audio)

        assert result.text == ""
        assert result.confidence is None


# ------------------------------------------------------------------
# OpenAISTT
# ------------------------------------------------------------------


class TestOpenAISTT:
    def test_numpy_to_wav_bytes_header(self) -> None:
        """Generated WAV bytes must start with a valid RIFF/WAVE header."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav = OpenAISTT._numpy_to_wav_bytes(audio, sample_rate=16000)

        # RIFF header
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

        # fmt sub-chunk
        assert wav[12:16] == b"fmt "
        fmt_size = struct.unpack_from("<I", wav, 16)[0]
        assert fmt_size == 16  # PCM

        audio_format = struct.unpack_from("<H", wav, 20)[0]
        assert audio_format == 1  # PCM

        num_channels = struct.unpack_from("<H", wav, 22)[0]
        assert num_channels == 1

        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 16000

        bits = struct.unpack_from("<H", wav, 34)[0]
        assert bits == 16

        # data sub-chunk
        assert wav[36:40] == b"data"
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == 5 * 2  # 5 samples * 2 bytes each

    def test_numpy_to_wav_bytes_roundtrip_length(self) -> None:
        """Total WAV length must be 44 (header) + num_samples * 2."""
        samples = 160
        audio = np.random.default_rng(42).uniform(-1, 1, samples).astype(np.float32)
        wav = OpenAISTT._numpy_to_wav_bytes(audio, sample_rate=16000)
        assert len(wav) == 44 + samples * 2

    async def test_transcribe_calls_openai_api(self) -> None:
        """transcribe() should call the OpenAI transcriptions endpoint."""
        settings = STTSettings(openai_api_key="sk-test")

        with patch("vox_terminal.stt.openai_stt.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "hello from openai"
            mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            engine = OpenAISTT(settings)
            audio = np.zeros(16000, dtype=np.float32)
            result = await engine.transcribe(audio, sample_rate=16000)

        assert result.text == "hello from openai"
        mock_client.audio.transcriptions.create.assert_awaited_once()


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


class TestCreateSttEngine:
    def test_whisper_local(self) -> None:
        settings = STTSettings(engine="whisper_local")
        engine = create_stt_engine(settings)
        assert isinstance(engine, WhisperLocalSTT)

    def test_openai(self) -> None:
        settings = STTSettings(engine="openai", openai_api_key="sk-test")
        with patch("vox_terminal.stt.openai_stt.AsyncOpenAI"):
            engine = create_stt_engine(settings)
        assert isinstance(engine, OpenAISTT)

    def test_unknown_engine(self) -> None:
        settings = STTSettings()
        # Force an invalid engine value by mutating the object
        object.__setattr__(settings, "engine", "unknown")
        with pytest.raises(ValueError, match="Unknown STT engine"):
            create_stt_engine(settings)
