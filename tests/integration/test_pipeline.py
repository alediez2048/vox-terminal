"""Integration test — STT → LLM pipeline with mocked API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vox_terminal.config import LLMSettings, STTSettings, TTSSettings
from vox_terminal.llm import Message, create_llm_client
from vox_terminal.stt import create_stt_engine
from vox_terminal.tts import create_tts_engine


@pytest.mark.integration
class TestSTTToLLMPipeline:
    """Test the STT → LLM pipeline with mocked external services."""

    async def test_transcription_to_llm_response(self) -> None:
        """Simulate: audio → transcription → LLM question → response."""
        # 1. Mock STT: transcribe audio to text
        stt_settings = STTSettings(engine="whisper_local")
        stt = create_stt_engine(stt_settings)

        fake_audio = np.random.randn(16000).astype(np.float32)
        mock_segments = [MagicMock(text="What does this project do?", avg_logprob=-0.2)]
        mock_info = MagicMock(language="en")

        with patch.object(stt, "_model") as mock_model:
            mock_model.transcribe.return_value = (mock_segments, mock_info)
            stt._model = mock_model  # force loaded

            result = await stt.transcribe(fake_audio, 16000)
            assert result.text == "What does this project do?"

        # 2. Mock LLM: ask the transcribed question
        llm_settings = LLMSettings(api_key="test-key")
        llm = create_llm_client(llm_settings, project_context="A Python CLI tool.")

        # Mock the anthropic stream
        mock_stream_cm = AsyncMock()
        mock_response = AsyncMock()

        async def mock_text_stream():
            for chunk in ["This project ", "is a CLI tool ", "for developers."]:
                yield chunk

        mock_response.text_stream = mock_text_stream()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(llm._client.messages, "stream", return_value=mock_stream_cm):
            response = await llm.ask(result.text)
            assert "CLI tool" in response.content

    async def test_llm_with_history(self) -> None:
        """Test that conversation history is passed correctly."""
        llm_settings = LLMSettings(api_key="test-key")
        llm = create_llm_client(llm_settings, project_context="Test project.")

        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi! How can I help?"),
        ]

        mock_stream_cm = AsyncMock()
        mock_response = AsyncMock()

        async def mock_text_stream():
            yield "Sure, I can help."

        mock_response.text_stream = mock_text_stream()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(
            llm._client.messages, "stream", return_value=mock_stream_cm
        ) as mock_stream:
            response = await llm.ask("Tell me more", history=history)
            assert response.content == "Sure, I can help."

            # Verify history was included in the call
            call_kwargs = mock_stream.call_args[1]
            messages = call_kwargs["messages"]
            assert len(messages) == 3  # 2 history + 1 new
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello"


@pytest.mark.integration
class TestTTSStreamPipeline:
    """Test TTS streaming with sentence chunking."""

    async def test_llm_chunks_to_tts_sentences(self) -> None:
        """Simulate LLM streaming chunks being spoken as sentences."""
        tts_settings = TTSSettings(engine="macos_say")
        tts = create_tts_engine(tts_settings)

        spoken: list[str] = []

        async def mock_speak(text: str) -> None:
            spoken.append(text)

        tts.speak = mock_speak  # type: ignore[assignment]

        # Simulate LLM chunks
        async def llm_chunks():
            chunks = ["Hello! ", "This is ", "Vox-Terminal. ", "How can ", "I help?"]
            for c in chunks:
                yield c

        await tts.speak_streamed(llm_chunks())

        assert "Hello!" in spoken
        assert "This is Vox-Terminal." in spoken
        assert "How can I help?" in spoken
