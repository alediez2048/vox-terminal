"""Tests for the TTS engine module."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vox_terminal.config import TTSSettings
from vox_terminal.tts import create_tts_engine
from vox_terminal.tts.base import FallbackTTSEngine
from vox_terminal.tts.base import TTSEngine as BaseTTSEngine
from vox_terminal.tts.elevenlabs_tts import ElevenLabsTTS
from vox_terminal.tts.macos_say import MacOSSayTTS
from vox_terminal.tts.openai_tts import OpenAITTS
from vox_terminal.tts.piper import PiperTTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConcreteTTS(BaseTTSEngine):
    """Minimal concrete implementation for testing the base class."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


async def _async_chunks(texts: list[str]) -> AsyncIterator[str]:
    """Yield *texts* as an async iterator."""
    for t in texts:
        yield t


# ---------------------------------------------------------------------------
# speak_streamed sentence buffering
# ---------------------------------------------------------------------------


class TestSpeakStreamed:
    async def test_splits_on_period(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["Hello world.", " How are you?"]))
        assert engine.spoken == ["Hello world.", "How are you?"]

    async def test_splits_on_exclamation(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["Wow!", " Amazing!"]))
        assert engine.spoken == ["Wow!", "Amazing!"]

    async def test_splits_on_question_mark(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["Really?", " Yes."]))
        assert engine.spoken == ["Really?", "Yes."]

    async def test_splits_on_newline(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["Line one\nLine two"]))
        assert engine.spoken == ["Line one", "Line two"]

    async def test_flushes_remaining_buffer(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["no boundary here"]))
        assert engine.spoken == ["no boundary here"]

    async def test_empty_stream(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks([]))
        assert engine.spoken == []

    async def test_multiple_sentences_single_chunk(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["One. Two! Three?"]))
        assert engine.spoken == ["One.", "Two!", "Three?"]

    async def test_sentence_split_across_chunks(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["Hel", "lo wor", "ld. Bye."]))
        assert engine.spoken == ["Hello world.", "Bye."]

    async def test_first_flush_can_trigger_on_clause_boundary(self) -> None:
        engine = ConcreteTTS()
        await engine.speak_streamed(_async_chunks(["One, two, three."]))
        assert engine.spoken == ["One,", "two, three."]

    async def test_emits_stream_timing_events(self) -> None:
        engine = ConcreteTTS()
        events: list[str] = []
        await engine.speak_streamed(
            _async_chunks(["Hello world."]),
            on_event=lambda name, _at: events.append(name),
        )
        assert events[0] == "tts_first_flush_ms"
        assert events[1] == "tts_first_audio_ms"
        assert events[-1] == "tts_end_ms"

    async def test_producer_continues_while_consumer_is_speaking(self) -> None:
        timeline: list[str] = []

        async def timed_chunks() -> AsyncIterator[str]:
            for idx in range(3):
                timeline.append(f"yield:{idx}")
                yield f"Part {idx}."

        class SlowTTS(ConcreteTTS):
            async def speak(self, text: str) -> None:
                timeline.append(f"speak_start:{text}")
                await asyncio.sleep(0.01)
                timeline.append(f"speak_end:{text}")
                self.spoken.append(text)

        engine = SlowTTS()
        await engine.speak_streamed(timed_chunks())
        # Producer should emit later chunks before all speaks finish.
        assert timeline.index("yield:1") < timeline.index("speak_end:Part 0.")


# ---------------------------------------------------------------------------
# MacOSSayTTS
# ---------------------------------------------------------------------------


class TestMacOSSayTTS:
    async def test_speak_calls_subprocess(self) -> None:
        settings = TTSSettings(macos_voice="Alex", macos_rate=180)
        engine = MacOSSayTTS(settings)

        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch(
            "vox_terminal.tts.macos_say.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ) as mock_exec:
            await engine.speak("Hello world")

            mock_exec.assert_called_once()
            args = mock_exec.call_args
            # Positional args: "say", "-v", voice, "-r", rate, text
            assert args[0][0] == "say"
            assert args[0][1] == "-v"
            assert args[0][2] == "Alex"
            assert args[0][3] == "-r"
            assert args[0][4] == "180"
            assert args[0][5] == "Hello world"
            mock_proc.wait.assert_awaited_once()

    async def test_subprocess_timeout_kills_process(self) -> None:
        """If the playback process hangs, it should be killed after timeout."""
        settings = TTSSettings(macos_voice="Alex", macos_rate=180)
        engine = MacOSSayTTS(settings)

        mock_proc = AsyncMock()
        # wait() returns normally after kill
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.kill = MagicMock()

        with (
            patch(
                "vox_terminal.tts.macos_say.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
            patch("vox_terminal.tts.macos_say.asyncio.wait_for", side_effect=TimeoutError),
        ):
            await engine.speak("Hello world")

            mock_proc.kill.assert_called_once()
            assert engine._proc is None


# ---------------------------------------------------------------------------
# create_tts_engine factory
# ---------------------------------------------------------------------------


class TestCreateTTSEngine:
    def test_creates_macos_say(self) -> None:
        settings = TTSSettings(engine="macos_say")
        engine = create_tts_engine(settings)
        assert isinstance(engine, MacOSSayTTS)

    def test_creates_openai_with_fallback(self) -> None:
        settings = TTSSettings(engine="openai", openai_api_key="sk-test")
        engine = create_tts_engine(settings)
        assert isinstance(engine, FallbackTTSEngine)
        assert isinstance(engine._primary, OpenAITTS)
        assert isinstance(engine._fallback, MacOSSayTTS)

    def test_creates_piper(self) -> None:
        settings = TTSSettings(engine="piper")
        engine = create_tts_engine(settings)
        assert isinstance(engine, PiperTTS)

    def test_creates_elevenlabs_with_fallback(self) -> None:
        settings = TTSSettings(engine="elevenlabs", elevenlabs_api_key="test-key")
        engine = create_tts_engine(settings)
        assert isinstance(engine, FallbackTTSEngine)
        assert isinstance(engine._primary, ElevenLabsTTS)
        assert isinstance(engine._fallback, MacOSSayTTS)


# ---------------------------------------------------------------------------
# ElevenLabsTTS
# ---------------------------------------------------------------------------


class TestElevenLabsTTS:
    async def test_speak_buffered_with_afplay(self) -> None:
        """Without ffplay, should buffer bytes and play with afplay."""
        settings = TTSSettings(
            engine="elevenlabs",
            elevenlabs_api_key="test-key",
            elevenlabs_voice_id="test-voice",
            elevenlabs_model_id="eleven_flash_v2_5",
        )

        with patch("vox_terminal.tts.elevenlabs_tts.shutil.which", return_value=None):
            engine = ElevenLabsTTS(settings)

        assert not engine._use_ffplay

        async def fake_response() -> AsyncIterator[bytes]:
            yield b"fake-audio-data"

        mock_convert = MagicMock(return_value=fake_response())

        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with (
            patch.object(engine._client.text_to_speech, "convert", mock_convert),
            patch(
                "vox_terminal.tts.elevenlabs_tts.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await engine.speak("Hello world")

            mock_convert.assert_called_once_with(
                text="Hello world",
                voice_id="test-voice",
                model_id="eleven_flash_v2_5",
            )
            mock_exec.assert_called_once()
            assert mock_exec.call_args[0][0] == "afplay"
            mock_proc.wait.assert_awaited_once()

    async def test_speak_streaming_with_ffplay(self) -> None:
        """With ffplay available (non-macOS), should pipe chunks to stdin."""
        settings = TTSSettings(
            engine="elevenlabs",
            elevenlabs_api_key="test-key",
            elevenlabs_voice_id="test-voice",
            elevenlabs_model_id="eleven_flash_v2_5",
        )

        with (
            patch("vox_terminal.tts.elevenlabs_tts.platform.system", return_value="Linux"),
            patch("vox_terminal.tts.elevenlabs_tts.shutil.which", return_value="/usr/bin/ffplay"),
        ):
            engine = ElevenLabsTTS(settings)

        assert engine._use_ffplay

        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def fake_response() -> AsyncIterator[bytes]:
            for c in chunks:
                yield c

        mock_convert = MagicMock(return_value=fake_response())

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()

        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = None

        with (
            patch.object(engine._client.text_to_speech, "convert", mock_convert),
            patch(
                "vox_terminal.tts.elevenlabs_tts.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await engine.speak("Hello streaming")

            mock_exec.assert_called_once()
            assert mock_exec.call_args[0][0] == "ffplay"
            # Verify chunks were written to stdin
            assert mock_stdin.write.call_count == 3
            written = [call.args[0] for call in mock_stdin.write.call_args_list]
            assert written == chunks
            mock_stdin.close.assert_called_once()

    async def test_ffplay_fallback_when_not_available(self) -> None:
        """shutil.which returning None should disable streaming."""
        settings = TTSSettings(
            engine="elevenlabs",
            elevenlabs_api_key="test-key",
        )
        with patch("vox_terminal.tts.elevenlabs_tts.shutil.which", return_value=None):
            engine = ElevenLabsTTS(settings)
        assert not engine._use_ffplay

    async def test_interrupt_during_streaming(self) -> None:
        """Interrupting during streaming should stop chunk delivery."""
        settings = TTSSettings(
            engine="elevenlabs",
            elevenlabs_api_key="test-key",
        )
        with patch("vox_terminal.tts.elevenlabs_tts.shutil.which", return_value="/usr/bin/ffplay"):
            engine = ElevenLabsTTS(settings)

        chunks_delivered = 0

        async def slow_response() -> AsyncIterator[bytes]:
            nonlocal chunks_delivered
            for _i in range(10):
                if engine._interrupted:
                    break
                chunks_delivered += 1
                yield b"chunk"

        mock_convert = MagicMock(return_value=slow_response())

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()
        mock_stdin.close = MagicMock()

        mock_proc = AsyncMock()
        mock_proc.stdin = mock_stdin
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = None

        with (
            patch.object(engine._client.text_to_speech, "convert", mock_convert),
            patch(
                "vox_terminal.tts.elevenlabs_tts.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
        ):
            # Interrupt after a short delay
            engine.interrupt()
            await engine.speak("Test interrupt")
            # speak returns immediately because _interrupted is True

    async def test_speak_returns_early_when_interrupted(self) -> None:
        """When _interrupted is True, speak does not call the API."""
        settings = TTSSettings(
            engine="elevenlabs",
            elevenlabs_api_key="test-key",
        )
        with patch("vox_terminal.tts.elevenlabs_tts.shutil.which", return_value=None):
            engine = ElevenLabsTTS(settings)
        engine._interrupted = True

        with patch.object(
            engine._client.text_to_speech,
            "convert",
            new_callable=MagicMock,
        ) as mock_convert:
            await engine.speak("Hello world")
            mock_convert.assert_not_called()

    def test_interrupt_terminates_process(self) -> None:
        """Interrupt calls terminate() on running playback process."""
        settings = TTSSettings(
            engine="elevenlabs",
            elevenlabs_api_key="test-key",
        )
        with patch("vox_terminal.tts.elevenlabs_tts.shutil.which", return_value=None):
            engine = ElevenLabsTTS(settings)

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()

        with patch("vox_terminal.tts.elevenlabs_tts.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.call_later = MagicMock()
            engine._proc = mock_proc
            engine.interrupt()

            mock_proc.terminate.assert_called_once()
            mock_loop.return_value.call_later.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAITTS
# ---------------------------------------------------------------------------


class TestOpenAITTS:
    async def test_speak_mocks_api_and_afplay(self) -> None:
        """Speak calls OpenAI API and spawns afplay for playback."""
        settings = TTSSettings(
            engine="openai",
            openai_api_key="sk-test",
            openai_voice="alloy",
            openai_model="tts-1",
        )
        engine = OpenAITTS(settings)

        mock_response = MagicMock()
        mock_response.content = b"fake-mp3-data"

        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = None

        with (
            patch.object(
                engine._client.audio.speech,
                "create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_create,
            patch(
                "vox_terminal.tts.openai_tts.asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
        ):
            await engine.speak("Hello world")

            mock_create.assert_awaited_once_with(
                model="tts-1",
                voice="alloy",
                input="Hello world",
            )
            mock_exec.assert_called_once()
            assert mock_exec.call_args[0][0] == "afplay"
            mock_proc.wait.assert_awaited_once()

    async def test_speak_returns_early_when_interrupted(self) -> None:
        """When _interrupted is True, speak does not call the API."""
        settings = TTSSettings(
            engine="openai",
            openai_api_key="sk-test",
        )
        engine = OpenAITTS(settings)
        engine._interrupted = True

        with patch.object(
            engine._client.audio.speech,
            "create",
            new_callable=AsyncMock,
        ) as mock_create:
            await engine.speak("Hello world")
            mock_create.assert_not_called()

    def test_interrupt_terminates_process(self) -> None:
        """Interrupt calls terminate() on running afplay process."""
        settings = TTSSettings(
            engine="openai",
            openai_api_key="sk-test",
        )
        engine = OpenAITTS(settings)

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.terminate = MagicMock()

        engine._proc = mock_proc
        engine.interrupt()

        mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# PiperTTS raises NotImplementedError
# ---------------------------------------------------------------------------


class TestPiperTTS:
    async def test_speak_raises_not_implemented(self) -> None:
        settings = TTSSettings(engine="piper")
        engine = PiperTTS(settings)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await engine.speak("Hello")


# ---------------------------------------------------------------------------
# FallbackTTSEngine
# ---------------------------------------------------------------------------


class TestFallbackTTSEngine:
    async def test_uses_primary_when_healthy(self) -> None:
        primary = ConcreteTTS()
        fallback = ConcreteTTS()
        engine = FallbackTTSEngine(primary, fallback)

        await engine.speak("Hello")
        assert primary.spoken == ["Hello"]
        assert fallback.spoken == []
        assert not engine._using_fallback

    async def test_switches_to_fallback_on_error(self) -> None:
        primary = ConcreteTTS()
        fallback = ConcreteTTS()

        # Make primary fail
        async def _fail(text: str) -> None:
            raise ConnectionError("API down")

        primary.speak = _fail  # type: ignore[assignment]
        engine = FallbackTTSEngine(primary, fallback)

        await engine.speak("Hello")
        assert fallback.spoken == ["Hello"]
        assert engine._using_fallback

    async def test_stays_on_fallback_after_switch(self) -> None:
        primary = ConcreteTTS()
        fallback = ConcreteTTS()

        call_count = 0

        async def _fail_once(text: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("API down")
            primary.spoken.append(text)

        primary.speak = _fail_once  # type: ignore[assignment]
        engine = FallbackTTSEngine(primary, fallback)

        await engine.speak("First")  # triggers fallback
        await engine.speak("Second")  # stays on fallback

        assert fallback.spoken == ["First", "Second"]
        assert primary.spoken == []

    def test_interrupt_delegates_to_active(self) -> None:
        primary = ConcreteTTS()
        fallback = ConcreteTTS()
        engine = FallbackTTSEngine(primary, fallback)

        engine.interrupt()
        assert engine._interrupted
        assert primary._interrupted  # delegates to active (primary)
