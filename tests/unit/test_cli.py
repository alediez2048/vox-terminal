"""Tests for the CLI module."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from vox_terminal.cli import _ask_and_speak, _ask_once, _build_interactive_context_settings, app
from vox_terminal.config import ContextSettings, VoxTerminalSettings
from vox_terminal.llm.base import LLMClient
from vox_terminal.tts.base import TTSEngine
from vox_terminal.tui.state import DisplayState

runner = CliRunner()


class TestContextCommand:
    def test_context_preview(self) -> None:
        result = runner.invoke(app, ["context", "--preview"])
        assert result.exit_code == 0

    def test_context_no_preview(self) -> None:
        result = runner.invoke(app, ["context", "--no-preview"])
        assert result.exit_code == 0


class TestAskCommand:
    def test_ask_requires_text(self) -> None:
        result = runner.invoke(app, ["ask"])
        assert result.exit_code != 0

    @patch("vox_terminal.cli._ask_once", new_callable=AsyncMock)
    def test_ask_with_text(self, mock_ask: AsyncMock) -> None:
        mock_ask.return_value = ""
        result = runner.invoke(app, ["ask", "--text", "What is this?"])
        assert result.exit_code == 0


class TestMainCommand:
    @patch("vox_terminal.cli._ask_once", new_callable=AsyncMock)
    def test_main_with_text(self, mock_ask: AsyncMock) -> None:
        mock_ask.return_value = ""
        result = runner.invoke(app, ["--text", "Hello"])
        assert result.exit_code == 0


class TestServeCommand:
    @patch("vox_terminal.mcp_server.create_mcp_server")
    def test_serve_starts_server(self, mock_create: MagicMock) -> None:
        mock_server = MagicMock()
        mock_create.return_value = mock_server
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_server.run.assert_called_once_with(transport="stdio")


class TestLogsCommand:
    def test_logs_handles_missing_file(self, tmp_path) -> None:
        with patch.dict(
            os.environ,
            {"VOX_TERMINAL_GENERAL__LOG_FILE": str(tmp_path / "missing.log")},
            clear=False,
        ):
            result = runner.invoke(app, ["logs"])
        assert result.exit_code == 0
        assert "No log file found" in result.output

    def test_logs_prints_last_lines(self, tmp_path) -> None:
        log_path = tmp_path / "vox.log"
        log_path.write_text("line1\nline2\nline3\n")
        with patch.dict(
            os.environ,
            {"VOX_TERMINAL_GENERAL__LOG_FILE": str(log_path)},
            clear=False,
        ):
            result = runner.invoke(app, ["logs", "--lines", "2"])
        assert result.exit_code == 0
        assert "line2" in result.output
        assert "line3" in result.output
        assert "line1" not in result.output


class TestDiagnoseCommand:
    @patch("vox_terminal.cli._run_diagnostics", new_callable=AsyncMock)
    def test_diagnose_success(self, mock_diag: AsyncMock) -> None:
        mock_diag.return_value = [
            ("context", True, "ok", 10.0),
            ("stt", True, "ok", 10.0),
            ("llm", True, "ok", 10.0),
            ("tts", True, "ok", 10.0),
        ]
        result = runner.invoke(app, ["diagnose", "--no-audio"])
        assert result.exit_code == 0
        assert "CONTEXT" in result.output
        assert "OK" in result.output

    @patch("vox_terminal.cli._run_diagnostics", new_callable=AsyncMock)
    def test_diagnose_failure_exits_nonzero(self, mock_diag: AsyncMock) -> None:
        mock_diag.return_value = [
            ("context", True, "ok", 10.0),
            ("stt", False, "failed", 10.0),
        ]
        result = runner.invoke(app, ["diagnose", "--no-audio"])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    @patch("vox_terminal.cli._run_diagnostics", new_callable=AsyncMock)
    def test_diagnose_iterations_prints_summary(self, mock_diag: AsyncMock) -> None:
        mock_diag.return_value = [
            ("context", True, "ok", 10.0),
            ("stt", True, "ok", 8.0),
            ("llm", True, "ok", 12.0),
            ("tts", True, "ok", 7.0),
        ]
        result = runner.invoke(app, ["diagnose", "--no-audio", "--iterations", "2"])
        assert result.exit_code == 0
        assert "Iteration 1/2" in result.output
        assert "Iteration 2/2" in result.output
        assert "Benchmark summary" in result.output
        assert mock_diag.await_count == 2

    @patch("vox_terminal.cli._run_diagnostics", new_callable=AsyncMock)
    def test_diagnose_passes_mode_flags(self, mock_diag: AsyncMock) -> None:
        mock_diag.return_value = [
            ("context", True, "ok", 10.0),
            ("stt", True, "ok", 10.0),
            ("llm", True, "ok", 10.0),
            ("tts", True, "ok", 10.0),
        ]
        result = runner.invoke(
            app,
            ["diagnose", "--no-audio", "--text-only", "--minimal-context"],
        )
        assert result.exit_code == 0
        call_kwargs = mock_diag.call_args.kwargs
        assert call_kwargs["text_only"] is True
        assert call_kwargs["minimal_context"] is True


class TestTeeStream:
    async def test_tee_yields_and_prints(self) -> None:
        from vox_terminal.cli import _tee_stream

        async def fake_stream():
            for chunk in ["Hello ", "world"]:
                yield chunk

        collected = []
        async for chunk in _tee_stream(fake_stream()):
            collected.append(chunk)

        assert collected == ["Hello ", "world"]


class TestInteractiveContextProfile:
    def test_compact_profile_trims_heavy_sources(self) -> None:
        base = ContextSettings(
            interactive_compact_context=True,
            include_files=["src/**/*.py"],
            doc_patterns=["README.md"],
            read_config_files=True,
            read_full_readme=True,
        )
        compact = _build_interactive_context_settings(base)
        assert compact.read_config_files is False
        assert compact.read_full_readme is False
        assert compact.doc_patterns == []
        assert compact.include_files == []

    def test_compact_profile_can_be_disabled(self) -> None:
        base = ContextSettings(
            interactive_compact_context=False,
            read_config_files=True,
            read_full_readme=True,
            doc_patterns=["README.md"],
        )
        profile = _build_interactive_context_settings(base)
        assert profile.read_config_files is True
        assert profile.read_full_readme is True
        assert profile.doc_patterns == ["README.md"]


class TestAskAndSpeak:
    """Tests for _ask_and_speak helper."""

    async def test_returns_full_response_from_stream(self) -> None:
        """Chunks flow through llm.stream -> tts.speak_streamed and full text returned."""
        async def mock_stream(question: str, history: object = None) -> AsyncIterator[str]:
            for chunk in ["Hello ", "world", "."]:
                yield chunk

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = mock_stream

        spoken: list[str] = []

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                spoken.append(text)

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())

        assert result == "Hello world."
        assert spoken == ["Hello world."]

    async def test_passes_history_to_llm(self) -> None:
        """History is passed through to llm.stream."""
        from vox_terminal.llm import Message

        history = [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]

        async def mock_stream(question: str, history: object = None) -> AsyncIterator[str]:
            assert history is not None
            assert len(history) == 2
            yield "OK"

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = AsyncMock(side_effect=mock_stream)

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        await _ask_and_speak("Follow up", mock_llm, MockTTS(), history=history)
        mock_llm.stream.assert_called_once_with("Follow up", history=history)

    async def test_returns_empty_on_auth_error(self) -> None:
        """AuthenticationError returns empty string and prints message."""
        import anthropic

        resp = MagicMock()
        resp.status_code = 401
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                "Invalid key", response=resp, body={}
            )
        )

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())
        assert result == ""

    async def test_returns_empty_on_rate_limit(self) -> None:
        """RateLimitError returns empty string."""
        import anthropic

        resp = MagicMock()
        resp.status_code = 429
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = AsyncMock(
            side_effect=anthropic.RateLimitError(
                "Rate limited", response=resp, body={}
            )
        )

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())
        assert result == ""

    async def test_returns_empty_on_timeout(self) -> None:
        """TimeoutError returns empty string."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = AsyncMock(side_effect=TimeoutError("Timed out"))

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())
        assert result == ""

    async def test_returns_empty_on_api_error(self) -> None:
        """anthropic.APIError returns empty string."""
        import anthropic

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = AsyncMock(
            side_effect=anthropic.APIError(
                "Server error", request=MagicMock(), body={}
            )
        )

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())
        assert result == ""

    async def test_returns_empty_on_network_error(self) -> None:
        """httpx.HTTPError returns empty string."""
        import httpx

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())
        assert result == ""

    async def test_updates_display_state_when_provided(self) -> None:
        """When display_state is passed, phase and response_chunks are updated."""
        async def mock_stream(question: str, history: object = None) -> AsyncIterator[str]:
            yield "Hello "
            yield "world."

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = mock_stream

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                pass

        display_state = DisplayState(model_name="test")
        result = await _ask_and_speak(
            "Hi", mock_llm, MockTTS(), display_state=display_state
        )

        assert result == "Hello world."
        assert display_state.phase == "speaking"
        assert display_state.response_chunks == ["Hello ", "world."]

    async def test_returns_chunks_on_cancelled(self) -> None:
        """CancelledError (barge-in) returns collected chunks so far."""
        async def mock_stream(question: str, history: object = None) -> AsyncIterator[str]:
            yield "Partial "
            raise asyncio.CancelledError()

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.stream = mock_stream  # Use real async gen, not AsyncMock

        spoken: list[str] = []

        class MockTTS(TTSEngine):
            async def speak(self, text: str) -> None:
                spoken.append(text)

        result = await _ask_and_speak("Hi", mock_llm, MockTTS())
        assert result == "Partial "


class TestAskOnce:
    """Tests for _ask_once helper."""

    @pytest.fixture
    def test_settings(self, project_root) -> VoxTerminalSettings:
        return VoxTerminalSettings(
            general={"project_root": project_root},  # type: ignore[arg-type]
            llm={"api_key": "test-key"},  # type: ignore[arg-type]
            tts={"engine": "macos_say"},  # type: ignore[arg-type]
        )

    async def test_assembles_context_and_calls_ask_and_speak(
        self, test_settings: VoxTerminalSettings
    ) -> None:
        """_ask_once assembles context, creates llm/tts, and delegates to _ask_and_speak."""
        with (
            patch("vox_terminal.cli._ask_and_speak", new_callable=AsyncMock) as mock_ask,
            patch("vox_terminal.cli.create_llm_client") as mock_llm_factory,
            patch("vox_terminal.cli.create_tts_engine") as mock_tts_factory,
        ):
            mock_ask.return_value = "The answer"

            result = await _ask_once("What is this?", test_settings)

            assert result == "The answer"
            mock_ask.assert_awaited_once()
            call_args = mock_ask.call_args
            assert call_args[0][0] == "What is this?"
            assert call_args[0][1] is mock_llm_factory.return_value
            assert call_args[0][2] is mock_tts_factory.return_value
