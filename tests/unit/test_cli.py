"""Tests for the CLI module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from vox_terminal.cli import app

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
