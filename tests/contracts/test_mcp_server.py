"""Contract tests for the MCP server module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from mcp.server.fastmcp import FastMCP

from vox_terminal.llm.base import LLMResponse
from vox_terminal.mcp_server import create_mcp_server

# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------


class TestCreateMCPServer:
    """Tests for the create_mcp_server factory."""

    def test_returns_fastmcp_instance(self) -> None:
        server = create_mcp_server()
        assert isinstance(server, FastMCP)

    def test_server_has_expected_tools(self) -> None:
        server = create_mcp_server()
        tools = server._tool_manager.list_tools()
        tool_names = {t.name for t in tools}
        assert "ask_project_question" in tool_names
        assert "get_project_summary" in tool_names

    def test_server_has_exactly_two_tools(self) -> None:
        server = create_mcp_server()
        tools = server._tool_manager.list_tools()
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# Tool: ask_project_question
# ---------------------------------------------------------------------------


class TestAskProjectQuestion:
    """Tests for the ask_project_question tool."""

    async def test_returns_llm_response_content(self) -> None:
        server = create_mcp_server()

        mock_response = LLMResponse(
            content="This is a Python project.",
            model="claude-sonnet-4-20250514",
        )
        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(return_value=mock_response)

        with (
            patch("vox_terminal.mcp_server.ContextAssembler") as mock_assembler_cls,
            patch(
                "vox_terminal.mcp_server.create_llm_client",
                return_value=mock_llm,
            ) as mock_create_llm,
        ):
            mock_assembler = MagicMock()
            mock_assembler.assemble.return_value = "## Project info\n\nsome context\n"
            mock_assembler_cls.return_value = mock_assembler

            result = await server._tool_manager.call_tool(
                "ask_project_question",
                {"question": "What does this project do?"},
            )

            assert result == "This is a Python project."
            mock_assembler.assemble.assert_called_once_with(include_git=True, include_tree=True)
            mock_create_llm.assert_called_once()
            mock_llm.ask.assert_awaited_once_with("What does this project do?")

    async def test_passes_include_flags(self) -> None:
        server = create_mcp_server()

        mock_response = LLMResponse(content="answer", model="test")
        mock_llm = MagicMock()
        mock_llm.ask = AsyncMock(return_value=mock_response)

        with (
            patch("vox_terminal.mcp_server.ContextAssembler") as mock_assembler_cls,
            patch(
                "vox_terminal.mcp_server.create_llm_client",
                return_value=mock_llm,
            ),
        ):
            mock_assembler = MagicMock()
            mock_assembler.assemble.return_value = ""
            mock_assembler_cls.return_value = mock_assembler

            await server._tool_manager.call_tool(
                "ask_project_question",
                {
                    "question": "hi",
                    "include_git": False,
                    "include_tree": False,
                },
            )

            mock_assembler.assemble.assert_called_once_with(include_git=False, include_tree=False)


# ---------------------------------------------------------------------------
# Tool: get_project_summary
# ---------------------------------------------------------------------------


class TestGetProjectSummary:
    """Tests for the get_project_summary tool."""

    async def test_returns_context_string(self) -> None:
        server = create_mcp_server()

        expected_context = "## Project info\n\nA cool project\n"

        with patch("vox_terminal.mcp_server.ContextAssembler") as mock_assembler_cls:
            mock_assembler = MagicMock()
            mock_assembler.assemble.return_value = expected_context
            mock_assembler_cls.return_value = mock_assembler

            result = await server._tool_manager.call_tool(
                "get_project_summary",
                {},
            )

            assert result == expected_context
            mock_assembler.assemble.assert_called_once()

    async def test_returns_empty_when_no_context(self) -> None:
        server = create_mcp_server()

        with patch("vox_terminal.mcp_server.ContextAssembler") as mock_assembler_cls:
            mock_assembler = MagicMock()
            mock_assembler.assemble.return_value = ""
            mock_assembler_cls.return_value = mock_assembler

            result = await server._tool_manager.call_tool(
                "get_project_summary",
                {},
            )

            assert result == ""
