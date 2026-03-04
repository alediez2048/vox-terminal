"""MCP server module — exposes Vox-Terminal tools via FastMCP with stdio transport."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from vox_terminal.config import load_settings
from vox_terminal.context.assembler import ContextAssembler
from vox_terminal.llm import create_llm_client


def create_mcp_server() -> FastMCP:
    """Create and return a configured FastMCP server with Vox-Terminal tools."""
    mcp = FastMCP("vox-terminal")
    settings = load_settings()

    @mcp.tool()
    async def ask_project_question(
        question: str,
        include_git: bool = True,
        include_tree: bool = True,
    ) -> str:
        """Ask a question about the current project with full context."""
        assembler = ContextAssembler(settings.general, settings.mcp)
        context = assembler.assemble(include_git=include_git, include_tree=include_tree)
        llm = create_llm_client(settings.llm, project_context=context)
        response = await llm.ask(question)
        return response.content

    @mcp.tool()
    async def get_project_summary() -> str:
        """Get a summary of the current project's structure and context."""
        assembler = ContextAssembler(settings.general, settings.mcp)
        return assembler.assemble()

    return mcp
