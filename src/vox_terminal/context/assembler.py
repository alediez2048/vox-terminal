"""Context assembler — orchestrates sources into a formatted markdown string."""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path

from vox_terminal.config import ContextSettings, GeneralSettings, MCPSettings
from vox_terminal.context.sources.configs import (
    get_config_context,
    get_config_file_paths,
)
from vox_terminal.context.sources.files import get_file_contents, resolve_file_patterns
from vox_terminal.context.sources.git import get_git_context
from vox_terminal.context.sources.tree import get_directory_tree

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Gathers project context from multiple sources and returns markdown."""

    def __init__(
        self,
        general: GeneralSettings | None = None,
        mcp: MCPSettings | None = None,
        context_settings: ContextSettings | None = None,
    ) -> None:
        self._general = general or GeneralSettings()
        self._mcp = mcp or MCPSettings()
        self._context = context_settings or ContextSettings()

    @property
    def project_root(self) -> Path:
        return self._general.project_root

    @property
    def tree_depth(self) -> int:
        return self._mcp.tree_depth

    def assemble(
        self,
        include_git: bool = True,
        include_tree: bool = True,
    ) -> str:
        """Gather all context and return a formatted markdown string.

        Parameters
        ----------
        include_git:
            Whether to include git metadata.
        include_tree:
            Whether to include the directory tree.
        """
        t0 = _time.monotonic()
        root = self.project_root
        ctx = self._context
        sections: list[str] = []
        remaining = ctx.max_context_chars

        # -- Project configs & README --
        config_ctx = get_config_context(
            root, read_full_readme=ctx.read_full_readme,
        )
        if config_ctx:
            sections.append(f"## Project info\n\n{config_ctx}")
            remaining -= len(config_ctx)

        # -- Config file contents --
        if ctx.read_config_files and remaining > 0:
            config_paths = get_config_file_paths(root)
            if config_paths:
                content = get_file_contents(
                    root, config_paths,
                    max_file_size=ctx.max_file_size,
                    max_total_chars=remaining,
                )
                if content:
                    sections.append(f"## Config file contents\n\n{content}")
                    remaining -= len(content)

        # -- Documentation files --
        if ctx.doc_patterns and remaining > 0:
            doc_paths = resolve_file_patterns(root, ctx.doc_patterns)
            if doc_paths:
                content = get_file_contents(
                    root, doc_paths,
                    max_file_size=ctx.max_file_size,
                    max_total_chars=remaining,
                )
                if content:
                    sections.append(f"## Documentation\n\n{content}")
                    remaining -= len(content)

        # -- User-specified files --
        if ctx.include_files and remaining > 0:
            user_paths = resolve_file_patterns(root, ctx.include_files)
            if user_paths:
                content = get_file_contents(
                    root, user_paths,
                    max_file_size=ctx.max_file_size,
                    max_total_chars=remaining,
                )
                if content:
                    sections.append(f"## Project files\n\n{content}")
                    remaining -= len(content)

        # -- Git --
        if include_git:
            git_ctx = get_git_context(root)
            if git_ctx:
                sections.append(f"## Git\n\n{git_ctx}")

        # -- Directory tree --
        if include_tree:
            tree = get_directory_tree(root, max_depth=self.tree_depth)
            if tree:
                sections.append(f"## Directory tree\n\n```\n{tree}\n```")

        if not sections:
            logger.debug("No context assembled for %s", root)
            return ""
        logger.debug("Context assembled in %.0fms", (_time.monotonic() - t0) * 1000)
        return "\n\n".join(sections) + "\n"
