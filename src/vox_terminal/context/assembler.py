"""Context assembler — orchestrates sources into a formatted markdown string."""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path

from vox_terminal.config import GeneralSettings, MCPSettings
from vox_terminal.context.sources.configs import get_config_context
from vox_terminal.context.sources.git import get_git_context
from vox_terminal.context.sources.tree import get_directory_tree

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Gathers project context from multiple sources and returns markdown."""

    def __init__(
        self,
        general: GeneralSettings | None = None,
        mcp: MCPSettings | None = None,
    ) -> None:
        self._general = general or GeneralSettings()
        self._mcp = mcp or MCPSettings()

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
        sections: list[str] = []

        # -- Project configs & README --
        config_ctx = get_config_context(root)
        if config_ctx:
            sections.append(f"## Project info\n\n{config_ctx}")

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
