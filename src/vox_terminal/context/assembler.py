"""Context assembler — orchestrates sources into a formatted markdown string."""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path

from vox_terminal.config import ContextSettings, GeneralSettings, MCPSettings
from vox_terminal.context.budget import ContextBudget, ContextFragment
from vox_terminal.context.sources.base import ContextSource, ContextSourceConfig
from vox_terminal.context.sources.registry import get_source_classes

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Gathers project context from multiple sources and returns markdown."""

    def __init__(
        self,
        general: GeneralSettings | None = None,
        mcp: MCPSettings | None = None,
        context_settings: ContextSettings | None = None,
        sources: list[ContextSource] | None = None,
    ) -> None:
        self._general = general or GeneralSettings()
        self._mcp = mcp or MCPSettings()
        self._context = context_settings or ContextSettings()
        self._sources = sources

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
        budget = ContextBudget(ctx.max_context_chars)
        source_config = ContextSourceConfig(
            max_file_size=ctx.max_file_size,
            max_context_chars=ctx.max_context_chars,
            read_config_files=ctx.read_config_files,
            read_full_readme=ctx.read_full_readme,
            doc_patterns=list(ctx.doc_patterns),
            include_files=list(ctx.include_files),
            tree_depth=self.tree_depth,
            include_git=include_git,
            include_tree=include_tree,
        )
        fragments: list[ContextFragment] = []

        if self._sources is not None:
            sources = list(self._sources)
        else:
            enabled_sources = ctx.enabled_sources or None
            source_classes = get_source_classes(enabled_sources)
            sources = [source_class(source_config) for source_class in source_classes]

        for source in sources:
            if ctx.skip_network_sources and source.requires_network:
                logger.debug("Skipping network source in offline mode: %s", source.name)
                continue
            source_t0 = _time.monotonic()
            try:
                fragment = source.gather(root)
            except Exception:
                logger.exception("Context source failed: %s", source.name)
                continue
            elapsed_ms = (_time.monotonic() - source_t0) * 1000
            if fragment and fragment.content:
                logger.info(
                    "Context source gathered (source=%s, chars=%d, elapsed_ms=%.0f)",
                    source.name,
                    len(fragment.content),
                    elapsed_ms,
                )
                fragments.append(fragment)
            else:
                logger.info(
                    "Context source empty (source=%s, elapsed_ms=%.0f)",
                    source.name,
                    elapsed_ms,
                )

        # Allocate budget greedily from highest to lowest priority.
        sections: list[str] = []
        for fragment in sorted(fragments, key=lambda f: f.priority, reverse=True):
            allocated = budget.allocate(fragment)
            if allocated:
                sections.append(allocated)
            logger.info(
                "Context source budget allocation (source=%s, priority=%d, input_chars=%d, output_chars=%d)",
                fragment.name,
                fragment.priority,
                len(fragment.content),
                len(allocated),
            )

        if not sections:
            logger.debug("No context assembled for %s", root)
            return ""
        elapsed_ms = (_time.monotonic() - t0) * 1000
        result = "\n\n".join(sections) + "\n"
        logger.info(
            "Context assembled (chars=%d, sections=%d, remaining_chars=%d, elapsed_ms=%.0f)",
            len(result),
            len(sections),
            budget.remaining_chars,
            elapsed_ms,
        )
        logger.debug("Context sections included: %s", ", ".join(s.name for s in fragments))
        return result
