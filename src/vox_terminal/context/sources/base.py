"""Abstract interface for context source plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from vox_terminal.context.budget import ContextFragment


@dataclass(slots=True)
class ContextSourceConfig:
    """Configuration passed to context sources."""

    max_file_size: int = 50_000
    max_context_chars: int = 200_000
    read_config_files: bool = True
    read_full_readme: bool = True
    doc_patterns: list[str] = field(default_factory=list)
    include_files: list[str] = field(default_factory=list)
    tree_depth: int = 3
    include_git: bool = True
    include_tree: bool = True


class ContextSource(ABC):
    """Base class for all context source plugins."""

    name: ClassVar[str]
    priority: ClassVar[int]
    requires_network: ClassVar[bool] = False

    def __init__(self, config: ContextSourceConfig) -> None:
        self._config = config

    @abstractmethod
    def gather(self, project_root: Path) -> ContextFragment | None:
        """Gather source content for *project_root*."""
