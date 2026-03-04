"""Context sources — individual data-gathering functions."""

from __future__ import annotations

from vox_terminal.context.sources.configs import (
    detect_project_configs,
    get_config_context,
    get_readme_summary,
)
from vox_terminal.context.sources.git import (
    get_git_branch,
    get_git_context,
    get_git_recent_commits,
    get_git_remote,
    get_git_status,
)
from vox_terminal.context.sources.tree import get_directory_tree

__all__ = [
    "detect_project_configs",
    "get_config_context",
    "get_directory_tree",
    "get_git_branch",
    "get_git_context",
    "get_git_recent_commits",
    "get_git_remote",
    "get_git_status",
    "get_readme_summary",
]
