"""Context sources — individual data-gathering functions."""

from __future__ import annotations

from vox_terminal.context.sources.configs import (
    detect_project_configs,
    get_config_context,
    get_config_file_paths,
    get_readme_content,
    get_readme_summary,
)
from vox_terminal.context.sources.files import (
    get_file_contents,
    resolve_file_patterns,
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
    "get_config_file_paths",
    "get_directory_tree",
    "get_file_contents",
    "get_git_branch",
    "get_git_context",
    "get_git_recent_commits",
    "get_git_remote",
    "get_git_status",
    "get_readme_content",
    "get_readme_summary",
    "resolve_file_patterns",
]
