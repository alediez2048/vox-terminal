"""Git context source — extracts repository metadata via subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vox_terminal.context.budget import ContextFragment
from vox_terminal.context.sources.base import ContextSource, ContextSourceConfig


def _run_git(root: Path, *args: str) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_branch(root: Path) -> str | None:
    """Return the current branch name, or None if not in a git repo."""
    return _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")


def get_git_status(root: Path) -> str:
    """Return short-form git status output."""
    return _run_git(root, "status", "--short") or ""


def get_git_recent_commits(root: Path, count: int = 5) -> str:
    """Return the last *count* commits as oneline summaries."""
    return _run_git(root, "log", "--oneline", f"-{count}") or ""


def get_git_remote(root: Path) -> str | None:
    """Return the URL of the 'origin' remote, or None."""
    return _run_git(root, "remote", "get-url", "origin")


def get_git_context(root: Path) -> str:
    """Combine all git metadata into a formatted context block."""
    parts: list[str] = []

    branch = get_git_branch(root)
    if branch:
        parts.append(f"**Branch:** {branch}")

    remote = get_git_remote(root)
    if remote:
        parts.append(f"**Remote:** {remote}")

    status = get_git_status(root)
    if status:
        parts.append(f"**Status:**\n```\n{status}\n```")

    commits = get_git_recent_commits(root)
    if commits:
        parts.append(f"**Recent commits:**\n```\n{commits}\n```")

    if not parts:
        return ""
    return "\n\n".join(parts)


class GitContextSource(ContextSource):
    """Context source for git branch, status, remote, and recent commits."""

    name = "git"
    priority = 60

    def __init__(self, config: ContextSourceConfig) -> None:
        super().__init__(config)
        self._include_git = config.include_git

    def gather(self, project_root: Path) -> ContextFragment | None:
        if not self._include_git:
            return None
        content = get_git_context(project_root)
        if not content:
            return None
        section = f"## Git\n\n{content}"
        return ContextFragment(
            name=self.name,
            content=section,
            priority=self.priority,
            token_estimate=max(1, len(section) // 4),
        )
