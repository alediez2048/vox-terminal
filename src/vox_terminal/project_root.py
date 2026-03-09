"""Project root resolution — walk up from a starting path to find the canonical project root."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_TOML_CONFIG = "vox-terminal.toml"


def _find_ancestor_with_file(start: Path, filename: str) -> Path | None:
    """Walk up from *start* looking for a directory containing *filename*."""
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / filename).is_file():
            return parent
    return None


def _find_git_root(start: Path) -> Path | None:
    """Use git to find the repository root above *start*."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=str(start),
            timeout=5,
        )
        if result.returncode == 0:
            root = Path(result.stdout.strip())
            if root.is_dir():
                return root
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def resolve_project_root(explicit: Path | None = None) -> Path:
    """Determine the canonical project root using the following precedence:

    1. Explicit path (CLI argument / ``--project-root``)
    2. Nearest ancestor containing ``vox-terminal.toml``
    3. Nearest ancestor git root
    4. Current working directory
    """
    if explicit is not None:
        resolved = explicit.resolve()
        if not resolved.is_dir():
            logger.warning("Explicit project root is not a directory: %s", resolved)
        logger.info("Project root from explicit path: %s", resolved)
        return resolved

    cwd = Path.cwd()

    toml_root = _find_ancestor_with_file(cwd, _TOML_CONFIG)
    if toml_root is not None:
        logger.info("Project root from %s: %s", _TOML_CONFIG, toml_root)
        return toml_root

    git_root = _find_git_root(cwd)
    if git_root is not None:
        logger.info("Project root from git: %s", git_root)
        return git_root

    logger.info("Project root from cwd: %s", cwd)
    return cwd
