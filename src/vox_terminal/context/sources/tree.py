"""Directory tree source — generates a depth-limited tree view."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "dist",
    "build",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
})


def get_directory_tree(
    root: Path,
    max_depth: int = 3,
    ignore_dirs: set[str] | None = None,
) -> str:
    """Return a depth-limited directory tree as a string.

    Parameters
    ----------
    root:
        The root directory to walk.
    max_depth:
        Maximum depth to recurse (1 = only root's immediate children).
    ignore_dirs:
        Directory names to skip.  Falls back to ``_DEFAULT_IGNORE_DIRS``.
    """
    if ignore_dirs is None:
        ignore_dirs = set(_DEFAULT_IGNORE_DIRS)

    lines: list[str] = [f"{root.name}/"]
    _walk(root, "", max_depth, 1, ignore_dirs, lines)
    return "\n".join(lines)


def _walk(
    directory: Path,
    prefix: str,
    max_depth: int,
    current_depth: int,
    ignore_dirs: set[str],
    lines: list[str],
) -> None:
    """Recursively build tree lines."""
    if current_depth > max_depth:
        return

    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return

    # Filter out ignored directories
    entries = [e for e in entries if not (e.is_dir() and e.name in ignore_dirs)]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            extension = "    " if is_last else "│   "
            _walk(entry, prefix + extension, max_depth, current_depth + 1, ignore_dirs, lines)
        else:
            lines.append(f"{prefix}{connector}{entry.name}")
