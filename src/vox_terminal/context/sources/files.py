"""File content reading source for context assembly."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".bin", ".o", ".a", ".lib", ".obj",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".db", ".sqlite", ".sqlite3",
    ".wasm", ".class", ".jar",
})

_SENSITIVE_NAMES: frozenset[str] = frozenset({
    ".env", ".env.local", ".env.production", ".env.staging",
    ".env.development", ".env.test",
})


def _is_binary(path: Path) -> bool:
    """Check if a file is likely binary based on extension."""
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _is_sensitive(path: Path) -> bool:
    """Check if a file contains sensitive data."""
    return path.name in _SENSITIVE_NAMES


def resolve_file_patterns(root: Path, patterns: list[str]) -> list[Path]:
    """Expand glob patterns relative to *root* and return sorted unique paths."""
    seen: set[Path] = set()
    result: list[Path] = []
    for pattern in patterns:
        # Support both glob patterns and exact filenames
        if any(c in pattern for c in ("*", "?", "[")):
            matches = sorted(root.glob(pattern))
        else:
            candidate = root / pattern
            matches = [candidate] if candidate.is_file() else []
        for path in matches:
            if path.is_file() and path not in seen:
                seen.add(path)
                result.append(path)
    return sorted(result)


def get_file_contents(
    root: Path,
    paths: list[Path],
    max_file_size: int = 50_000,
    max_total_chars: int = 200_000,
) -> str:
    """Read files and return formatted code blocks, respecting size budgets.

    Parameters
    ----------
    root:
        Project root for computing relative paths.
    paths:
        Absolute paths to read.
    max_file_size:
        Skip individual files larger than this (bytes).
    max_total_chars:
        Stop reading once total characters exceed this budget.
    """
    blocks: list[str] = []
    total = 0

    for path in paths:
        if _is_binary(path):
            logger.debug("Skipping binary file: %s", path)
            continue
        if _is_sensitive(path):
            logger.debug("Skipping sensitive file: %s", path)
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue

        if size > max_file_size:
            logger.debug("Skipping large file (%d bytes): %s", size, path)
            continue

        if total >= max_total_chars:
            logger.debug("Context budget exhausted, stopping file reads")
            break

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Trim to stay within budget
        remaining = max_total_chars - total
        if len(text) > remaining:
            text = text[:remaining]

        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path

        blocks.append(f"**`{rel}`**\n```\n{text}\n```")
        total += len(text)

    return "\n\n".join(blocks)
