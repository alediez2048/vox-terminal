"""Inline file injection — detect file references in user questions and read them."""

from __future__ import annotations

import logging
import re
import time as _time
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches common file path patterns in natural speech/text
_FILE_PATTERN = re.compile(
    r"""(?:^|[\s"'`(])"""  # preceded by whitespace or quotes
    r"""((?:[\w./~-]+/)?"""  # optional directory prefix
    r"""[\w.-]+\.\w{1,10})"""  # filename.ext
    r"""(?:[\s"'`),.:;?!]|$)""",  # followed by whitespace, punctuation, or EOL
)

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".o",
        ".a",
        ".lib",
        ".obj",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".mkv",
        ".flac",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)

_MAX_FILE_SIZE = 50_000  # bytes
_MAX_INLINE_CHARS = 30_000  # total budget for inline files
_INDEX_SEARCH_ROOTS: tuple[str, ...] = ("src", "tests", "test", "app", "lib")
_INDEX_REFRESH_SECONDS = 120.0
_MAX_INDEXED_FILES = 20_000
_MAX_INDEX_MATCHES = 5

_FILENAME_INDEX_CACHE: dict[Path, tuple[float, dict[str, list[Path]]]] = {}


def extract_file_references(text: str) -> list[str]:
    """Extract potential file path references from a user question."""
    return _FILE_PATTERN.findall(text)


def _build_filename_index(project_root: Path) -> dict[str, list[Path]]:
    """Build a filename -> path index under constrained source roots."""
    index: dict[str, list[Path]] = {}
    indexed_files = 0

    def _add(path: Path) -> bool:
        nonlocal indexed_files
        if not path.is_file():
            return True
        index.setdefault(path.name, []).append(path)
        indexed_files += 1
        return indexed_files < _MAX_INDEXED_FILES

    try:
        for entry in project_root.iterdir():
            if entry.is_file() and not _add(entry.resolve()):
                return index
    except OSError:
        return index

    for root_name in _INDEX_SEARCH_ROOTS:
        root = (project_root / root_name).resolve()
        if not root.is_dir():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and not _add(path.resolve()):
                    return index
        except OSError:
            continue

    return index


def _get_filename_index(project_root: Path) -> dict[str, list[Path]]:
    """Return cached filename index, refreshing periodically."""
    root = project_root.resolve()
    now = _time.monotonic()
    cached = _FILENAME_INDEX_CACHE.get(root)
    if cached is not None:
        cached_at, index = cached
        if now - cached_at <= _INDEX_REFRESH_SECONDS:
            return index

    t0 = _time.monotonic()
    index = _build_filename_index(root)
    _FILENAME_INDEX_CACHE[root] = (now, index)
    logger.debug(
        "Inline filename index refreshed (root=%s, unique_names=%d, elapsed_ms=%.0f)",
        root,
        len(index),
        (_time.monotonic() - t0) * 1000,
    )
    return index


def resolve_and_read_files(
    project_root: Path,
    references: list[str],
    max_total_chars: int = _MAX_INLINE_CHARS,
) -> str:
    """Resolve file references relative to project root and read their contents.

    Returns a formatted string with file contents, or empty string if none found.
    """
    blocks: list[str] = []
    total = 0
    seen: set[Path] = set()
    filename_index = _get_filename_index(project_root)

    for ref in references:
        # Try as-is relative to project root
        candidates = [project_root / ref]

        # Also search common source directories
        for prefix in ("src", "lib", "app", "tests", "test"):
            candidates.append(project_root / prefix / ref)

        # Resolve bare filenames from a cached, constrained index.
        if "/" not in ref:
            matches = filename_index.get(ref, [])
            candidates.extend(matches[:_MAX_INDEX_MATCHES])

        for path in candidates:
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)

            if path.suffix.lower() in _BINARY_EXTENSIONS:
                continue

            try:
                size = path.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_SIZE:
                continue

            if total >= max_total_chars:
                break

            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            remaining = max_total_chars - total
            if len(content) > remaining:
                content = content[:remaining]

            try:
                rel = path.relative_to(project_root)
            except ValueError:
                rel = path

            blocks.append(f"[File: {rel}]\n```\n{content}\n```")
            total += len(content)
            break  # found a match for this reference, move on

    return "\n\n".join(blocks)


def inject_file_context(
    question: str,
    project_root: Path,
) -> str:
    """If the question references files, prepend their contents to the question.

    Returns the original question if no files are found.
    """
    refs = extract_file_references(question)
    if not refs:
        return question

    file_context = resolve_and_read_files(project_root, refs)
    if not file_context:
        return question

    logger.info("Injected %d file(s) into question context", len(refs))
    return (
        f"Here are the contents of the referenced file(s):\n\n"
        f"{file_context}\n\n"
        f"User question: {question}"
    )
