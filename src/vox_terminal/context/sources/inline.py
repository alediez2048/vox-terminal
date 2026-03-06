"""Inline file injection — detect file references in user questions and read them."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches common file path patterns in natural speech/text
_FILE_PATTERN = re.compile(
    r"""(?:^|[\s"'`(])"""  # preceded by whitespace or quotes
    r"""((?:[\w./~-]+/)?"""  # optional directory prefix
    r"""[\w.-]+\.\w{1,10})"""  # filename.ext
    r"""(?:[\s"'`),.:;?!]|$)""",  # followed by whitespace, punctuation, or EOL
)

_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".bin", ".o", ".a", ".lib", ".obj",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".db", ".sqlite", ".sqlite3",
})

_MAX_FILE_SIZE = 50_000  # bytes
_MAX_INLINE_CHARS = 30_000  # total budget for inline files


def extract_file_references(text: str) -> list[str]:
    """Extract potential file path references from a user question."""
    return _FILE_PATTERN.findall(text)


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

    for ref in references:
        # Try as-is relative to project root
        candidates = [project_root / ref]

        # Also search common source directories
        for prefix in ("src", "lib", "app", "tests", "test"):
            candidates.append(project_root / prefix / ref)

        # Try recursive glob if it's just a filename
        if "/" not in ref:
            matches = list(project_root.rglob(ref))
            candidates.extend(matches[:3])  # limit to avoid huge scans

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
