"""Tests for inline file injection context source."""

from __future__ import annotations

from pathlib import Path

import pytest

from vox_terminal.context.sources.inline import (
    extract_file_references,
    inject_file_context,
    resolve_and_read_files,
)


class TestExtractFileReferences:
    """Tests for extract_file_references."""

    def test_extracts_simple_filename(self) -> None:
        assert extract_file_references("Look at main.py") == ["main.py"]

    def test_extracts_path_with_directory(self) -> None:
        assert extract_file_references("Check src/foo/bar.py") == ["src/foo/bar.py"]

    def test_extracts_multiple_refs(self) -> None:
        text = "Compare main.py and utils/helper.py"
        result = extract_file_references(text)
        assert "main.py" in result
        assert "utils/helper.py" in result

    def test_returns_empty_when_no_refs(self) -> None:
        assert extract_file_references("No file here") == []

    def test_extracts_ref_preceded_by_quote(self) -> None:
        assert extract_file_references('What about "config.json"?') == ["config.json"]

    def test_extracts_ref_at_start(self) -> None:
        assert extract_file_references("README.md has the docs") == ["README.md"]


class TestResolveAndReadFiles:
    """Tests for resolve_and_read_files."""

    def test_reads_existing_file_at_root(self, project_root: Path) -> None:
        (project_root / "main.py").write_text("x = 1\n")
        result = resolve_and_read_files(project_root, ["main.py"])
        assert "[File: main.py]" in result
        assert "x = 1" in result

    def test_reads_file_in_src(self, project_root: Path) -> None:
        result = resolve_and_read_files(project_root, ["main.py"])
        assert "[File: src/main.py]" in result
        assert "print('hello')" in result

    def test_returns_empty_when_no_files_found(self, project_root: Path) -> None:
        result = resolve_and_read_files(project_root, ["nonexistent.py"])
        assert result == ""

    def test_skips_binary_extensions(self, project_root: Path) -> None:
        (project_root / "image.png").write_bytes(b"\x89PNG\r\n")
        result = resolve_and_read_files(project_root, ["image.png"])
        assert result == ""

    def test_skips_file_over_size_limit(self, project_root: Path) -> None:
        large_content = "x" * 60_000
        (project_root / "huge.py").write_text(large_content)
        result = resolve_and_read_files(project_root, ["huge.py"])
        assert result == ""

    def test_respects_max_total_chars(self, project_root: Path) -> None:
        (project_root / "a.py").write_text("a" * 20_000)
        (project_root / "b.py").write_text("b" * 20_000)
        result = resolve_and_read_files(
            project_root, ["a.py", "b.py"], max_total_chars=25_000
        )
        assert "[File:" in result
        assert len(result) <= 25_000 + 100  # some overhead for formatting

    def test_deduplicates_same_file(self, project_root: Path) -> None:
        (project_root / "dup.py").write_text("content")
        result = resolve_and_read_files(
            project_root, ["dup.py", "dup.py", "dup.py"]
        )
        assert result.count("[File:") == 1

    def test_reads_readme(self, project_root: Path) -> None:
        result = resolve_and_read_files(project_root, ["README.md"])
        assert "[File: README.md]" in result
        assert "Test Project" in result


class TestInjectFileContext:
    """Tests for inject_file_context."""

    def test_returns_question_unchanged_when_no_refs(
        self, project_root: Path
    ) -> None:
        question = "What does this project do?"
        result = inject_file_context(question, project_root)
        assert result == question

    def test_prepends_file_content_when_refs_found(
        self, project_root: Path
    ) -> None:
        question = "Look at main.py and explain it"
        result = inject_file_context(question, project_root)
        assert "Here are the contents of the referenced file(s):" in result
        assert "User question: Look at main.py and explain it" in result
        assert "print('hello')" in result

    def test_returns_question_when_refs_not_resolvable(
        self, project_root: Path
    ) -> None:
        question = "Check nonexistent_file.xyz"
        result = inject_file_context(question, project_root)
        assert result == question
