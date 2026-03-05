"""Tests for context sources and assembler."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vox_terminal.config import ContextSettings, GeneralSettings, MCPSettings
from vox_terminal.context import ContextAssembler
from vox_terminal.context.sources.configs import (
    detect_project_configs,
    get_config_context,
    get_config_file_paths,
    get_readme_content,
    get_readme_summary,
)
from vox_terminal.context.sources.files import get_file_contents, resolve_file_patterns
from vox_terminal.context.sources.git import (
    get_git_branch,
    get_git_context,
    get_git_recent_commits,
    get_git_remote,
    get_git_status,
)
from vox_terminal.context.sources.tree import get_directory_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(root: Path) -> None:
    """Initialise a throwaway git repo with one commit."""
    subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=root, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root, capture_output=True, check=True,
    )
    (root / "hello.txt").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=root, capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# Git source tests
# ---------------------------------------------------------------------------

class TestGitSource:
    def test_branch_in_repo(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        branch = get_git_branch(tmp_path)
        assert branch is not None
        # Could be "main" or "master" depending on git config
        assert branch in ("main", "master")

    def test_branch_outside_repo(self, tmp_path: Path) -> None:
        assert get_git_branch(tmp_path) is None

    def test_status_clean(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        status = get_git_status(tmp_path)
        assert status == ""

    def test_status_dirty(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        (tmp_path / "new.txt").write_text("new\n")
        status = get_git_status(tmp_path)
        assert "new.txt" in status

    def test_recent_commits(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        commits = get_git_recent_commits(tmp_path)
        assert "Initial commit" in commits

    def test_recent_commits_outside_repo(self, tmp_path: Path) -> None:
        assert get_git_recent_commits(tmp_path) == ""

    def test_remote_none(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        assert get_git_remote(tmp_path) is None

    def test_get_git_context_returns_string(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        ctx = get_git_context(tmp_path)
        assert "Branch" in ctx
        assert "Initial commit" in ctx

    def test_get_git_context_empty_outside_repo(self, tmp_path: Path) -> None:
        ctx = get_git_context(tmp_path)
        assert ctx == ""


# ---------------------------------------------------------------------------
# Tree source tests
# ---------------------------------------------------------------------------

class TestTreeSource:
    def test_basic_tree(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        tree = get_directory_tree(tmp_path)
        assert "a.txt" in tree
        assert "sub/" in tree
        assert "b.txt" in tree

    def test_ignores_default_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.pyc").write_text("")
        (tmp_path / "real.py").write_text("")
        tree = get_directory_tree(tmp_path)
        assert "__pycache__" not in tree
        assert "real.py" in tree

    def test_max_depth(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep.txt").write_text("deep")
        tree = get_directory_tree(tmp_path, max_depth=2)
        assert "a/" in tree
        assert "b/" in tree
        # depth 3 content should NOT appear at max_depth=2
        assert "deep.txt" not in tree

    def test_custom_ignore(self, tmp_path: Path) -> None:
        (tmp_path / "vendor").mkdir()
        (tmp_path / "vendor" / "lib.py").write_text("")
        tree = get_directory_tree(tmp_path, ignore_dirs={"vendor"})
        assert "vendor" not in tree

    def test_empty_dir(self, tmp_path: Path) -> None:
        tree = get_directory_tree(tmp_path)
        # Should at least have the root name
        assert tmp_path.name in tree


# ---------------------------------------------------------------------------
# Config source tests
# ---------------------------------------------------------------------------

class TestConfigSource:
    def test_detect_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        found = detect_project_configs(tmp_path)
        assert "pyproject.toml" in found

    def test_detect_multiple(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "Dockerfile").write_text("")
        found = detect_project_configs(tmp_path)
        assert "pyproject.toml" in found
        assert "Dockerfile" in found

    def test_detect_none(self, tmp_path: Path) -> None:
        assert detect_project_configs(tmp_path) == []

    def test_readme_summary(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Hello World\nThis is a project.\n")
        summary = get_readme_summary(tmp_path)
        assert summary is not None
        assert summary.startswith("# Hello World")

    def test_readme_truncation(self, tmp_path: Path) -> None:
        long_text = "x" * 1000
        (tmp_path / "README.md").write_text(long_text)
        summary = get_readme_summary(tmp_path)
        assert summary is not None
        assert len(summary) == 500

    def test_no_readme(self, tmp_path: Path) -> None:
        assert get_readme_summary(tmp_path) is None

    def test_get_config_context(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "README.md").write_text("# My Project\n")
        ctx = get_config_context(tmp_path)
        assert "`pyproject.toml`" in ctx
        assert "README" in ctx

    def test_get_config_context_full_readme(self, tmp_path: Path) -> None:
        long_text = "# Title\n" + "x" * 1000
        (tmp_path / "README.md").write_text(long_text)
        ctx = get_config_context(tmp_path, read_full_readme=True)
        assert "**README:**" in ctx
        assert "x" * 1000 in ctx

    def test_get_config_context_excerpt_readme(self, tmp_path: Path) -> None:
        long_text = "x" * 1000
        (tmp_path / "README.md").write_text(long_text)
        ctx = get_config_context(tmp_path, read_full_readme=False)
        assert "(excerpt)" in ctx

    def test_readme_content_full(self, tmp_path: Path) -> None:
        full = "# Hello\n" + "content " * 200
        (tmp_path / "README.md").write_text(full)
        result = get_readme_content(tmp_path)
        assert result == full

    def test_readme_content_none(self, tmp_path: Path) -> None:
        assert get_readme_content(tmp_path) is None

    def test_config_file_paths_excludes_env(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / ".env").write_text("SECRET=123\n")
        paths = get_config_file_paths(tmp_path)
        names = [p.name for p in paths]
        assert "pyproject.toml" in names
        assert ".env" not in names


# ---------------------------------------------------------------------------
# File source tests
# ---------------------------------------------------------------------------

class TestFileSource:
    def test_resolve_glob_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')\n")
        (tmp_path / "src" / "util.py").write_text("pass\n")
        (tmp_path / "readme.txt").write_text("hello\n")
        paths = resolve_file_patterns(tmp_path, ["src/*.py"])
        names = [p.name for p in paths]
        assert "main.py" in names
        assert "util.py" in names
        assert "readme.txt" not in names

    def test_resolve_exact_filename(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[x]\n")
        paths = resolve_file_patterns(tmp_path, ["config.toml"])
        assert len(paths) == 1
        assert paths[0].name == "config.toml"

    def test_resolve_missing_pattern(self, tmp_path: Path) -> None:
        paths = resolve_file_patterns(tmp_path, ["nonexistent.py"])
        assert paths == []

    def test_resolve_deduplicates(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("a\n")
        paths = resolve_file_patterns(tmp_path, ["a.py", "*.py"])
        assert len(paths) == 1

    def test_get_file_contents_basic(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("print('hello')\n")
        result = get_file_contents(tmp_path, [tmp_path / "hello.py"])
        assert "**`hello.py`**" in result
        assert "print('hello')" in result

    def test_get_file_contents_skips_binary(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        result = get_file_contents(tmp_path, [tmp_path / "image.png"])
        assert result == ""

    def test_get_file_contents_skips_sensitive(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=abc\n")
        result = get_file_contents(tmp_path, [tmp_path / ".env"])
        assert result == ""

    def test_get_file_contents_skips_large(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("x" * 100_000)
        result = get_file_contents(
            tmp_path, [tmp_path / "big.txt"], max_file_size=1000,
        )
        assert result == ""

    def test_get_file_contents_respects_total_budget(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a" * 100)
        (tmp_path / "b.txt").write_text("b" * 100)
        result = get_file_contents(
            tmp_path,
            [tmp_path / "a.txt", tmp_path / "b.txt"],
            max_total_chars=150,
        )
        assert "a" * 100 in result
        # b.txt should be trimmed or partial
        assert "b" * 100 not in result

    def test_get_file_contents_multiple(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("x_content\n")
        (tmp_path / "y.py").write_text("y_content\n")
        result = get_file_contents(
            tmp_path, [tmp_path / "x.py", tmp_path / "y.py"],
        )
        assert "x_content" in result
        assert "y_content" in result

    def test_get_file_contents_missing_file(self, tmp_path: Path) -> None:
        result = get_file_contents(tmp_path, [tmp_path / "gone.txt"])
        assert result == ""


# ---------------------------------------------------------------------------
# Assembler tests
# ---------------------------------------------------------------------------

class TestContextAssembler:
    def test_assemble_with_configs(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "README.md").write_text("# Test\n")
        general = GeneralSettings(project_root=tmp_path)
        assembler = ContextAssembler(general=general)
        result = assembler.assemble(include_git=False, include_tree=False)
        assert "## Project info" in result
        assert "`pyproject.toml`" in result

    def test_assemble_with_tree(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("")
        general = GeneralSettings(project_root=tmp_path)
        mcp = MCPSettings(tree_depth=2)
        assembler = ContextAssembler(general=general, mcp=mcp)
        result = assembler.assemble(include_git=False, include_tree=True)
        assert "## Directory tree" in result
        assert "app.py" in result

    def test_assemble_with_git(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        general = GeneralSettings(project_root=tmp_path)
        assembler = ContextAssembler(general=general)
        result = assembler.assemble(include_git=True, include_tree=False)
        assert "## Git" in result
        assert "Initial commit" in result

    def test_assemble_empty_project(self, tmp_path: Path) -> None:
        general = GeneralSettings(project_root=tmp_path)
        assembler = ContextAssembler(general=general)
        result = assembler.assemble(include_git=False, include_tree=False)
        # No configs, no readme -> empty
        assert result == ""

    def test_assemble_full(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "README.md").write_text("# Full test\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')\n")

        general = GeneralSettings(project_root=tmp_path)
        mcp = MCPSettings(tree_depth=2)
        assembler = ContextAssembler(general=general, mcp=mcp)
        result = assembler.assemble()

        assert "## Project info" in result
        assert "## Git" in result
        assert "## Directory tree" in result
        assert result.endswith("\n")

    def test_default_settings(self) -> None:
        assembler = ContextAssembler()
        assert assembler.tree_depth == 3
        assert assembler.project_root == Path.cwd()

    def test_assemble_reads_config_contents(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        general = GeneralSettings(project_root=tmp_path)
        ctx_settings = ContextSettings(
            read_config_files=True,
            read_full_readme=False,
            doc_patterns=[],
        )
        assembler = ContextAssembler(
            general=general, context_settings=ctx_settings,
        )
        result = assembler.assemble(include_git=False, include_tree=False)
        assert "## Config file contents" in result
        assert "name = 'test'" in result

    def test_assemble_reads_doc_files(self, tmp_path: Path) -> None:
        (tmp_path / "CHANGELOG.md").write_text("## v1.0\n- Initial release\n")
        general = GeneralSettings(project_root=tmp_path)
        ctx_settings = ContextSettings(
            read_config_files=False,
            doc_patterns=["CHANGELOG.md"],
        )
        assembler = ContextAssembler(
            general=general, context_settings=ctx_settings,
        )
        result = assembler.assemble(include_git=False, include_tree=False)
        assert "## Documentation" in result
        assert "Initial release" in result

    def test_assemble_reads_user_files(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def main(): pass\n")
        general = GeneralSettings(project_root=tmp_path)
        ctx_settings = ContextSettings(
            read_config_files=False,
            doc_patterns=[],
            include_files=["src/*.py"],
        )
        assembler = ContextAssembler(
            general=general, context_settings=ctx_settings,
        )
        result = assembler.assemble(include_git=False, include_tree=False)
        assert "## Project files" in result
        assert "def main(): pass" in result

    def test_assemble_backward_compat(self, tmp_path: Path) -> None:
        """ContextAssembler works without context_settings (backward compat)."""
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "README.md").write_text("# Test\n")
        general = GeneralSettings(project_root=tmp_path)
        assembler = ContextAssembler(general=general)
        result = assembler.assemble(include_git=False, include_tree=False)
        assert "## Project info" in result

    def test_assemble_respects_budget(self, tmp_path: Path) -> None:
        (tmp_path / "big.py").write_text("x" * 500)
        general = GeneralSettings(project_root=tmp_path)
        ctx_settings = ContextSettings(
            read_config_files=False,
            doc_patterns=[],
            include_files=["big.py"],
            max_context_chars=100,
        )
        assembler = ContextAssembler(
            general=general, context_settings=ctx_settings,
        )
        result = assembler.assemble(include_git=False, include_tree=False)
        # File content should be truncated to budget
        if "## Project files" in result:
            assert "x" * 500 not in result
