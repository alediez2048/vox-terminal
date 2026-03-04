"""Tests for context sources and assembler."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vox_terminal.config import GeneralSettings, MCPSettings
from vox_terminal.context import ContextAssembler
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
