"""Tests for project root resolution logic."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from vox_terminal.project_root import (
    _find_ancestor_with_file,
    _find_git_root,
    resolve_project_root,
)


class TestFindAncestorWithFile:
    def test_finds_file_in_start_dir(self, tmp_path: Path) -> None:
        (tmp_path / "vox-terminal.toml").write_text("")
        assert _find_ancestor_with_file(tmp_path, "vox-terminal.toml") == tmp_path

    def test_finds_file_in_parent(self, tmp_path: Path) -> None:
        (tmp_path / "vox-terminal.toml").write_text("")
        nested = tmp_path / "src" / "deep"
        nested.mkdir(parents=True)
        assert _find_ancestor_with_file(nested, "vox-terminal.toml") == tmp_path

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert _find_ancestor_with_file(tmp_path, "vox-terminal.toml") is None


class TestFindGitRoot:
    def test_returns_git_root_when_present(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        with patch(
            "vox_terminal.project_root.subprocess.run"
        ) as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = str(tmp_path)
            result = _find_git_root(tmp_path)
        assert result == tmp_path

    def test_returns_none_when_not_git(self, tmp_path: Path) -> None:
        with patch(
            "vox_terminal.project_root.subprocess.run"
        ) as mock_run:
            mock_run.return_value.returncode = 128
            mock_run.return_value.stdout = ""
            result = _find_git_root(tmp_path)
        assert result is None

    def test_returns_none_on_oserror(self, tmp_path: Path) -> None:
        with patch(
            "vox_terminal.project_root.subprocess.run",
            side_effect=OSError("git not found"),
        ):
            assert _find_git_root(tmp_path) is None


class TestResolveProjectRoot:
    def test_explicit_path_takes_precedence(self, tmp_path: Path) -> None:
        result = resolve_project_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_toml_ancestor_preferred_over_git(self, tmp_path: Path) -> None:
        (tmp_path / "vox-terminal.toml").write_text("")
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        with patch("vox_terminal.project_root.Path.cwd", return_value=nested):
            result = resolve_project_root(None)
        assert result == tmp_path

    def test_git_root_used_when_no_toml(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub"
        nested.mkdir()
        with (
            patch("vox_terminal.project_root.Path.cwd", return_value=nested),
            patch(
                "vox_terminal.project_root._find_git_root",
                return_value=tmp_path,
            ),
        ):
            result = resolve_project_root(None)
        assert result == tmp_path

    def test_falls_back_to_cwd(self, tmp_path: Path) -> None:
        with (
            patch("vox_terminal.project_root.Path.cwd", return_value=tmp_path),
            patch(
                "vox_terminal.project_root._find_git_root",
                return_value=None,
            ),
        ):
            result = resolve_project_root(None)
        assert result == tmp_path

    def test_explicit_path_resolves_relative(self, tmp_path: Path) -> None:
        sub = tmp_path / "myrepo"
        sub.mkdir()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = resolve_project_root(Path("myrepo"))
        finally:
            os.chdir(old_cwd)
        assert result == sub

    def test_nested_subdir_walks_up_to_toml(self, tmp_path: Path) -> None:
        """Running from project/src/app/components still finds the root."""
        (tmp_path / "vox-terminal.toml").write_text("")
        deep = tmp_path / "src" / "app" / "components"
        deep.mkdir(parents=True)
        with patch("vox_terminal.project_root.Path.cwd", return_value=deep):
            result = resolve_project_root(None)
        assert result == tmp_path
