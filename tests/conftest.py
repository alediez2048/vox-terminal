"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from vox_terminal.config import VoxTerminalSettings


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
    (tmp_path / "README.md").write_text("# Test Project\nA test project.\n")
    return tmp_path


@pytest.fixture
def settings(project_root: Path) -> VoxTerminalSettings:
    """Create test settings."""
    return VoxTerminalSettings(
        general={"project_root": project_root},  # type: ignore[arg-type]
        llm={"api_key": "test-key"},  # type: ignore[arg-type]
    )


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Remove VOX_TERMINAL_ env vars for clean tests."""
    old = {k: v for k, v in os.environ.items() if k.startswith("VOX_TERMINAL_")}
    for k in old:
        del os.environ[k]
    yield
    os.environ.update(old)
