"""Project configuration detection source."""

from __future__ import annotations

from pathlib import Path

from vox_terminal.context.budget import ContextFragment
from vox_terminal.context.sources.base import ContextSource, ContextSourceConfig

_CONFIG_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "CMakeLists.txt",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Dockerfile",
    ".env",
    ".env.example",
    "requirements.txt",
    "Pipfile",
    "Gemfile",
    "pom.xml",
    "build.gradle",
)

# Files that should never have their contents read (secrets, credentials).
_SENSITIVE_CONFIG_FILES: frozenset[str] = frozenset({".env"})

_README_MAX_CHARS = 500

_README_CANDIDATES: tuple[str, ...] = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
)


def detect_project_configs(root: Path) -> list[str]:
    """Return names of recognised config files found directly under *root*."""
    return [name for name in _CONFIG_FILES if (root / name).is_file()]


def get_config_file_paths(root: Path) -> list[Path]:
    """Return paths of detected config files (excluding sensitive ones)."""
    return [
        root / name
        for name in _CONFIG_FILES
        if (root / name).is_file() and name not in _SENSITIVE_CONFIG_FILES
    ]


def get_readme_content(root: Path) -> str | None:
    """Read the full README file content, or return None."""
    for candidate in _README_CANDIDATES:
        readme = root / candidate
        if readme.is_file():
            try:
                return readme.read_text(encoding="utf-8")
            except OSError:
                return None
    return None


def get_readme_summary(root: Path) -> str | None:
    """Read the first 500 characters of a README file, or return None."""
    for candidate in _README_CANDIDATES:
        readme = root / candidate
        if readme.is_file():
            try:
                text = readme.read_text(encoding="utf-8")
            except OSError:
                return None
            return text[:_README_MAX_CHARS]
    return None


def get_config_context(root: Path, *, read_full_readme: bool = False) -> str:
    """Combine config detection and README summary into a context block.

    Parameters
    ----------
    root:
        Project root directory.
    read_full_readme:
        If True, include the full README content instead of a 500-char excerpt.
    """
    parts: list[str] = []

    configs = detect_project_configs(root)
    if configs:
        parts.append("**Config files:** " + ", ".join(f"`{c}`" for c in configs))

    if read_full_readme:
        readme = get_readme_content(root)
        if readme:
            parts.append(f"**README:**\n```\n{readme}\n```")
    else:
        readme = get_readme_summary(root)
        if readme:
            parts.append(f"**README (excerpt):**\n```\n{readme}\n```")

    return "\n\n".join(parts)


class ConfigsContextSource(ContextSource):
    """Context source for project-level config metadata and README content."""

    name = "project_info"
    priority = 100

    def __init__(self, config: ContextSourceConfig) -> None:
        super().__init__(config)
        self._read_full_readme = config.read_full_readme

    def gather(self, project_root: Path) -> ContextFragment | None:
        content = get_config_context(project_root, read_full_readme=self._read_full_readme)
        if not content:
            return None
        section = f"## Project info\n\n{content}"
        return ContextFragment(
            name=self.name,
            content=section,
            priority=self.priority,
            token_estimate=max(1, len(section) // 4),
        )
