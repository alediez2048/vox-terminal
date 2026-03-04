"""Project configuration detection source."""

from __future__ import annotations

from pathlib import Path

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

_README_MAX_CHARS = 500


def detect_project_configs(root: Path) -> list[str]:
    """Return names of recognised config files found directly under *root*."""
    return [name for name in _CONFIG_FILES if (root / name).is_file()]


def get_readme_summary(root: Path) -> str | None:
    """Read the first 500 characters of a README file, or return None."""
    for candidate in ("README.md", "README.rst", "README.txt", "README"):
        readme = root / candidate
        if readme.is_file():
            try:
                text = readme.read_text(encoding="utf-8")
            except OSError:
                return None
            return text[:_README_MAX_CHARS]
    return None


def get_config_context(root: Path) -> str:
    """Combine config detection and README summary into a context block."""
    parts: list[str] = []

    configs = detect_project_configs(root)
    if configs:
        parts.append("**Config files:** " + ", ".join(f"`{c}`" for c in configs))

    readme = get_readme_summary(root)
    if readme:
        parts.append(f"**README (excerpt):**\n```\n{readme}\n```")

    return "\n\n".join(parts)
