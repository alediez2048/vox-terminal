"""Registry for context source plugins."""

from __future__ import annotations

from collections.abc import Iterable

from vox_terminal.context.sources.base import ContextSource

_REGISTRY: dict[str, type[ContextSource]] = {}
_BUILTINS_REGISTERED = False


def register_source(name: str, source_class: type[ContextSource]) -> None:
    """Register a context source class under *name*."""
    _REGISTRY[name] = source_class


def get_source_classes(enabled_sources: Iterable[str] | None = None) -> list[type[ContextSource]]:
    """Return registered source classes, optionally filtered by name."""
    _ensure_builtins_registered()
    if enabled_sources:
        return [source_class for name, source_class in _REGISTRY.items() if name in enabled_sources]
    return list(_REGISTRY.values())


def get_source_names() -> list[str]:
    """Return all known source names."""
    _ensure_builtins_registered()
    return sorted(_REGISTRY)


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from vox_terminal.context.sources.configs import ConfigsContextSource
    from vox_terminal.context.sources.files import (
        ConfigFileContentsContextSource,
        DocumentationContextSource,
        ProjectFilesContextSource,
    )
    from vox_terminal.context.sources.git import GitContextSource
    from vox_terminal.context.sources.tree import TreeContextSource

    builtins: list[type[ContextSource]] = [
        ConfigsContextSource,
        ConfigFileContentsContextSource,
        DocumentationContextSource,
        ProjectFilesContextSource,
        GitContextSource,
        TreeContextSource,
    ]
    for source_class in builtins:
        register_source(source_class.name, source_class)

    _BUILTINS_REGISTERED = True
