"""Tests for context source classes and source registry."""

from __future__ import annotations

from pathlib import Path

from vox_terminal.config import ContextSettings, GeneralSettings
from vox_terminal.context import ContextAssembler
from vox_terminal.context.budget import ContextFragment
from vox_terminal.context.sources.base import ContextSource, ContextSourceConfig
from vox_terminal.context.sources.configs import ConfigsContextSource
from vox_terminal.context.sources.git import GitContextSource
from vox_terminal.context.sources.registry import get_source_classes, get_source_names


class TestSourceRegistry:
    def test_get_source_names_includes_builtins(self) -> None:
        names = get_source_names()
        assert "project_info" in names
        assert "config_file_contents" in names
        assert "documentation" in names
        assert "project_files" in names
        assert "git" in names
        assert "tree" in names

    def test_get_source_classes_with_filter(self) -> None:
        classes = get_source_classes(["project_info", "tree"])
        names = {source_class.name for source_class in classes}
        assert names == {"project_info", "tree"}


class TestSourceClasses:
    def test_configs_source_gathers_project_info(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        config = ContextSourceConfig(read_full_readme=False)
        source = ConfigsContextSource(config)

        fragment = source.gather(tmp_path)

        assert fragment is not None
        assert "## Project info" in fragment.content
        assert "`pyproject.toml`" in fragment.content

    def test_git_source_respects_include_flag(self, tmp_path: Path) -> None:
        config = ContextSourceConfig(include_git=False)
        source = GitContextSource(config)
        assert source.gather(tmp_path) is None


class TestAssemblerWithSources:
    def test_assembler_uses_custom_source_instances(self, tmp_path: Path) -> None:
        class StaticSource(ContextSource):
            name = "static"
            priority = 100

            def gather(self, project_root: Path) -> ContextFragment | None:
                return ContextFragment(name=self.name, content="## Static\n\nvalue", priority=100)

        source = StaticSource(ContextSourceConfig())
        general = GeneralSettings(project_root=tmp_path)
        assembler = ContextAssembler(general=general, sources=[source])

        result = assembler.assemble(include_git=False, include_tree=False)

        assert "## Static" in result
        assert "value" in result

    def test_assembler_respects_enabled_sources(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "README.md").write_text("# Hello\n")
        (tmp_path / "notes.md").write_text("doc\n")

        general = GeneralSettings(project_root=tmp_path)
        context = ContextSettings(
            enabled_sources=["project_info"],
            doc_patterns=["notes.md"],
            include_files=[],
            read_config_files=False,
        )
        assembler = ContextAssembler(general=general, context_settings=context)

        result = assembler.assemble(include_git=False, include_tree=False)

        assert "## Project info" in result
        assert "## Documentation" not in result

    def test_assembler_skips_network_sources(self, tmp_path: Path) -> None:
        class NetworkOnlySource(ContextSource):
            name = "network"
            priority = 100
            requires_network = True

            def gather(self, project_root: Path) -> ContextFragment | None:
                return ContextFragment(name=self.name, content="## Network\n\nx", priority=100)

        general = GeneralSettings(project_root=tmp_path)
        context = ContextSettings(skip_network_sources=True)
        source = NetworkOnlySource(ContextSourceConfig())
        assembler = ContextAssembler(
            general=general,
            context_settings=context,
            sources=[source],
        )

        result = assembler.assemble(include_git=False, include_tree=False)

        assert result == ""
