"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

from vox_terminal.config import (
    ContextSettings,
    GeneralSettings,
    LLMSettings,
    MCPSettings,
    STTSettings,
    TTSSettings,
    VoxTerminalSettings,
    load_settings,
)


class TestGeneralSettings:
    def test_defaults(self) -> None:
        s = GeneralSettings()
        assert s.verbose is False
        assert s.log_level == "INFO"
        assert s.log_sensitive is False
        assert s.log_file is None
        assert s.log_rotate_max_bytes == 5_000_000
        assert s.log_rotate_backup_count == 3
        assert s.barge_in_enabled is False
        assert s.barge_in_grace_min_seconds == 0.35
        assert s.barge_in_grace_max_seconds == 0.9
        assert s.barge_in_required_hits == 3
        assert s.barge_in_poll_interval_ms == 30

    def test_custom(self, tmp_path: Path) -> None:
        s = GeneralSettings(
            project_root=tmp_path,
            verbose=True,
            log_level="DEBUG",
            log_sensitive=True,
            log_file=tmp_path / "vox.log",
            log_rotate_max_bytes=1_024,
            log_rotate_backup_count=2,
            barge_in_grace_min_seconds=0.2,
            barge_in_grace_max_seconds=0.7,
            barge_in_required_hits=2,
            barge_in_poll_interval_ms=20,
        )
        assert s.project_root == tmp_path
        assert s.verbose is True
        assert s.log_sensitive is True
        assert s.log_file == tmp_path / "vox.log"
        assert s.log_rotate_max_bytes == 1_024
        assert s.log_rotate_backup_count == 2
        assert s.barge_in_grace_min_seconds == 0.2
        assert s.barge_in_grace_max_seconds == 0.7
        assert s.barge_in_required_hits == 2
        assert s.barge_in_poll_interval_ms == 20


class TestSTTSettings:
    def test_defaults(self) -> None:
        s = STTSettings()
        assert s.engine == "whisper_local"
        assert s.whisper_model == "base.en"
        assert s.whisper_profile == "conversational"
        assert s.whisper_beam_size == 3
        assert s.sample_rate == 16000
        assert s.silence_duration == 0.7
        assert s.silence_duration_after_speech == 0.5
        assert s.adaptive_endpointing is True


class TestLLMSettings:
    def test_defaults(self) -> None:
        s = LLMSettings()
        assert s.provider == "claude"
        assert s.max_tokens == 1024
        assert s.temperature == 0.3
        assert s.max_history_turns == 10
        assert s.prompt_caching_enabled is True


class TestTTSSettings:
    def test_defaults(self) -> None:
        s = TTSSettings()
        assert s.engine == "macos_say"
        assert s.macos_voice == "Samantha"
        assert s.macos_rate == 200


class TestMCPSettings:
    def test_defaults(self) -> None:
        s = MCPSettings()
        assert s.enabled is True
        assert s.transport == "stdio"
        assert s.tree_depth == 3


class TestContextSettings:
    def test_defaults(self) -> None:
        s = ContextSettings()
        assert s.include_files == []
        assert s.enabled_sources == []
        assert s.max_file_size == 50_000
        assert s.max_context_chars == 200_000
        assert s.interactive_compact_context is True
        assert s.interactive_max_context_chars == 60_000
        assert s.interactive_enabled_sources == ["project_info", "git", "tree"]
        assert s.read_config_files is True
        assert s.read_full_readme is True
        assert s.skip_network_sources is False
        assert len(s.doc_patterns) > 0
        assert "CHANGELOG.md" in s.doc_patterns

    def test_custom_values(self) -> None:
        s = ContextSettings(
            include_files=["src/**/*.py"],
            enabled_sources=["project_info", "git"],
            max_file_size=10_000,
            max_context_chars=50_000,
            interactive_compact_context=False,
            interactive_max_context_chars=20_000,
            interactive_enabled_sources=["project_info"],
            read_config_files=False,
            read_full_readme=False,
            skip_network_sources=True,
            doc_patterns=["NOTES.md"],
        )
        assert s.include_files == ["src/**/*.py"]
        assert s.enabled_sources == ["project_info", "git"]
        assert s.max_file_size == 10_000
        assert s.max_context_chars == 50_000
        assert s.interactive_compact_context is False
        assert s.interactive_max_context_chars == 20_000
        assert s.interactive_enabled_sources == ["project_info"]
        assert s.read_config_files is False
        assert s.read_full_readme is False
        assert s.skip_network_sources is True
        assert s.doc_patterns == ["NOTES.md"]


class TestVoxTerminalSettings:
    def test_defaults(self, clean_env: None) -> None:
        s = VoxTerminalSettings()
        assert isinstance(s.general, GeneralSettings)
        assert isinstance(s.stt, STTSettings)
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.tts, TTSSettings)
        assert isinstance(s.mcp, MCPSettings)
        assert isinstance(s.context, ContextSettings)

    def test_env_override(self, clean_env: None) -> None:
        os.environ["VOX_TERMINAL_LLM__API_KEY"] = "sk-test-key"
        s = VoxTerminalSettings()
        assert s.llm.api_key == "sk-test-key"
        del os.environ["VOX_TERMINAL_LLM__API_KEY"]

    def test_load_settings(self, clean_env: None) -> None:
        s = load_settings()
        assert isinstance(s, VoxTerminalSettings)
