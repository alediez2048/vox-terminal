"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vox_terminal.config import (
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
        assert s.barge_in_enabled is False

    def test_custom(self, tmp_path: Path) -> None:
        s = GeneralSettings(project_root=tmp_path, verbose=True, log_level="DEBUG")
        assert s.project_root == tmp_path
        assert s.verbose is True


class TestSTTSettings:
    def test_defaults(self) -> None:
        s = STTSettings()
        assert s.engine == "whisper_local"
        assert s.whisper_model == "base.en"
        assert s.sample_rate == 16000


class TestLLMSettings:
    def test_defaults(self) -> None:
        s = LLMSettings()
        assert s.provider == "claude"
        assert s.max_tokens == 1024
        assert s.temperature == 0.3
        assert s.max_history_turns == 10


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


class TestVoxTerminalSettings:
    def test_defaults(self, clean_env: None) -> None:
        s = VoxTerminalSettings()
        assert isinstance(s.general, GeneralSettings)
        assert isinstance(s.stt, STTSettings)
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.tts, TTSSettings)
        assert isinstance(s.mcp, MCPSettings)

    def test_env_override(self, clean_env: None) -> None:
        os.environ["VOX_TERMINAL_LLM__API_KEY"] = "sk-test-key"
        s = VoxTerminalSettings()
        assert s.llm.api_key == "sk-test-key"
        del os.environ["VOX_TERMINAL_LLM__API_KEY"]

    def test_load_settings(self, clean_env: None) -> None:
        s = load_settings()
        assert isinstance(s, VoxTerminalSettings)
