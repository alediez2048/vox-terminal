"""Configuration module using pydantic-settings with TOML support."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeneralSettings(BaseModel):
    """General application settings."""

    project_root: Path = Field(default_factory=Path.cwd)
    verbose: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_sensitive: bool = False
    log_file: Path | None = None
    log_rotate_max_bytes: int = 5_000_000
    log_rotate_backup_count: int = 3
    memory_enabled: bool = True
    memory_db_path: Path | None = None
    memory_max_messages: int = 40
    barge_in_enabled: bool = False
    barge_in_grace_min_seconds: float = 0.35
    barge_in_grace_max_seconds: float = 0.9
    barge_in_required_hits: int = 3
    barge_in_poll_interval_ms: int = 30


class STTSettings(BaseModel):
    """Speech-to-text settings."""

    engine: Literal["whisper_local", "openai"] = "whisper_local"
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_profile: Literal["conversational", "balanced", "accurate"] = "conversational"
    whisper_beam_size: int = 3
    sample_rate: int = 16000
    openai_api_key: str = ""
    silence_threshold: float = 0.01
    speech_start_threshold: float | None = None
    speech_end_threshold: float | None = None
    silence_duration: float = 0.7
    silence_duration_after_speech: float = 0.5
    adaptive_endpointing: bool = True
    max_record_duration: float = 30.0
    vad_engine: Literal["silero", "energy"] = "silero"
    vad_threshold: float = 0.5


class LLMSettings(BaseModel):
    """LLM settings."""

    provider: Literal["claude"] = "claude"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.3
    api_key: str = ""
    max_history_turns: int = 10
    stream_timeout: float = 60.0
    prompt_caching_enabled: bool = True


class TTSSettings(BaseModel):
    """Text-to-speech settings."""

    engine: Literal["macos_say", "openai", "piper", "elevenlabs"] = "macos_say"
    macos_voice: str = "Samantha"
    macos_rate: int = 200
    openai_api_key: str = ""
    openai_voice: str = "alloy"
    openai_model: str = "tts-1"
    piper_model_path: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_model_id: str = "eleven_flash_v2_5"
    elevenlabs_speed: float = 1.0
    elevenlabs_output_format: str = "mp3_44100_128"


class ContextSettings(BaseModel):
    """Settings for project context assembly."""

    include_files: list[str] = Field(default_factory=list)
    enabled_sources: list[str] = Field(default_factory=list)
    max_file_size: int = 50_000
    max_context_chars: int = 200_000
    interactive_compact_context: bool = True
    interactive_max_context_chars: int = 60_000
    interactive_enabled_sources: list[str] = Field(
        default_factory=lambda: ["project_info", "git", "tree"]
    )
    read_config_files: bool = True
    read_full_readme: bool = True
    skip_network_sources: bool = False
    doc_patterns: list[str] = Field(
        default_factory=lambda: [
            "DEVLOG.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            "CLAUDE.md",
            "AGENTS.md",
            "docs/*.md",
        ]
    )


class MCPSettings(BaseModel):
    """MCP server settings."""

    enabled: bool = True
    transport: Literal["stdio"] = "stdio"
    include_git: bool = True
    include_tree: bool = True
    tree_depth: int = 3


class VoxTerminalSettings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_prefix="VOX_TERMINAL_",
        env_nested_delimiter="__",
        toml_file="vox-terminal.toml",
        extra="ignore",
    )

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)


def load_settings(**overrides: object) -> VoxTerminalSettings:
    """Load settings from TOML file + environment variables."""
    return VoxTerminalSettings(**overrides)  # type: ignore[arg-type]
