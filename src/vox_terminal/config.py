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
    memory_enabled: bool = True
    memory_db_path: Path | None = None
    memory_max_messages: int = 40
    barge_in_enabled: bool = False


class STTSettings(BaseModel):
    """Speech-to-text settings."""

    engine: Literal["whisper_local", "openai"] = "whisper_local"
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    sample_rate: int = 16000
    openai_api_key: str = ""
    silence_threshold: float = 0.01
    silence_duration: float = 1.5
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


def load_settings(**overrides: object) -> VoxTerminalSettings:
    """Load settings from TOML file + environment variables."""
    return VoxTerminalSettings(**overrides)  # type: ignore[arg-type]
