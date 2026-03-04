"""LLM client package — provider-agnostic interface to language models."""

from __future__ import annotations

from vox_terminal.config import LLMSettings
from vox_terminal.llm.base import LLMClient, LLMResponse, Message
from vox_terminal.llm.claude import ClaudeLLMClient


def create_llm_client(
    settings: LLMSettings, project_context: str = ""
) -> LLMClient:
    """Factory: build the appropriate LLM client from *settings*."""
    if settings.provider == "claude":
        return ClaudeLLMClient(settings, project_context)
    msg = f"Unsupported LLM provider: {settings.provider!r}"
    raise ValueError(msg)


__all__ = [
    "ClaudeLLMClient",
    "LLMClient",
    "LLMResponse",
    "Message",
    "create_llm_client",
]
