"""Tests for the LLM client module."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vox_terminal.config import LLMSettings
from vox_terminal.llm import (
    ClaudeLLMClient,
    LLMClient,
    LLMResponse,
    Message,
    create_llm_client,
)

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_user_message(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self) -> None:
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_create_with_defaults(self) -> None:
        resp = LLMResponse(content="answer", model="claude-sonnet-4-20250514")
        assert resp.content == "answer"
        assert resp.model == "claude-sonnet-4-20250514"
        assert resp.usage is None

    def test_create_with_usage(self) -> None:
        usage = {"input_tokens": 10, "output_tokens": 20}
        resp = LLMResponse(content="answer", model="test-model", usage=usage)
        assert resp.usage == usage


# ---------------------------------------------------------------------------
# ClaudeLLMClient tests
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> LLMSettings:
    """Helper to create LLMSettings with sensible test defaults."""
    defaults: dict[str, Any] = {
        "api_key": "test-api-key",
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 512,
        "temperature": 0.3,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)


class TestClaudeLLMClientInit:
    """Tests for ClaudeLLMClient initialization."""

    def test_creates_anthropic_client(self) -> None:
        settings = _make_settings()
        client = ClaudeLLMClient(settings, project_context="ctx")
        assert client._client is not None

    def test_system_prompt_includes_context(self) -> None:
        settings = _make_settings()
        client = ClaudeLLMClient(settings, project_context="Python web app")
        prompt = client.system_prompt
        assert "Vox-Terminal" in prompt
        assert "Python web app" in prompt
        assert "<project_context>" in prompt
        assert "untrusted repository data" in prompt

    def test_system_prompt_default_empty_context(self) -> None:
        settings = _make_settings()
        client = ClaudeLLMClient(settings)
        prompt = client.system_prompt
        assert "<project_context>" in prompt

    def test_is_llm_client(self) -> None:
        settings = _make_settings()
        client = ClaudeLLMClient(settings)
        assert isinstance(client, LLMClient)


class TestClaudeLLMClientBuildMessages:
    """Tests for internal message building."""

    def test_prompt_only(self) -> None:
        messages = ClaudeLLMClient._build_messages("Hello", None)
        assert messages == [{"role": "user", "content": "Hello"}]

    def test_with_history(self) -> None:
        history = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        messages = ClaudeLLMClient._build_messages("Follow up", history)
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hi"}
        assert messages[1] == {"role": "assistant", "content": "Hello!"}
        assert messages[2] == {"role": "user", "content": "Follow up"}


class TestClaudeLLMClientStream:
    """Tests for the stream() method with a mocked Anthropic client."""

    @pytest.fixture()
    def mock_client(self) -> ClaudeLLMClient:
        """Create a ClaudeLLMClient with a mocked Anthropic async client."""
        settings = _make_settings()
        client = ClaudeLLMClient(settings, project_context="test project")
        return client

    async def test_stream_yields_text_deltas(self, mock_client: ClaudeLLMClient) -> None:
        deltas = ["Hello", " world", "!"]

        # Build the async iterator for text_stream
        async def _text_stream() -> Any:
            for d in deltas:
                yield d

        # Build the context manager returned by messages.stream()
        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock()
        stream_response = MagicMock()
        stream_response.text_stream = _text_stream()
        stream_cm.__aenter__.return_value = stream_response
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client._client = MagicMock()
        mock_client._client.messages = MagicMock()
        mock_client._client.messages.stream = MagicMock(return_value=stream_cm)

        collected: list[str] = []
        async for chunk in mock_client.stream("test prompt"):
            collected.append(chunk)

        assert collected == deltas

    async def test_stream_passes_correct_params(self, mock_client: ClaudeLLMClient) -> None:
        async def _empty_stream() -> Any:
            return
            yield

        stream_cm = AsyncMock()
        stream_response = MagicMock()
        stream_response.text_stream = _empty_stream()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_messages = MagicMock()
        mock_messages.stream = MagicMock(return_value=stream_cm)
        mock_client._client = MagicMock()
        mock_client._client.messages = mock_messages

        async for _ in mock_client.stream("hello"):
            pass

        mock_messages.stream.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.3,
            system=mock_client.system_prompt,
            messages=[{"role": "user", "content": "hello"}],
            cache_control={"type": "ephemeral"},
        )

    async def test_stream_omits_cache_control_when_disabled(self, mock_client: ClaudeLLMClient) -> None:
        async def _empty_stream() -> Any:
            return
            yield

        stream_cm = AsyncMock()
        stream_response = MagicMock()
        stream_response.text_stream = _empty_stream()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_messages = MagicMock()
        mock_messages.stream = MagicMock(return_value=stream_cm)
        mock_client._client = MagicMock()
        mock_client._client.messages = mock_messages
        object.__setattr__(mock_client._settings, "prompt_caching_enabled", False)

        async for _ in mock_client.stream("hello"):
            pass

        call_kwargs = mock_messages.stream.call_args.kwargs
        assert "cache_control" not in call_kwargs


class TestClaudeLLMClientAsk:
    """Tests for the ask() method."""

    async def test_ask_returns_llm_response(self) -> None:
        settings = _make_settings()
        client = ClaudeLLMClient(settings, project_context="ctx")

        deltas = ["Collected", " response"]

        async def _text_stream() -> Any:
            for d in deltas:
                yield d

        stream_cm = AsyncMock()
        stream_response = MagicMock()
        stream_response.text_stream = _text_stream()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        client._client = MagicMock()
        client._client.messages = MagicMock()
        client._client.messages.stream = MagicMock(return_value=stream_cm)

        response = await client.ask("test")

        assert isinstance(response, LLMResponse)
        assert response.content == "Collected response"
        assert response.model == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateLLMClient:
    """Tests for the create_llm_client factory function."""

    def test_creates_claude_client(self) -> None:
        settings = _make_settings(provider="claude")
        client = create_llm_client(settings, project_context="ctx")
        assert isinstance(client, ClaudeLLMClient)

    def test_unsupported_provider_raises(self) -> None:
        # We have to bypass the Literal type check at runtime
        settings = _make_settings()
        object.__setattr__(settings, "provider", "openai")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client(settings)


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------


class TestClaudeLLMClientTimeout:
    async def test_stream_timeout_raises(self) -> None:
        settings = _make_settings(stream_timeout=0.1)
        client = ClaudeLLMClient(settings, project_context="test")

        async def _slow_stream() -> Any:
            await asyncio.sleep(10)
            yield "never"

        stream_cm = AsyncMock()
        stream_response = MagicMock()
        stream_response.text_stream = _slow_stream()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_response)
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        client._client = MagicMock()
        client._client.messages = MagicMock()
        client._client.messages.stream = MagicMock(return_value=stream_cm)

        with pytest.raises(TimeoutError):
            async for _ in client.stream("test"):
                pass
