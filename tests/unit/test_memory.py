"""Tests for the persistent conversation memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from vox_terminal.memory.store import ConversationStore


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    """Create a ConversationStore with a temporary database."""
    return ConversationStore(db_path=tmp_path / "test.db")


class TestConversationStore:
    def test_create_session(self, store: ConversationStore) -> None:
        session_id = store.create_session("/projects/test")
        assert isinstance(session_id, str)
        assert len(session_id) == 32  # uuid hex

    def test_round_trip(self, store: ConversationStore) -> None:
        session_id = store.create_session("/projects/test")
        store.add_message(session_id, "user", "Hello")
        store.add_message(session_id, "assistant", "Hi there!")

        messages = store.get_recent_messages("/projects/test")
        assert len(messages) == 2
        assert messages[0] == ("user", "Hello")
        assert messages[1] == ("assistant", "Hi there!")

    def test_max_messages_limit(self, store: ConversationStore) -> None:
        session_id = store.create_session("/projects/test")
        for i in range(10):
            store.add_message(session_id, "user", f"Question {i}")
            store.add_message(session_id, "assistant", f"Answer {i}")

        messages = store.get_recent_messages("/projects/test", max_messages=4)
        assert len(messages) == 4
        # Should be the 4 most recent messages in chronological order
        assert messages[0] == ("user", "Question 8")
        assert messages[1] == ("assistant", "Answer 8")
        assert messages[2] == ("user", "Question 9")
        assert messages[3] == ("assistant", "Answer 9")

    def test_chronological_order(self, store: ConversationStore) -> None:
        session_id = store.create_session("/projects/test")
        store.add_message(session_id, "user", "First")
        store.add_message(session_id, "assistant", "Second")
        store.add_message(session_id, "user", "Third")

        messages = store.get_recent_messages("/projects/test")
        roles = [m[0] for m in messages]
        contents = [m[1] for m in messages]
        assert roles == ["user", "assistant", "user"]
        assert contents == ["First", "Second", "Third"]

    def test_project_isolation(self, store: ConversationStore) -> None:
        s1 = store.create_session("/projects/alpha")
        s2 = store.create_session("/projects/beta")

        store.add_message(s1, "user", "Alpha question")
        store.add_message(s2, "user", "Beta question")

        alpha = store.get_recent_messages("/projects/alpha")
        beta = store.get_recent_messages("/projects/beta")

        assert len(alpha) == 1
        assert alpha[0] == ("user", "Alpha question")
        assert len(beta) == 1
        assert beta[0] == ("user", "Beta question")

    def test_messages_across_sessions(self, store: ConversationStore) -> None:
        """Messages from multiple sessions for the same project are combined."""
        s1 = store.create_session("/projects/test")
        store.add_message(s1, "user", "Session 1 question")
        store.add_message(s1, "assistant", "Session 1 answer")

        s2 = store.create_session("/projects/test")
        store.add_message(s2, "user", "Session 2 question")

        messages = store.get_recent_messages("/projects/test")
        assert len(messages) == 3

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persist.db"
        store1 = ConversationStore(db_path=db_path)
        session_id = store1.create_session("/projects/test")
        store1.add_message(session_id, "user", "Remember me")
        store1.close()

        store2 = ConversationStore(db_path=db_path)
        messages = store2.get_recent_messages("/projects/test")
        assert len(messages) == 1
        assert messages[0] == ("user", "Remember me")
        store2.close()

    def test_empty_project_returns_empty(self, store: ConversationStore) -> None:
        messages = store.get_recent_messages("/nonexistent/project")
        assert messages == []
