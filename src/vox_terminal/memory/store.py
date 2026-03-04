"""Persistent conversation store backed by SQLite."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".vox-terminal"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    project_root TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_root);
"""


class ConversationStore:
    """SQLite-backed store for conversation history.

    Uses WAL mode for safe concurrent reads.  The database file is placed
    at *db_path* (defaults to ``~/.vox-terminal/conversations.db``).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = _DEFAULT_DB_DIR / "conversations.db"
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.debug("ConversationStore opened at %s", db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, project_root: str) -> str:
        """Create a new session and return its ID."""
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO sessions (id, started_at, project_root) VALUES (?, ?, ?)",
            (session_id, now, project_root),
        )
        self._conn.commit()
        logger.debug("Session created: %s", session_id)
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Persist a single message."""
        msg_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (msg_id, session_id, role, content, now),
        )
        self._conn.commit()

    def get_recent_messages(
        self,
        project_root: str,
        max_messages: int = 40,
    ) -> list[tuple[str, str]]:
        """Return recent ``(role, content)`` pairs for *project_root*.

        Messages are returned in chronological order (oldest first) and
        limited to *max_messages*.
        """
        rows = self._conn.execute(
            """
            SELECT m.role, m.content
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE s.project_root = ?
            ORDER BY m.timestamp DESC
            LIMIT ?
            """,
            (project_root, max_messages),
        ).fetchall()
        # Reverse to chronological order
        return list(reversed(rows))

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("ConversationStore closed")
