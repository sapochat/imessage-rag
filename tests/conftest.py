"""Shared fixtures for the personal-rag test suite."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.chunker import Chunk


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory (pytest built-in)."""
    return tmp_path


@pytest.fixture
def vector_db(tmp_path):
    """Provide a fresh temporary vector DB path."""
    return tmp_path / "test_vectors.db"


@pytest.fixture
def imessage_db(tmp_path):
    """Create a minimal iMessage-style SQLite database for testing."""
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE handle (
            ROWID INTEGER PRIMARY KEY,
            id TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            attributedBody BLOB,
            date INTEGER,
            is_from_me INTEGER,
            handle_id INTEGER
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def make_chunk(
    source="imessage",
    contact="test-contact",
    start_time=None,
    end_time=None,
    text="Hello, this is a test message.",
    message_count=1,
    metadata=None,
) -> Chunk:
    """Helper to build a Chunk with sensible defaults."""
    now = datetime.now(tz=timezone.utc)
    return Chunk(
        source=source,
        contact=contact,
        start_time=start_time or now,
        end_time=end_time or now,
        text=text,
        message_count=message_count,
        metadata=metadata or {},
    )
