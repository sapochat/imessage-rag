"""Shared fixtures for the imessage-rag test suite."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from imessage_rag.chunker import Chunk


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
    conn.execute("""
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY,
            guid TEXT,
            display_name TEXT,
            room_name TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE chat_message_join (
            chat_id INTEGER,
            message_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE chat_handle_join (
            chat_id INTEGER,
            handle_id INTEGER
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def make_chunk(
    contact="test-contact",
    thread_key=None,
    start_time=None,
    end_time=None,
    text="Hello, this is a test message.",
    message_count=1,
    metadata=None,
) -> Chunk:
    """Helper to build a Chunk with sensible defaults."""
    now = datetime.now(tz=timezone.utc)
    return Chunk(
        contact=contact,
        thread_key=thread_key or contact,
        start_time=start_time or now,
        end_time=end_time or now,
        text=text,
        message_count=message_count,
        metadata=metadata or {},
    )
