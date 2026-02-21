"""Streaming extraction of messages from the iMessage SQLite database."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from src.config import APPLE_EPOCH_OFFSET, IMESSAGE_DB

BATCH_SIZE = 500

_NSSTRING_MARKER = b"NSString"


@dataclass
class RawMessage:
    rowid: int
    text: str
    date: datetime
    is_from_me: bool
    contact: str  # phone number or email


def apple_ts_to_datetime(apple_ns: int) -> datetime:
    """Convert Apple Core Data nanosecond timestamp to UTC datetime."""
    unix_ts = apple_ns / 1_000_000_000 + APPLE_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def datetime_to_apple_ts(dt: datetime) -> int:
    """Convert a datetime to Apple Core Data nanosecond timestamp."""
    unix_ts = dt.timestamp()
    return int((unix_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000)


def _extract_text_from_attributed_body(blob: bytes) -> str | None:
    """Extract plain text from an NSAttributedString typedstream blob.

    The blob is a serialized NSAttributedString. The actual text content
    lives after an 'NSString' marker, preceded by a '+' (0x2B) byte and
    a variable-length size field:
      - 0x01–0x7F: size is the byte value directly
      - 0x8N (N>0):  N more bytes follow holding the size (little-endian)
    """
    idx = blob.find(_NSSTRING_MARKER)
    if idx == -1:
        return None

    # Advance past the marker and skip type-descriptor bytes until '+'
    pos = idx + len(_NSSTRING_MARKER)
    while pos < len(blob) and blob[pos] != 0x2B:
        pos += 1

    pos += 1  # skip the '+' itself
    if pos >= len(blob):
        return None

    # Read length
    length_byte = blob[pos]
    pos += 1

    if length_byte < 0x80:
        text_len = length_byte
    else:
        num_extra = length_byte & 0x7F
        if pos + num_extra > len(blob):
            return None
        text_len = int.from_bytes(blob[pos : pos + num_extra], "little")
        pos += num_extra

    if pos + text_len > len(blob):
        return None

    return blob[pos : pos + text_len].decode("utf-8", errors="replace")


def extract_messages(
    since: datetime | None = None,
    db_path: Path = IMESSAGE_DB,
) -> Generator[RawMessage, None, None]:
    """Stream messages from chat.db, optionally filtered by date.

    Prefers the `text` column when available, otherwise decodes the
    `attributedBody` blob.  Yields RawMessage objects sorted by contact
    then date, suitable for downstream chunking by conversation window.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT
                m.ROWID   AS rowid,
                m.text    AS text,
                m.attributedBody AS attributed_body,
                m.date    AS date,
                m.is_from_me AS is_from_me,
                COALESCE(h.id, 'unknown') AS contact
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE ((m.text IS NOT NULL AND m.text != '')
               OR m.attributedBody IS NOT NULL)
        """
        params: list = []

        if since is not None:
            apple_cutoff = datetime_to_apple_ts(since)
            query += " AND m.date >= ?"
            params.append(apple_cutoff)

        query += " ORDER BY contact, m.date"

        cursor = conn.execute(query, params)

        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break
            for row in rows:
                text = row["text"]
                if not text and row["attributed_body"]:
                    text = _extract_text_from_attributed_body(
                        bytes(row["attributed_body"])
                    )
                if not text:
                    continue

                yield RawMessage(
                    rowid=row["rowid"],
                    text=text,
                    date=apple_ts_to_datetime(row["date"]),
                    is_from_me=bool(row["is_from_me"]),
                    contact=row["contact"],
                )
    finally:
        conn.close()
