"""Tests for iMessage extraction and timestamp conversion."""

import sqlite3
from datetime import datetime, timezone

from src.config import APPLE_EPOCH_OFFSET
from src.ingest.imessage import (
    RawMessage,
    apple_ts_to_datetime,
    datetime_to_apple_ts,
    extract_messages,
    _extract_text_from_attributed_body,
)


class TestAppleTimestamps:
    def test_roundtrip(self):
        """Converting datetime→apple→datetime should be identity."""
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        apple_ns = datetime_to_apple_ts(dt)
        result = apple_ts_to_datetime(apple_ns)
        assert abs((result - dt).total_seconds()) < 1

    def test_known_epoch(self):
        """Apple epoch (2001-01-01 00:00:00 UTC) should map to nanosecond value 0."""
        dt = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        apple_ns = datetime_to_apple_ts(dt)
        assert apple_ns == 0

    def test_before_epoch(self):
        """Dates before Apple epoch produce negative nanosecond values."""
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        apple_ns = datetime_to_apple_ts(dt)
        assert apple_ns < 0

    def test_recent_date(self):
        """A recent date produces a large positive nanosecond value."""
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        apple_ns = datetime_to_apple_ts(dt)
        assert apple_ns > 0
        # Verify the Unix timestamp relationship
        expected_unix = dt.timestamp()
        reconstructed_unix = apple_ns / 1_000_000_000 + APPLE_EPOCH_OFFSET
        assert abs(reconstructed_unix - expected_unix) < 1


class TestExtractTextFromAttributedBody:
    def test_returns_none_for_no_marker(self):
        assert _extract_text_from_attributed_body(b"no marker here") is None

    def test_returns_none_for_truncated_blob(self):
        """Blob with NSString marker but truncated before text."""
        blob = b"prefix\x00NSString"
        assert _extract_text_from_attributed_body(blob) is None

    def test_extracts_short_text(self):
        """Build a minimal blob with NSString marker + short length + text."""
        text = b"Hello world"
        # NSString marker, then some type bytes, then 0x2B (+), then length, then text
        blob = b"\x00\x00NSString\x01\x02\x2B" + bytes([len(text)]) + text
        result = _extract_text_from_attributed_body(blob)
        assert result == "Hello world"

    def test_extracts_multibyte_length(self):
        """Text longer than 127 bytes uses multi-byte length encoding."""
        text = b"A" * 200
        # 0x81 means 1 extra byte follows with the length
        length_bytes = b"\x81" + (200).to_bytes(1, "little")
        blob = b"\x00NSString\x2B" + length_bytes + text
        result = _extract_text_from_attributed_body(blob)
        assert result == "A" * 200

    def test_handles_utf8_text(self):
        text = "Café résumé".encode("utf-8")
        blob = b"\x00NSString\x2B" + bytes([len(text)]) + text
        result = _extract_text_from_attributed_body(blob)
        assert result == "Café résumé"


class TestExtractMessages:
    def test_empty_db(self, imessage_db):
        """No messages in the DB yields nothing."""
        msgs = list(extract_messages(db_path=imessage_db))
        assert msgs == []

    def test_extracts_plain_text_messages(self, imessage_db):
        """Messages with a text column are extracted directly."""
        conn = sqlite3.connect(str(imessage_db))
        conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+15551234567')")
        # Apple timestamp for 2024-01-15 12:00:00 UTC
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        apple_ns = datetime_to_apple_ts(dt)
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (1, 'Hello', ?, 0, 1)",
            (apple_ns,),
        )
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (2, 'Hi back', ?, 1, 1)",
            (apple_ns + 60_000_000_000,),  # 60 seconds later
        )
        conn.commit()
        conn.close()

        msgs = list(extract_messages(db_path=imessage_db))
        assert len(msgs) == 2
        assert msgs[0].text == "Hello"
        assert msgs[0].contact == "+15551234567"
        assert msgs[0].is_from_me is False
        assert msgs[1].text == "Hi back"
        assert msgs[1].is_from_me is True

    def test_since_filter(self, imessage_db):
        """The since parameter filters out older messages."""
        conn = sqlite3.connect(str(imessage_db))
        conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+15551234567')")

        old_dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        new_dt = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (1, 'Old message', ?, 0, 1)",
            (datetime_to_apple_ts(old_dt),),
        )
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (2, 'New message', ?, 0, 1)",
            (datetime_to_apple_ts(new_dt),),
        )
        conn.commit()
        conn.close()

        cutoff = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        msgs = list(extract_messages(since=cutoff, db_path=imessage_db))
        assert len(msgs) == 1
        assert msgs[0].text == "New message"

    def test_skips_empty_text(self, imessage_db):
        """Messages with NULL/empty text and no attributedBody are skipped."""
        conn = sqlite3.connect(str(imessage_db))
        conn.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+15551234567')")
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (1, '', ?, 0, 1)",
            (datetime_to_apple_ts(dt),),
        )
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (2, NULL, ?, 0, 1)",
            (datetime_to_apple_ts(dt),),
        )
        conn.commit()
        conn.close()

        msgs = list(extract_messages(db_path=imessage_db))
        assert msgs == []

    def test_unknown_contact_for_missing_handle(self, imessage_db):
        """Messages with no handle get contact='unknown'."""
        conn = sqlite3.connect(str(imessage_db))
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        conn.execute(
            "INSERT INTO message (ROWID, text, date, is_from_me, handle_id) VALUES (1, 'orphan', ?, 0, 999)",
            (datetime_to_apple_ts(dt),),
        )
        conn.commit()
        conn.close()

        msgs = list(extract_messages(db_path=imessage_db))
        assert len(msgs) == 1
        assert msgs[0].contact == "unknown"
