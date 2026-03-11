"""Tests for the chunking logic."""

from datetime import datetime, timedelta, timezone

from src.chunker import Chunk, chunk_emails, chunk_imessages
from src.ingest.email import RawEmail
from src.ingest.imessage import RawMessage


def _msg(contact, minutes_offset, text="Hi", is_from_me=False):
    """Helper to create a RawMessage at a given minute offset."""
    base = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return RawMessage(
        rowid=minutes_offset,
        text=text,
        date=base + timedelta(minutes=minutes_offset),
        is_from_me=is_from_me,
        contact=contact,
    )


class TestChunkImessages:
    def test_empty_input(self):
        assert list(chunk_imessages([])) == []

    def test_single_message(self):
        msgs = [_msg("alice", 0, "Hello")]
        chunks = list(chunk_imessages(msgs))
        assert len(chunks) == 1
        assert chunks[0].source == "imessage"
        assert chunks[0].contact == "alice"
        assert chunks[0].message_count == 1
        assert "Hello" in chunks[0].text

    def test_groups_within_window(self):
        """Messages from the same contact within the time window form one chunk."""
        msgs = [
            _msg("alice", 0, "Hello"),
            _msg("alice", 30, "How are you?"),
            _msg("alice", 60, "Goodbye"),
        ]
        chunks = list(chunk_imessages(msgs, window_hours=4))
        assert len(chunks) == 1
        assert chunks[0].message_count == 3

    def test_splits_on_time_gap(self):
        """A gap larger than window_hours starts a new chunk."""
        msgs = [
            _msg("alice", 0, "Morning"),
            _msg("alice", 300, "Afternoon"),  # 5 hours later
        ]
        chunks = list(chunk_imessages(msgs, window_hours=4))
        assert len(chunks) == 2
        assert chunks[0].message_count == 1
        assert chunks[1].message_count == 1

    def test_splits_on_contact_change(self):
        """Different contacts always produce separate chunks."""
        msgs = [
            _msg("alice", 0, "From Alice"),
            _msg("bob", 1, "From Bob"),
        ]
        chunks = list(chunk_imessages(msgs, window_hours=4))
        assert len(chunks) == 2
        assert chunks[0].contact == "alice"
        assert chunks[1].contact == "bob"

    def test_chunk_text_format(self):
        """Chunk text has the expected [timestamp] sender: message format."""
        msgs = [
            _msg("alice", 0, "Hello", is_from_me=False),
            _msg("alice", 1, "Hi back", is_from_me=True),
        ]
        chunks = list(chunk_imessages(msgs))
        assert "[2024-01-15 12:00] alice: Hello" in chunks[0].text
        assert "[2024-01-15 12:01] Me: Hi back" in chunks[0].text

    def test_start_end_times(self):
        """Chunk start_time and end_time match first/last message."""
        msgs = [
            _msg("alice", 0, "First"),
            _msg("alice", 60, "Last"),
        ]
        chunks = list(chunk_imessages(msgs))
        assert chunks[0].start_time == msgs[0].date
        assert chunks[0].end_time == msgs[1].date

    def test_multiple_contacts_interleaved(self):
        """Handles sorted-by-contact input with multiple contacts."""
        msgs = [
            _msg("alice", 0),
            _msg("alice", 10),
            _msg("bob", 5),
            _msg("bob", 15),
        ]
        chunks = list(chunk_imessages(msgs))
        assert len(chunks) == 2
        assert chunks[0].contact == "alice"
        assert chunks[1].contact == "bob"


class TestChunkEmails:
    def test_empty_input(self):
        assert list(chunk_emails([])) == []

    def test_single_email(self):
        email = RawEmail(
            filepath="/test/1.emlx",
            subject="Test Subject",
            sender="sender@example.com",
            recipients="recip@example.com",
            date=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            body="Email body text.",
            message_id="<test-123@example.com>",
        )
        chunks = list(chunk_emails([email]))
        assert len(chunks) == 1
        c = chunks[0]
        assert c.source == "email"
        assert c.contact == "sender@example.com"
        assert c.message_count == 1
        assert "From: sender@example.com" in c.text
        assert "Subject: Test Subject" in c.text
        assert "Email body text." in c.text
        assert c.metadata == {"message_id": "<test-123@example.com>"}

    def test_email_without_message_id(self):
        email = RawEmail(
            filepath="/test/2.emlx",
            subject="No ID",
            sender="a@b.com",
            recipients="c@d.com",
            date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            body="Body.",
            message_id="",
        )
        chunks = list(chunk_emails([email]))
        assert chunks[0].metadata == {}

    def test_multiple_emails_each_one_chunk(self):
        emails = [
            RawEmail(
                filepath=f"/test/{i}.emlx",
                subject=f"Email {i}",
                sender=f"sender{i}@example.com",
                recipients="r@example.com",
                date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                body=f"Body {i}",
                message_id=f"<{i}@example.com>",
            )
            for i in range(3)
        ]
        chunks = list(chunk_emails(emails))
        assert len(chunks) == 3
        for i, c in enumerate(chunks):
            assert c.contact == f"sender{i}@example.com"
