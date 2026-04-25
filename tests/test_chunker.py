"""Tests for the chunking logic."""

from datetime import datetime, timedelta, timezone

from imessage_rag.chunker import Chunk, chunk_imessages
from imessage_rag.ingest import RawMessage


def _msg(contact, minutes_offset, text="Hi", is_from_me=False):
    """Helper to create a RawMessage at a given minute offset."""
    base = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return RawMessage(
        rowid=minutes_offset,
        text=text,
        date=base + timedelta(minutes=minutes_offset),
        is_from_me=is_from_me,
        contact=contact,
        sender=contact,
        conversation_id=f"thread:{contact}",
        participants=(contact,),
    )


class TestChunkImessages:
    def test_empty_input(self):
        assert list(chunk_imessages([])) == []

    def test_single_message(self):
        msgs = [_msg("alice", 0, "Hello")]
        chunks = list(chunk_imessages(msgs))
        assert len(chunks) == 1
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

    def test_splits_on_max_messages(self):
        msgs = [_msg("alice", i, f"Message {i}") for i in range(5)]

        chunks = list(chunk_imessages(msgs, window_hours=4, max_messages=2))

        assert [chunk.message_count for chunk in chunks] == [2, 2, 1]
        assert "Message 0" in chunks[0].text
        assert "Message 2" in chunks[1].text
        assert "Message 4" in chunks[2].text

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

    def test_embedding_text_includes_searchable_metadata(self):
        msgs = [_msg("alice", 0, "LA trip")]
        chunks = list(chunk_imessages(msgs))

        assert "Conversation: alice" in chunks[0].embedding_text
        assert "Participants: alice" in chunks[0].embedding_text
        assert "Date range: 2024-01-15" in chunks[0].embedding_text
        assert "LA trip" in chunks[0].embedding_text

    def test_skips_attachment_only_messages(self):
        msgs = [
            _msg("alice", 0, "\ufffc"),
            _msg("alice", 1, "Actual text"),
        ]
        chunks = list(chunk_imessages(msgs))

        assert len(chunks) == 1
        assert "\ufffc" not in chunks[0].text
        assert chunks[0].message_count == 1
        assert chunks[0].metadata["raw_message_count"] == 2

    def test_drops_attachment_only_chunks(self):
        assert list(chunk_imessages([_msg("alice", 0, "\ufffc")])) == []

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

    def test_group_chat_uses_sender_not_group_label(self):
        msgs = [
            RawMessage(
                rowid=1,
                text="First",
                date=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                is_from_me=False,
                contact="alice, bob",
                sender="alice",
                conversation_id="chat:1",
                participants=("alice", "bob"),
            ),
            RawMessage(
                rowid=2,
                text="Reply",
                date=datetime(2024, 1, 15, 12, 1, tzinfo=timezone.utc),
                is_from_me=False,
                contact="alice, bob",
                sender="bob",
                conversation_id="chat:1",
                participants=("alice", "bob"),
            ),
        ]
        chunks = list(chunk_imessages(msgs))
        assert "alice: First" in chunks[0].text
        assert "bob: Reply" in chunks[0].text

    def test_same_label_different_thread_ids_do_not_merge(self):
        msgs = [
            RawMessage(
                rowid=1,
                text="Thread one",
                date=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                is_from_me=False,
                contact="alice, bob",
                sender="alice",
                conversation_id="chat:1",
                participants=("alice", "bob"),
            ),
            RawMessage(
                rowid=2,
                text="Thread two",
                date=datetime(2024, 1, 15, 12, 5, tzinfo=timezone.utc),
                is_from_me=False,
                contact="alice, bob",
                sender="alice",
                conversation_id="chat:2",
                participants=("alice", "bob"),
            ),
        ]
        chunks = list(chunk_imessages(msgs))
        assert len(chunks) == 2
