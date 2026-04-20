"""Group raw messages into conversation chunks for embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Generator, Iterable

from src.config import CHUNK_WINDOW_HOURS

if TYPE_CHECKING:
    from src.ingest.email import RawEmail
    from src.ingest.imessage import RawMessage


@dataclass
class Chunk:
    source: str  # 'imessage' or 'email'
    contact: str
    thread_key: str
    start_time: datetime
    end_time: datetime
    text: str
    message_count: int
    metadata: dict = field(default_factory=dict)


def _format_imessage_chunk(messages: list[RawMessage], contact: str) -> Chunk:
    """Format a list of messages from one conversation window into a Chunk."""
    lines = []
    for msg in messages:
        sender = "Me" if msg.is_from_me else (msg.sender or contact)
        ts = msg.date.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{ts}] {sender}: {msg.text}")

    metadata = {}
    if messages[0].participants:
        metadata["participants"] = list(messages[0].participants)
    if messages[0].conversation_id:
        metadata["conversation_id"] = messages[0].conversation_id

    return Chunk(
        source="imessage",
        contact=contact,
        thread_key=messages[0].conversation_id or contact,
        start_time=messages[0].date,
        end_time=messages[-1].date,
        text="\n".join(lines),
        message_count=len(messages),
        metadata=metadata,
    )


def chunk_imessages(
    messages: Iterable[RawMessage],
    window_hours: int = CHUNK_WINDOW_HOURS,
) -> Generator[Chunk, None, None]:
    """Group messages by contact + time window into conversation chunks.

    Expects messages sorted by contact then date (as produced by extract_messages).
    A gap of more than `window_hours` between consecutive messages from the same
    contact starts a new chunk.
    """
    window = timedelta(hours=window_hours)
    current_contact: str | None = None
    current_conversation_id: str | None = None
    buffer: list[RawMessage] = []

    for msg in messages:
        conversation_id = msg.conversation_id or msg.contact
        if conversation_id != current_conversation_id:
            # New conversation — flush buffer
            if buffer:
                yield _format_imessage_chunk(buffer, current_contact)
            buffer = [msg]
            current_contact = msg.contact
            current_conversation_id = conversation_id
        elif buffer and (msg.date - buffer[-1].date) > window:
            # Same conversation but gap exceeds window — flush and start new chunk
            yield _format_imessage_chunk(buffer, current_contact)
            buffer = [msg]
        else:
            buffer.append(msg)

    # Flush remaining
    if buffer:
        yield _format_imessage_chunk(buffer, current_contact)


def chunk_emails(
    emails: Iterable[RawEmail],
) -> Generator[Chunk, None, None]:
    """Create one chunk per email with a formatted header + body."""
    for em in emails:
        date_str = em.date.strftime("%Y-%m-%d %H:%M")
        text = (
            f"From: {em.sender}\n"
            f"To: {em.recipients}\n"
            f"Date: {date_str}\n"
            f"Subject: {em.subject}\n\n"
            f"{em.body}"
        )
        metadata = {}
        if em.message_id:
            metadata["message_id"] = em.message_id
        yield Chunk(
            source="email",
            contact=em.sender,
            thread_key=em.message_id or em.sender,
            start_time=em.date,
            end_time=em.date,
            text=text,
            message_count=1,
            metadata=metadata,
        )
