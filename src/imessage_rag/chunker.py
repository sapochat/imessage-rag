"""Group raw messages into conversation chunks for embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Generator, Iterable

from imessage_rag.config import CHUNK_MAX_MESSAGES, CHUNK_WINDOW_HOURS

if TYPE_CHECKING:
    from imessage_rag.ingest import RawMessage

_ATTACHMENT_PLACEHOLDER = "\ufffc"


@dataclass
class Chunk:
    contact: str
    thread_key: str
    start_time: datetime
    end_time: datetime
    text: str
    message_count: int
    metadata: dict = field(default_factory=dict)
    embedding_text: str = ""


def _meaningful_text(text: str) -> str:
    """Remove attachment placeholders and whitespace-only content."""
    return text.replace(_ATTACHMENT_PLACEHOLDER, "").strip()


def _format_imessage_chunk(messages: list[RawMessage], contact: str) -> Chunk | None:
    """Format a list of messages from one conversation window into a Chunk."""
    lines = []
    searchable_lines = []
    for msg in messages:
        meaningful = _meaningful_text(msg.text)
        if not meaningful:
            continue
        sender = "Me" if msg.is_from_me else (msg.sender or contact)
        ts = msg.date.strftime("%Y-%m-%d %H:%M")
        line = f"[{ts}] {sender}: {meaningful}"
        lines.append(line)
        searchable_lines.append(line)

    if not lines:
        return None

    metadata = {}
    if messages[0].participants:
        metadata["participants"] = list(messages[0].participants)
    if messages[0].conversation_id:
        metadata["conversation_id"] = messages[0].conversation_id
    if len(messages) != len(lines):
        metadata["raw_message_count"] = len(messages)

    start = messages[0].date
    end = messages[-1].date
    participants = ", ".join(messages[0].participants) if messages[0].participants else contact
    embedding_text = "\n".join(
        [
            f"Conversation: {contact}",
            f"Participants: {participants}",
            f"Date range: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}",
            "Messages:",
            *searchable_lines,
        ]
    )

    return Chunk(
        contact=contact,
        thread_key=messages[0].conversation_id or contact,
        start_time=start,
        end_time=end,
        text="\n".join(lines),
        message_count=len(lines),
        metadata=metadata,
        embedding_text=embedding_text,
    )


def chunk_imessages(
    messages: Iterable[RawMessage],
    window_hours: int = CHUNK_WINDOW_HOURS,
    max_messages: int = CHUNK_MAX_MESSAGES,
) -> Generator[Chunk, None, None]:
    """Group messages by contact + time window into conversation chunks.

    Expects messages sorted by contact then date (as produced by extract_messages).
    A gap of more than `window_hours` between consecutive messages from the same
    contact starts a new chunk.
    """
    window = timedelta(hours=window_hours)
    max_messages = max(1, max_messages)
    current_contact: str | None = None
    current_conversation_id: str | None = None
    buffer: list[RawMessage] = []

    for msg in messages:
        conversation_id = msg.conversation_id or msg.contact
        if conversation_id != current_conversation_id:
            if buffer:
                chunk = _format_imessage_chunk(buffer, current_contact)
                if chunk:
                    yield chunk
            buffer = [msg]
            current_contact = msg.contact
            current_conversation_id = conversation_id
        elif buffer and (msg.date - buffer[-1].date) > window:
            chunk = _format_imessage_chunk(buffer, current_contact)
            if chunk:
                yield chunk
            buffer = [msg]
        elif len(buffer) >= max_messages:
            chunk = _format_imessage_chunk(buffer, current_contact)
            if chunk:
                yield chunk
            buffer = [msg]
        else:
            buffer.append(msg)

    if buffer:
        chunk = _format_imessage_chunk(buffer, current_contact)
        if chunk:
            yield chunk
