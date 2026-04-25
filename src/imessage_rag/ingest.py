"""Streaming extraction of messages from the iMessage SQLite database."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable

from imessage_rag.contacts import ContactResolver, get_default_resolver, normalize_handle
from imessage_rag.config import APPLE_EPOCH_OFFSET, IMESSAGE_DB

BATCH_SIZE = 500

_NSSTRING_MARKER = b"NSString"


@dataclass
class RawMessage:
    rowid: int
    text: str
    date: datetime
    is_from_me: bool
    contact: str  # conversation label: contact or group participant list
    sender: str = "unknown"  # actual message sender for group chats
    conversation_id: str = ""
    participants: tuple[str, ...] = ()


def _normalize_handle(value: str) -> str:
    """Normalize phone numbers and emails for matching."""
    return normalize_handle(value)


def _canonicalize_handles(values: Iterable[str]) -> tuple[str, ...]:
    """Return a sorted, unique tuple of normalized handles."""
    normalized = {_normalize_handle(value) for value in values if value and value.strip()}
    return tuple(sorted(value for value in normalized if value))


def _group_label(participants: tuple[str, ...], fallback: str) -> str:
    """Format a user-facing label for a conversation."""
    if not participants:
        return fallback
    if len(participants) == 1:
        return participants[0]
    return ", ".join(participants)


def _load_chat_participants(conn: sqlite3.Connection) -> dict[int, tuple[str, ...]]:
    """Map chat IDs to their participant handle list."""
    rows = conn.execute(
        """
        SELECT ordered.chat_id, GROUP_CONCAT(ordered.handle_id, X'1F') AS participants_raw
        FROM (
            SELECT chj.chat_id AS chat_id, h.id AS handle_id
            FROM chat_handle_join chj
            JOIN handle h ON h.ROWID = chj.handle_id
            ORDER BY chj.chat_id, h.id
        ) ordered
        GROUP BY ordered.chat_id
        """
    ).fetchall()
    chat_index: dict[int, tuple[str, ...]] = {}
    for row in rows:
        raw = row["participants_raw"] or ""
        chat_index[row["chat_id"]] = tuple(
            value for value in raw.split("\x1f") if value
        )
    return chat_index


def _build_handle_match_clause(column: str) -> str:
    """Build a SQL clause that matches a handle as exact string or normalized phone."""
    normalized = (
        f"REPLACE(REPLACE(REPLACE(REPLACE(REPLACE({column}, '+', ''), '-', ''), '(', ''), ')', ''), ' ', '')"
    )
    return f"({column} = ? OR LOWER({column}) = LOWER(?) OR {normalized} = ?)"


def _parse_participants_arg(
    contact: str | None,
    participants: Iterable[str] | None,
) -> tuple[str, ...] | None:
    if contact and participants:
        raise ValueError("Use either 'contact' or 'participants', not both.")
    if contact:
        return _canonicalize_handles([contact])
    if participants:
        normalized = _canonicalize_handles(participants)
        return normalized or None
    return None


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
    contact: str | None = None,
    participants: Iterable[str] | None = None,
    db_path: Path = IMESSAGE_DB,
    contact_resolver: ContactResolver | None = None,
    resolve_contacts: bool = True,
) -> Generator[RawMessage, None, None]:
    """Stream messages from chat.db, optionally filtered by date.

    Prefers the `text` column when available, otherwise decodes the
    `attributedBody` blob. Yields RawMessage objects sorted by conversation
    thread and date, suitable for downstream chunking by conversation window.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        resolver = (
            contact_resolver
            if contact_resolver is not None
            else get_default_resolver()
            if resolve_contacts
            else ContactResolver.empty()
        )
        participant_values = list(participants or [])
        if contact and participant_values:
            raise ValueError("Use either 'contact' or 'participants', not both.")
        target_participants = _parse_participants_arg(None, participant_values)
        contact_match_values: tuple[str, ...] | None = None
        if contact:
            contact_match_values = _canonicalize_handles(
                [contact, *resolver.handles_for_contact(contact)]
            )
        matching_chat_ids: set[int] | None = None

        if contact_match_values is not None:
            chat_participants = _load_chat_participants(conn)
            matching_chat_ids = {
                chat_id
                for chat_id, handles in chat_participants.items()
                if (canonical := _canonicalize_handles(handles))
                and len(canonical) == 1
                and canonical[0] in contact_match_values
            }
            if not contact_match_values:
                return
        elif target_participants is not None:
            chat_participants = _load_chat_participants(conn)
            matching_chat_ids = {
                chat_id
                for chat_id, handles in chat_participants.items()
                if _canonicalize_handles(handles) == target_participants
            }
            if len(target_participants) > 1 and not matching_chat_ids:
                return

        query = """
            WITH message_chat AS (
                SELECT message_id, MIN(chat_id) AS chat_id
                FROM chat_message_join
                GROUP BY message_id
            ),
            chat_participants AS (
                SELECT ordered.chat_id, GROUP_CONCAT(ordered.handle_id, X'1F') AS participants_raw
                FROM (
                    SELECT chj.chat_id AS chat_id, h.id AS handle_id
                    FROM chat_handle_join chj
                    JOIN handle h ON h.ROWID = chj.handle_id
                    ORDER BY chj.chat_id, h.id
                ) ordered
                GROUP BY ordered.chat_id
            )
            SELECT
                m.ROWID   AS rowid,
                m.text    AS text,
                m.attributedBody AS attributed_body,
                m.date    AS date,
                m.is_from_me AS is_from_me,
                mc.chat_id AS chat_id,
                cp.participants_raw AS participants_raw,
                COALESCE(h.id, 'unknown') AS sender_handle,
                COALESCE(
                    printf('chat:%d', mc.chat_id),
                    'handle:' || COALESCE(h.id, 'unknown')
                ) AS conversation_sort
            FROM message m
            LEFT JOIN message_chat mc ON mc.message_id = m.ROWID
            LEFT JOIN chat_participants cp ON cp.chat_id = mc.chat_id
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE ((m.text IS NOT NULL AND m.text != '')
               OR m.attributedBody IS NOT NULL)
        """
        params: list = []

        if since is not None:
            apple_cutoff = datetime_to_apple_ts(since)
            query += " AND m.date >= ?"
            params.append(apple_cutoff)

        if contact_match_values is not None:
            filter_clauses: list[str] = []
            if matching_chat_ids:
                placeholders = ",".join("?" for _ in matching_chat_ids)
                filter_clauses.append(f"mc.chat_id IN ({placeholders})")
                params.extend(sorted(matching_chat_ids))

            direct_handle_clauses: list[str] = []
            for value in contact_match_values:
                direct_handle_clauses.append(
                    f"(mc.chat_id IS NULL AND {_build_handle_match_clause('h.id')})"
                )
                params.extend([value, value, value])
            if direct_handle_clauses:
                filter_clauses.append("(" + " OR ".join(direct_handle_clauses) + ")")

            if not filter_clauses:
                return
            query += " AND (" + " OR ".join(filter_clauses) + ")"
        elif target_participants is not None:
            filter_clauses: list[str] = []
            if matching_chat_ids:
                placeholders = ",".join("?" for _ in matching_chat_ids)
                filter_clauses.append(f"mc.chat_id IN ({placeholders})")
                params.extend(sorted(matching_chat_ids))

            if len(target_participants) == 1 and participant_values:
                filter_clauses.append(
                    f"(mc.chat_id IS NULL AND {_build_handle_match_clause('h.id')})"
                )
                raw_single_target = participant_values[0]
                params.extend(
                    [raw_single_target, raw_single_target, target_participants[0]]
                )

            if not filter_clauses:
                return
            query += " AND (" + " OR ".join(filter_clauses) + ")"

        query += " ORDER BY conversation_sort, m.date"

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

                participant_tuple = ()
                if row["participants_raw"]:
                    participant_tuple = tuple(
                        value
                        for value in str(row["participants_raw"]).split("\x1f")
                        if value
                    )

                sender_handle = row["sender_handle"]
                sender_label = (
                    resolver.label_for_handle(sender_handle)
                    if sender_handle != "unknown"
                    else sender_handle
                )
                if participant_tuple:
                    conversation_id = f"chat:{row['chat_id']}"
                    display_participants = resolver.label_for_participants(
                        participant_tuple
                    )
                    contact_label = _group_label(display_participants, sender_label)
                else:
                    display_participants = ()
                    contact_label = sender_label
                    conversation_id = (
                        f"handle:{_normalize_handle(sender_handle)}"
                        if sender_handle != "unknown"
                        else f"message:{row['rowid']}"
                    )
                    if sender_handle != "unknown":
                        display_participants = (sender_label,)

                yield RawMessage(
                    rowid=row["rowid"],
                    text=text,
                    date=apple_ts_to_datetime(row["date"]),
                    is_from_me=bool(row["is_from_me"]),
                    contact=contact_label,
                    sender=sender_label,
                    conversation_id=conversation_id,
                    participants=display_participants,
                )
    finally:
        conn.close()
