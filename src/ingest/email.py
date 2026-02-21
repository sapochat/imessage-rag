"""Streaming extraction of emails from Apple Mail .emlx files."""

import email
import email.message
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Generator

from bs4 import BeautifulSoup

from src.config import MAIL_DIR

logger = logging.getLogger(__name__)

# Folder names to include (case-insensitive substring match on .mbox dir name)
_ALLOWED_FOLDERS = {"inbox", "sent", "archive", "all mail"}
# Folder names to always exclude
_BLOCKED_FOLDERS = {"spam", "junk", "trash", "drafts", "deleted"}


@dataclass
class RawEmail:
    filepath: str
    subject: str
    sender: str
    recipients: str
    date: datetime
    body: str
    message_id: str


def _parse_emlx(path: Path) -> RawEmail | None:
    """Parse a single .emlx file into a RawEmail.

    .emlx format: line 1 is a byte count, followed by that many bytes of
    RFC822 content, then an Apple plist trailer.
    """
    try:
        raw = path.read_bytes()
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
        return None

    # First line is the byte count
    newline_idx = raw.find(b"\n")
    if newline_idx == -1:
        logger.warning("No newline in %s — not a valid .emlx", path)
        return None

    try:
        byte_count = int(raw[:newline_idx].strip())
    except ValueError:
        logger.warning("Invalid byte count in %s", path)
        return None

    rfc822_start = newline_idx + 1
    rfc822_bytes = raw[rfc822_start : rfc822_start + byte_count]

    try:
        msg = email.message_from_bytes(rfc822_bytes)
    except Exception as e:
        logger.warning("Failed to parse email in %s: %s", path, e)
        return None

    # Extract text body
    body = _extract_body(msg)
    if not body or not body.strip():
        return None

    # Parse date
    date_str = msg.get("Date", "")
    try:
        date = parsedate_to_datetime(date_str)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
    except Exception:
        date = datetime.now(tz=timezone.utc)

    return RawEmail(
        filepath=str(path),
        subject=msg.get("Subject", "") or "",
        sender=msg.get("From", "") or "",
        recipients=msg.get("To", "") or "",
        date=date,
        body=body.strip(),
        message_id=msg.get("Message-ID", "") or "",
    )


def _decode_payload(payload: bytes, charset: str | None) -> str:
    """Decode payload bytes, falling back to utf-8 for invalid charsets."""
    charset = charset or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


def _extract_body(msg: email.message.Message) -> str | None:
    """Extract plain text from an email message.

    Prefers text/plain parts. Falls back to text/html stripped via BeautifulSoup.
    """
    if not msg.is_multipart():
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload is None:
            return None
        text = _decode_payload(payload, msg.get_content_charset())
        if content_type == "text/plain":
            return text
        if content_type == "text/html":
            return BeautifulSoup(text, "html.parser").get_text(separator="\n")
        return None

    # Multipart: collect text/plain parts first, fall back to text/html
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        text = _decode_payload(payload, part.get_content_charset())
        if content_type == "text/plain":
            plain_parts.append(text)
        else:
            html_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        return "\n".join(
            BeautifulSoup(h, "html.parser").get_text(separator="\n")
            for h in html_parts
        )
    return None


def _allowed_folder(mbox_name: str) -> bool:
    """Check if a .mbox folder name matches our inclusion criteria."""
    name_lower = mbox_name.lower()
    # Exclude blocked folders first
    for blocked in _BLOCKED_FOLDERS:
        if blocked in name_lower:
            return False
    # Include only allowed folders
    for allowed in _ALLOWED_FOLDERS:
        if allowed in name_lower:
            return True
    return False


def _find_allowed_mboxes(mail_dir: Path) -> list[Path]:
    """Find all .mbox directories that pass the folder filter."""
    allowed: list[Path] = []
    for mbox in mail_dir.rglob("*.mbox"):
        if mbox.is_dir() and _allowed_folder(mbox.name):
            allowed.append(mbox)
    return allowed


def extract_emails(
    since: datetime | None = None,
    mail_dir: Path = MAIL_DIR,
) -> Generator[RawEmail, None, None]:
    """Stream parsed emails from Apple Mail .emlx files.

    Finds allowed .mbox directories first, then walks only those for .emlx
    files. Uses file mtime as a cheap pre-filter before full parsing when
    a `since` cutoff is provided.
    """
    if not mail_dir.exists():
        logger.warning("Mail directory does not exist: %s", mail_dir)
        return

    since_ts = since.timestamp() if since else None
    allowed_mboxes = _find_allowed_mboxes(mail_dir)
    logger.info("Found %d allowed mailboxes", len(allowed_mboxes))

    for mbox_dir in allowed_mboxes:
        for emlx_path in mbox_dir.rglob("*.emlx"):
            # Quick mtime check before expensive parsing
            if since_ts is not None:
                try:
                    if emlx_path.stat().st_mtime < since_ts:
                        continue
                except OSError:
                    continue

            raw_email = _parse_emlx(emlx_path)
            if raw_email is None:
                continue

            if since is not None and raw_email.date < since:
                continue

            yield raw_email
