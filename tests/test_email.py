"""Tests for Apple Mail .emlx parsing."""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingest.email import (
    RawEmail,
    _allowed_folder,
    _extract_body,
    _parse_emlx,
    extract_emails,
)


def _write_emlx(path: Path, rfc822_content: str) -> None:
    """Write a valid .emlx file (byte-count line + RFC822 content)."""
    content_bytes = rfc822_content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"{len(content_bytes)}\n".encode())
        f.write(content_bytes)


SIMPLE_EMAIL = """\
From: sender@example.com
To: recipient@example.com
Subject: Test Subject
Date: Mon, 15 Jan 2024 12:00:00 +0000
Message-ID: <test-123@example.com>

This is the email body."""

HTML_EMAIL = """\
From: sender@example.com
To: recipient@example.com
Subject: HTML Test
Date: Mon, 15 Jan 2024 12:00:00 +0000
Content-Type: text/html

<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>"""

MULTIPART_EMAIL = """\
From: sender@example.com
To: recipient@example.com
Subject: Multipart Test
Date: Mon, 15 Jan 2024 12:00:00 +0000
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain

Plain text version.
--boundary123
Content-Type: text/html

<html><body><p>HTML version.</p></body></html>
--boundary123--"""


class TestParseEmlx:
    def test_simple_email(self, tmp_path):
        emlx = tmp_path / "test.emlx"
        _write_emlx(emlx, SIMPLE_EMAIL)

        result = _parse_emlx(emlx)
        assert result is not None
        assert result.subject == "Test Subject"
        assert result.sender == "sender@example.com"
        assert result.recipients == "recipient@example.com"
        assert result.message_id == "<test-123@example.com>"
        assert "email body" in result.body

    def test_html_email_stripped(self, tmp_path):
        emlx = tmp_path / "html.emlx"
        _write_emlx(emlx, HTML_EMAIL)

        result = _parse_emlx(emlx)
        assert result is not None
        assert "Title" in result.body
        assert "Paragraph text." in result.body
        assert "<html>" not in result.body

    def test_multipart_prefers_plain(self, tmp_path):
        emlx = tmp_path / "multi.emlx"
        _write_emlx(emlx, MULTIPART_EMAIL)

        result = _parse_emlx(emlx)
        assert result is not None
        assert "Plain text version." in result.body

    def test_invalid_byte_count(self, tmp_path):
        emlx = tmp_path / "bad.emlx"
        emlx.write_text("not_a_number\nsome content")

        result = _parse_emlx(emlx)
        assert result is None

    def test_no_newline(self, tmp_path):
        emlx = tmp_path / "nonewline.emlx"
        emlx.write_bytes(b"just bytes no newline")

        result = _parse_emlx(emlx)
        assert result is None

    def test_empty_body_returns_none(self, tmp_path):
        email_content = "From: a@b.com\nTo: c@d.com\nSubject: Empty\nDate: Mon, 15 Jan 2024 12:00:00 +0000\n\n"
        emlx = tmp_path / "empty.emlx"
        _write_emlx(emlx, email_content)

        result = _parse_emlx(emlx)
        assert result is None

    def test_missing_date_defaults_to_now(self, tmp_path):
        email_content = "From: a@b.com\nTo: c@d.com\nSubject: No Date\n\nBody here."
        emlx = tmp_path / "nodate.emlx"
        _write_emlx(emlx, email_content)

        result = _parse_emlx(emlx)
        assert result is not None
        # Should default to roughly now
        assert result.date.year >= 2024

    def test_nonexistent_file(self, tmp_path):
        result = _parse_emlx(tmp_path / "missing.emlx")
        assert result is None


class TestAllowedFolder:
    @pytest.mark.parametrize("name,expected", [
        ("INBOX.mbox", True),
        ("Sent Messages.mbox", True),
        ("Archive.mbox", True),
        ("All Mail.mbox", True),
        ("Spam.mbox", False),
        ("Junk.mbox", False),
        ("Trash.mbox", False),
        ("Drafts.mbox", False),
        ("Deleted Messages.mbox", False),
        ("Custom Folder.mbox", False),  # not in allowed list
    ])
    def test_folder_filtering(self, name, expected):
        assert _allowed_folder(name) is expected

    def test_blocked_takes_priority(self):
        """A folder matching both allowed and blocked should be blocked."""
        # "Inbox Spam Archive" contains both "inbox" and "spam"
        assert _allowed_folder("Inbox Spam Archive.mbox") is False


class TestExtractEmails:
    def test_nonexistent_dir_yields_nothing(self, tmp_path):
        result = list(extract_emails(mail_dir=tmp_path / "nope"))
        assert result == []

    def test_finds_emails_in_allowed_mbox(self, tmp_path):
        """Emails in an INBOX.mbox are discovered and parsed."""
        inbox = tmp_path / "INBOX.mbox" / "Messages"
        inbox.mkdir(parents=True)
        _write_emlx(inbox / "1.emlx", SIMPLE_EMAIL)

        result = list(extract_emails(mail_dir=tmp_path))
        assert len(result) == 1
        assert result[0].subject == "Test Subject"

    def test_skips_blocked_folders(self, tmp_path):
        """Emails in Spam.mbox are not extracted."""
        spam = tmp_path / "Spam.mbox" / "Messages"
        spam.mkdir(parents=True)
        _write_emlx(spam / "1.emlx", SIMPLE_EMAIL)

        result = list(extract_emails(mail_dir=tmp_path))
        assert result == []

    def test_since_filter(self, tmp_path):
        inbox = tmp_path / "INBOX.mbox" / "Messages"
        inbox.mkdir(parents=True)
        _write_emlx(inbox / "1.emlx", SIMPLE_EMAIL)

        # Email date is 2024-01-15, so filtering since 2025 should exclude it
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = list(extract_emails(since=cutoff, mail_dir=tmp_path))
        assert result == []
