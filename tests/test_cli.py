"""Tests for CLI argument parsing and helpers."""

from datetime import datetime, timedelta, timezone

import pytest

from imessage_rag.cli import _imessage_db_readable, _print_kv, parse_participants, parse_since


class TestParseSince:
    def test_days(self):
        result = parse_since("30d")
        expected = datetime.now(tz=timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_hours(self):
        result = parse_since("24h")
        expected = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        assert abs((result - expected).total_seconds()) < 2

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            parse_since("30m")

    def test_invalid_number_raises(self):
        with pytest.raises(ValueError):
            parse_since("abcd")


class TestParseParticipants:
    def test_comma_separated(self):
        assert parse_participants("+15551234567, +15557654321") == [
            "+15551234567",
            "+15557654321",
        ]

    def test_ignores_empty_values(self):
        assert parse_participants("a,, b ,") == ["a", "b"]


class TestPrintKv:
    def test_alignment_format(self, capsys):
        _print_kv("Vector DB", "/tmp/test.db")
        assert "Vector DB" in capsys.readouterr().out


class TestImessageDbReadable:
    def test_existing_sqlite_db_is_readable(self, imessage_db):
        readable, error = _imessage_db_readable(imessage_db)

        assert readable is True
        assert error is None

    def test_missing_sqlite_db_is_not_readable(self, tmp_path):
        readable, error = _imessage_db_readable(tmp_path / "missing.db")

        assert readable is False
        assert error is not None
