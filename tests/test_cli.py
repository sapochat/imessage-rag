"""Tests for CLI argument parsing and helpers."""

from datetime import datetime, timedelta, timezone

import pytest

from cli import parse_since


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
