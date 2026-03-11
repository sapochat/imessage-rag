"""Tests for embedding text cleaning logic."""

from src.embed import _clean, _MAX_CHARS


class TestClean:
    def test_strips_attachment_placeholder(self):
        text = "Check out this photo \ufffc and this one \ufffc"
        result = _clean(text)
        assert "\ufffc" not in result
        assert "Check out this photo  and this one " == result

    def test_truncates_long_text(self):
        text = "A" * (_MAX_CHARS + 1000)
        result = _clean(text)
        assert len(result) == _MAX_CHARS

    def test_short_text_unchanged(self):
        text = "Hello world"
        assert _clean(text) == text

    def test_empty_string(self):
        assert _clean("") == ""

    def test_only_placeholders(self):
        assert _clean("\ufffc\ufffc\ufffc") == ""
