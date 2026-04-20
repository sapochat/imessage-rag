"""Tests for embedding text cleaning logic."""

from src.embed import _FALLBACK_CHAR_LIMITS, _MAX_CHARS, _candidate_prompts, _clean


class TestClean:
    def test_strips_attachment_placeholder(self):
        text = "Check out this photo \ufffc and this one \ufffc"
        result = _clean(text)
        assert "\ufffc" not in result
        assert "Check out this photo  and this one" == result

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

    def test_strips_control_chars(self):
        text = "hello\x00world\x07\nok"
        assert _clean(text) == "helloworld\nok"

    def test_collapses_excess_newlines(self):
        text = "a\n\n\n\nb"
        assert _clean(text) == "a\n\nb"


class TestCandidatePrompts:
    def test_builds_progressively_shorter_unique_prompts(self):
        text = "A" * (_MAX_CHARS + 1000)
        prompts = _candidate_prompts(text)
        assert len(prompts) == 1 + len(_FALLBACK_CHAR_LIMITS)
        assert len(prompts[0]) == _MAX_CHARS
        assert len(prompts[-1]) == _FALLBACK_CHAR_LIMITS[-1]

    def test_skips_empty_candidates(self):
        assert _candidate_prompts("\ufffc\x00") == []
