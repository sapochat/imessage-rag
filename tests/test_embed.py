"""Tests for embedding text cleaning and batching logic."""

import pytest

from imessage_rag.embed import (
    EmbeddingConfigError,
    EmbeddingInputTooLong,
    _FALLBACK_CHAR_LIMITS,
    _MAX_CHARS,
    _candidate_prompts,
    _clean,
    get_embedding,
    get_embeddings,
)


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


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", reason="error"):
        self.status_code = status_code
        self._payload = payload or {}
        self.url = "http://localhost:11434/api/embed"
        self.text = text
        self.reason = reason

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("raise_for_status should not be called")


class TestGetEmbeddings:
    def test_empty_batch_returns_empty_list(self):
        assert get_embeddings([]) == []

    def test_posts_batch_and_returns_embeddings(self, monkeypatch):
        captured = {}

        def fake_post(input_value):
            captured["input"] = input_value
            return _FakeResponse(
                payload={"embeddings": [[1.0, 0.0], [0.0, 1.0]]}
            )

        monkeypatch.setattr("imessage_rag.embed._post_embedding", fake_post)

        assert get_embeddings(["hello", "world"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert captured["input"] == ["hello", "world"]

    def test_rejects_embedding_count_mismatch(self, monkeypatch):
        monkeypatch.setattr(
            "imessage_rag.embed._post_embedding",
            lambda input_value: _FakeResponse(payload={"embeddings": [[1.0]]}),
        )

        with pytest.raises(ValueError, match="unexpected number"):
            get_embeddings(["hello", "world"])

    def test_model_not_found_is_config_error(self, monkeypatch):
        monkeypatch.setattr(
            "imessage_rag.embed._post_embedding",
            lambda input_value: _FakeResponse(
                status_code=404,
                payload={"error": "model not found"},
            ),
        )

        with pytest.raises(EmbeddingConfigError, match="ollama pull"):
            get_embeddings(["hello"])

    def test_context_length_batch_error_is_recoverable_type(self, monkeypatch):
        monkeypatch.setattr(
            "imessage_rag.embed._post_embedding",
            lambda input_value: _FakeResponse(
                status_code=400,
                payload={"error": "the input length exceeds the context length"},
            ),
        )

        with pytest.raises(EmbeddingInputTooLong):
            get_embeddings(["hello"])

    def test_single_embedding_retries_shorter_prompt_on_context_error(self, monkeypatch):
        calls = []

        def fake_post(input_value):
            calls.append(input_value)
            if len(calls) == 1:
                return _FakeResponse(
                    status_code=400,
                    payload={"error": "the input length exceeds the context length"},
                )
            return _FakeResponse(payload={"embeddings": [[1.0, 0.0]]})

        monkeypatch.setattr("imessage_rag.embed._post_embedding", fake_post)

        assert get_embedding("A" * (_MAX_CHARS + 1000)) == [1.0, 0.0]
        assert len(calls[1]) < len(calls[0])
