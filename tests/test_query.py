"""Tests for query formatting and prompt construction."""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.query import _build_prompt, _format_context, reformulate_query


class TestFormatContext:
    def test_single_chunk(self):
        results = [{
            "source": "imessage",
            "contact": "alice",
            "start_time": datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc).timestamp(),
            "end_time": datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc).timestamp(),
            "message_count": 5,
            "similarity": 0.95,
            "text": "Hello there",
        }]
        ctx = _format_context(results)
        assert "[Chunk 1" in ctx
        assert "imessage" in ctx
        assert "alice" in ctx
        assert "0.950" in ctx
        assert "Hello there" in ctx

    def test_multiple_chunks_separated(self):
        results = [
            {
                "source": "imessage",
                "contact": "alice",
                "start_time": 1705320000.0,
                "end_time": 1705320000.0,
                "message_count": 1,
                "similarity": 0.9,
                "text": "First",
            },
            {
                "source": "email",
                "contact": "bob@example.com",
                "start_time": 1705320000.0,
                "end_time": 1705320000.0,
                "message_count": 1,
                "similarity": 0.8,
                "text": "Second",
            },
        ]
        ctx = _format_context(results)
        assert "[Chunk 1" in ctx
        assert "[Chunk 2" in ctx
        assert "---" in ctx  # separator


class TestBuildPrompt:
    def test_contains_query_and_context(self):
        prompt = _build_prompt("What restaurant?", "Some context here")
        assert "What restaurant?" in prompt
        assert "Some context here" in prompt
        assert "ONLY use information from the excerpts" in prompt

    def test_contains_anti_hallucination_rules(self):
        prompt = _build_prompt("test", "ctx")
        assert "NEVER use your own knowledge" in prompt


class TestReformulateQuery:
    def test_no_history_returns_original(self):
        assert reformulate_query("What time?", []) == "What time?"

    @patch("src.query.generate_once")
    def test_with_history_calls_llm(self, mock_gen):
        mock_gen.return_value = "What time did Alice suggest for dinner?"
        history = [
            {"role": "user", "content": "Tell me about Alice's dinner plans"},
            {"role": "assistant", "content": "Alice mentioned dinner at 7pm"},
        ]
        result = reformulate_query("What time?", history)
        assert result == "What time did Alice suggest for dinner?"
        mock_gen.assert_called_once()

    @patch("src.query.generate_once")
    def test_llm_failure_returns_original(self, mock_gen):
        mock_gen.side_effect = Exception("connection refused")
        history = [{"role": "user", "content": "prior"}]
        result = reformulate_query("follow up", history)
        assert result == "follow up"

    @patch("src.query.generate_once")
    def test_empty_llm_response_returns_original(self, mock_gen):
        mock_gen.return_value = ""
        history = [{"role": "user", "content": "prior"}]
        result = reformulate_query("follow up", history)
        assert result == "follow up"
