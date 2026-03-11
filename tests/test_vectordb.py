"""Tests for the SQLite vector database."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.vectordb import (
    EMBEDDING_DIM,
    _ensure_db,
    fetch_by_ids,
    get_stats,
    insert_chunk,
    search,
)
from tests.conftest import make_chunk


def _random_embedding(dim=EMBEDDING_DIM, seed=None):
    """Generate a random unit-norm embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


class TestEnsureDb:
    def test_creates_db_file(self, vector_db):
        conn = _ensure_db(vector_db)
        conn.close()
        assert vector_db.exists()

    def test_creates_chunks_table(self, vector_db):
        conn = _ensure_db(vector_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "chunks" in table_names

    def test_idempotent(self, vector_db):
        """Calling _ensure_db twice doesn't error."""
        conn1 = _ensure_db(vector_db)
        conn1.close()
        conn2 = _ensure_db(vector_db)
        conn2.close()


class TestInsertAndSearch:
    def test_insert_and_retrieve(self, vector_db):
        chunk = make_chunk(text="Test message about pizza")
        emb = _random_embedding(seed=42)
        row_id = insert_chunk(chunk, emb, db_path=vector_db)
        assert row_id > 0

        results = search(emb, top_k=5, db_path=vector_db)
        assert len(results) == 1
        assert results[0]["text"] == "Test message about pizza"
        assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_upsert_on_conflict(self, vector_db):
        """Inserting a chunk with the same (source, contact, start_time) updates it."""
        now = datetime.now(tz=timezone.utc)
        chunk1 = make_chunk(text="Version 1", start_time=now)
        chunk2 = make_chunk(text="Version 2", start_time=now)
        emb = _random_embedding(seed=1)

        insert_chunk(chunk1, emb, db_path=vector_db)
        insert_chunk(chunk2, emb, db_path=vector_db)

        results = search(emb, top_k=10, db_path=vector_db)
        assert len(results) == 1
        assert results[0]["text"] == "Version 2"

    def test_search_top_k_ordering(self, vector_db):
        """Search returns results ordered by similarity (most similar first)."""
        # Insert 3 chunks with known embeddings
        base = _random_embedding(seed=100)

        for i in range(3):
            chunk = make_chunk(
                text=f"Chunk {i}",
                start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            # Make each embedding slightly less similar to the query
            emb = base.copy()
            emb[0] += i * 0.5  # perturb
            # Re-normalize
            norm = np.linalg.norm(emb)
            emb = [x / norm for x in emb]
            insert_chunk(chunk, emb, db_path=vector_db)

        results = search(base, top_k=3, db_path=vector_db)
        assert len(results) == 3
        # Similarities should be in descending order
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_search_source_filter(self, vector_db):
        """Source filter restricts results."""
        emb = _random_embedding(seed=50)

        c1 = make_chunk(source="imessage", text="iMessage text",
                        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
        c2 = make_chunk(source="email", text="Email text",
                        start_time=datetime(2024, 1, 2, tzinfo=timezone.utc))

        insert_chunk(c1, emb, db_path=vector_db)
        insert_chunk(c2, emb, db_path=vector_db)

        results = search(emb, top_k=10, source="email", db_path=vector_db)
        assert len(results) == 1
        assert results[0]["source"] == "email"

    def test_search_empty_db(self, vector_db):
        emb = _random_embedding()
        results = search(emb, top_k=5, db_path=vector_db)
        assert results == []

    def test_search_zero_vector(self, vector_db):
        """A zero query vector returns no results."""
        chunk = make_chunk()
        insert_chunk(chunk, _random_embedding(seed=1), db_path=vector_db)

        zero_emb = [0.0] * EMBEDDING_DIM
        results = search(zero_emb, top_k=5, db_path=vector_db)
        assert results == []

    def test_top_k_clamped(self, vector_db):
        """top_k is clamped between 1 and 50."""
        emb = _random_embedding(seed=1)
        insert_chunk(make_chunk(), emb, db_path=vector_db)

        # top_k=0 gets clamped to 1
        results = search(emb, top_k=0, db_path=vector_db)
        assert len(results) == 1

    def test_dimension_mismatch_skipped(self, vector_db):
        """Chunks with wrong embedding dimension are silently skipped."""
        chunk = make_chunk()
        wrong_dim_emb = [1.0] * 100  # wrong dimension
        insert_chunk(chunk, wrong_dim_emb, db_path=vector_db)

        query = _random_embedding()  # correct dimension
        results = search(query, top_k=5, db_path=vector_db)
        assert results == []

    def test_metadata_roundtrip(self, vector_db):
        """Metadata dict survives insert→search."""
        chunk = make_chunk(metadata={"message_id": "<abc@test.com>"})
        emb = _random_embedding(seed=99)
        insert_chunk(chunk, emb, db_path=vector_db)

        results = search(emb, top_k=1, db_path=vector_db)
        assert results[0]["metadata"] == {"message_id": "<abc@test.com>"}


class TestFetchByIds:
    def test_fetch_existing(self, vector_db):
        chunk = make_chunk(text="Fetchable")
        emb = _random_embedding(seed=1)
        row_id = insert_chunk(chunk, emb, db_path=vector_db)

        results = fetch_by_ids([row_id], db_path=vector_db)
        assert len(results) == 1
        assert results[0]["text"] == "Fetchable"
        assert results[0]["similarity"] == 0.0  # not from search

    def test_fetch_empty_list(self, vector_db):
        assert fetch_by_ids([], db_path=vector_db) == []

    def test_fetch_missing_id(self, vector_db):
        results = fetch_by_ids([9999], db_path=vector_db)
        assert results == []


class TestGetStats:
    def test_empty_db(self, vector_db):
        stats = get_stats(vector_db)
        assert stats["total_chunks"] == 0

    def test_nonexistent_db(self, tmp_path):
        stats = get_stats(tmp_path / "nope.db")
        assert stats == {"total_chunks": 0, "by_source": {}, "db_size_mb": 0}

    def test_counts_by_source(self, vector_db):
        emb = _random_embedding(seed=1)
        for i in range(3):
            insert_chunk(
                make_chunk(source="imessage",
                          start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc)),
                emb, db_path=vector_db,
            )
        for i in range(2):
            insert_chunk(
                make_chunk(source="email",
                          start_time=datetime(2024, 2, i + 1, tzinfo=timezone.utc)),
                emb, db_path=vector_db,
            )

        stats = get_stats(vector_db)
        assert stats["total_chunks"] == 5
        assert stats["by_source"]["imessage"] == 3
        assert stats["by_source"]["email"] == 2
        assert stats["db_size_mb"] > 0
