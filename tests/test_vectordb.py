"""Tests for the SQLite vector database."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from imessage_rag.vectordb import (
    EMBEDDING_DIM,
    _ensure_db,
    fetch_by_ids,
    filter_new_chunks,
    get_stats,
    insert_chunks,
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


class TestSchema:
    def test_schema_has_no_source_column(self, tmp_path: Path):
        db = tmp_path / "v.db"
        conn = _ensure_db(db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        conn.close()
        assert "source" not in cols
        assert "thread_key" in cols
        assert "contact" in cols

    def test_unique_index_is_on_thread_key_and_start_time(self, tmp_path: Path):
        db = tmp_path / "v.db"
        conn = _ensure_db(db)
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='chunks'"
        ).fetchall()
        conn.close()
        names = {r[0]: r[1] for r in rows}
        assert "idx_chunks_dedup" in names
        assert "thread_key" in names["idx_chunks_dedup"]
        assert "start_time" in names["idx_chunks_dedup"]
        assert "source" not in names["idx_chunks_dedup"]

    def test_required_columns_are_not_null(self, tmp_path: Path):
        db = tmp_path / "v.db"
        conn = _ensure_db(db)
        info = conn.execute("PRAGMA table_info(chunks)").fetchall()
        conn.close()
        notnull = {row[1]: bool(row[3]) for row in info}
        for col in ("thread_key", "start_time", "end_time", "text", "message_count", "embedding"):
            assert notnull[col], f"{col} should be NOT NULL"


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
        """Inserting a chunk with the same (thread_key, start_time) updates it."""
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
        base = _random_embedding(seed=100)

        for i in range(3):
            chunk = make_chunk(
                text=f"Chunk {i}",
                start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            emb = base.copy()
            emb[0] += i * 0.5
            norm = np.linalg.norm(emb)
            emb = [x / norm for x in emb]
            insert_chunk(chunk, emb, db_path=vector_db)

        results = search(base, top_k=3, db_path=vector_db)
        assert len(results) == 3
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

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

        results = search(emb, top_k=0, db_path=vector_db)
        assert len(results) == 1

    def test_dimension_mismatch_skipped(self, vector_db):
        """Chunks with wrong embedding dimension are silently skipped."""
        chunk = make_chunk()
        wrong_dim_emb = [1.0] * 100
        insert_chunk(chunk, wrong_dim_emb, db_path=vector_db)

        query = _random_embedding()
        results = search(query, top_k=5, db_path=vector_db)
        assert results == []

    def test_metadata_roundtrip(self, vector_db):
        """Metadata dict survives insert→search."""
        chunk = make_chunk(metadata={"message_id": "<abc@test.com>"})
        emb = _random_embedding(seed=99)
        insert_chunk(chunk, emb, db_path=vector_db)

        results = search(emb, top_k=1, db_path=vector_db)
        assert results[0]["metadata"] == {"message_id": "<abc@test.com>"}

    def test_bulk_insert_chunks(self, vector_db):
        chunks = [
            make_chunk(
                thread_key=f"bulk-{i}",
                text=f"Bulk {i}",
                start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            for i in range(3)
        ]
        embeddings = [_random_embedding(seed=i) for i in range(3)]

        inserted = insert_chunks(chunks, embeddings, db_path=vector_db)

        assert inserted == 3
        stats = get_stats(vector_db)
        assert stats["total_chunks"] == 3

    def test_bulk_insert_validates_lengths(self, vector_db):
        with pytest.raises(ValueError, match="same length"):
            insert_chunks([make_chunk()], [], db_path=vector_db)

    def test_filter_new_chunks_skips_existing_dedupe_keys(self, vector_db):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        existing = make_chunk(thread_key="thread-a", start_time=start)
        new = make_chunk(thread_key="thread-b", start_time=start)
        insert_chunk(existing, _random_embedding(seed=1), db_path=vector_db)

        assert filter_new_chunks([existing, new], db_path=vector_db) == [new]


class TestFetchByIds:
    def test_fetch_existing(self, vector_db):
        chunk = make_chunk(text="Fetchable")
        emb = _random_embedding(seed=1)
        row_id = insert_chunk(chunk, emb, db_path=vector_db)

        results = fetch_by_ids([row_id], db_path=vector_db)
        assert len(results) == 1
        assert results[0]["text"] == "Fetchable"
        assert results[0]["similarity"] == 0.0

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
        assert stats == {"total_chunks": 0, "db_size_mb": 0}

    def test_counts_all_chunks(self, vector_db):
        emb = _random_embedding(seed=1)
        for i in range(5):
            insert_chunk(
                make_chunk(
                    thread_key=f"thread-{i}",
                    start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                ),
                emb, db_path=vector_db,
            )

        stats = get_stats(vector_db)
        assert stats["total_chunks"] == 5
        assert stats["db_size_mb"] > 0

    def test_stats_does_not_write_to_existing_db(self, vector_db):
        conn = _ensure_db(vector_db)
        conn.close()
        vector_db.chmod(0o444)
        try:
            stats = get_stats(vector_db)
        finally:
            vector_db.chmod(0o644)

        assert stats["total_chunks"] == 0
