"""SQLite-backed vector database with numpy cosine similarity search."""

import heapq
import json
import sqlite3
from pathlib import Path

import numpy as np

from imessage_rag.chunker import Chunk
from imessage_rag.config import EMBED_DIMENSIONS, VECTOR_DB

EMBEDDING_DIM = EMBED_DIMENSIONS or 768

_UPSERT_CHUNK_SQL = """
    INSERT INTO chunks (thread_key, contact, start_time, end_time, text, message_count, embedding, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(thread_key, start_time) DO UPDATE SET
        contact = excluded.contact,
        end_time = excluded.end_time,
        text = excluded.text,
        message_count = excluded.message_count,
        embedding = excluded.embedding,
        metadata = excluded.metadata,
        created_at = unixepoch()
"""


def _ensure_db(db_path: Path = VECTOR_DB) -> sqlite3.Connection:
    """Create the DB and table if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_key    TEXT    NOT NULL,
            contact       TEXT,
            start_time    REAL    NOT NULL,
            end_time      REAL    NOT NULL,
            text          TEXT    NOT NULL,
            message_count INTEGER NOT NULL,
            embedding     BLOB    NOT NULL,
            metadata      TEXT,
            created_at    REAL    DEFAULT (unixepoch())
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedup "
        "ON chunks(thread_key, start_time)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_contact ON chunks(contact)"
    )
    conn.commit()
    return conn


def insert_chunk(chunk: Chunk, embedding: list[float], db_path: Path = VECTOR_DB) -> int:
    """Insert a chunk with its embedding. Returns the row ID."""
    conn = _ensure_db(db_path)
    try:
        cursor = conn.execute(_UPSERT_CHUNK_SQL, _chunk_row(chunk, embedding))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _chunk_row(chunk: Chunk, embedding: list[float]) -> tuple:
    emb_blob = np.array(embedding, dtype=np.float32).tobytes()
    meta_json = json.dumps(chunk.metadata) if chunk.metadata else None
    return (
        chunk.thread_key,
        chunk.contact,
        chunk.start_time.timestamp(),
        chunk.end_time.timestamp(),
        chunk.text,
        chunk.message_count,
        emb_blob,
        meta_json,
    )


def insert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    db_path: Path = VECTOR_DB,
) -> int:
    """Bulk upsert chunks and embeddings in one SQLite transaction."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        return 0

    conn = _ensure_db(db_path)
    try:
        rows = [_chunk_row(chunk, embedding) for chunk, embedding in zip(chunks, embeddings)]
        conn.executemany(_UPSERT_CHUNK_SQL, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def filter_new_chunks(chunks: list[Chunk], db_path: Path = VECTOR_DB) -> list[Chunk]:
    """Return only chunks whose dedupe key is not already present."""
    if not chunks or not db_path.exists():
        return chunks

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        keys = [(chunk.thread_key, chunk.start_time.timestamp()) for chunk in chunks]
        values_sql = ",".join("(?, ?)" for _ in keys)
        params = [value for key in keys for value in key]
        rows = conn.execute(
            f"""
            WITH incoming(thread_key, start_time) AS (VALUES {values_sql})
            SELECT chunks.thread_key, chunks.start_time
            FROM chunks
            JOIN incoming
              ON incoming.thread_key = chunks.thread_key
             AND incoming.start_time = chunks.start_time
            """,
            params,
        ).fetchall()
        existing = {(row[0], row[1]) for row in rows}
        return [
            chunk
            for chunk, key in zip(chunks, keys, strict=False)
            if key not in existing
        ]
    except sqlite3.OperationalError:
        return chunks
    finally:
        conn.close()


def search(
    query_embedding: list[float],
    top_k: int = 5,
    db_path: Path = VECTOR_DB,
) -> list[dict]:
    """Find the top-k most similar chunks by cosine similarity."""
    top_k = max(1, min(top_k, 50))
    if not db_path.exists():
        return []

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT id, contact, thread_key, start_time, end_time, text, message_count, embedding, metadata "
            "FROM chunks WHERE embedding IS NOT NULL"
        )

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        top: list[tuple[float, int, tuple]] = []

        while True:
            rows = cursor.fetchmany(4096)
            if not rows:
                break

            vectors = []
            valid_rows = []
            for row in rows:
                emb = np.frombuffer(row[7], dtype=np.float32)
                if emb.shape == query_vec.shape:
                    vectors.append(emb)
                    valid_rows.append(row)
            if not vectors:
                continue

            matrix = np.vstack(vectors)
            norms = np.linalg.norm(matrix, axis=1)
            valid_mask = norms > 0
            if not np.any(valid_mask):
                continue

            similarities = matrix[valid_mask] @ query_vec
            similarities = similarities / (norms[valid_mask] * query_norm)
            masked_rows = [
                row for row, keep in zip(valid_rows, valid_mask, strict=False) if keep
            ]

            for sim, row in zip(similarities, masked_rows, strict=False):
                item = (float(sim), int(row[0]), row)
                if len(top) < top_k:
                    heapq.heappush(top, item)
                elif item[0] > top[0][0]:
                    heapq.heapreplace(top, item)

        scored = sorted(top, key=lambda x: x[0], reverse=True)

        return [
            {
                "id": row[0],
                "contact": row[1],
                "thread_key": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "text": row[5],
                "message_count": row[6],
                "similarity": sim,
                "metadata": json.loads(row[8]) if row[8] else {},
            }
            for sim, _, row in scored
        ]
    finally:
        conn.close()


def fetch_by_ids(chunk_ids: list[int], db_path: Path = VECTOR_DB) -> list[dict]:
    """Fetch chunks by their row IDs. Returns them in the same dict format as search()."""
    if not chunk_ids:
        return []
    if not db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT id, contact, thread_key, start_time, end_time, text, message_count, metadata "
            f"FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return [
            {
                "id": r[0],
                "contact": r[1],
                "thread_key": r[2],
                "start_time": r[3],
                "end_time": r[4],
                "text": r[5],
                "message_count": r[6],
                "similarity": 0.0,
                "metadata": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_stats(db_path: Path = VECTOR_DB) -> dict:
    """Return basic stats about the vector DB."""
    if not db_path.exists():
        return {"total_chunks": 0, "db_size_mb": 0}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        try:
            total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        except sqlite3.OperationalError:
            total = 0
        db_size = db_path.stat().st_size / (1024 * 1024)
        return {
            "total_chunks": total,
            "db_size_mb": round(db_size, 2),
        }
    finally:
        conn.close()
