"""SQLite-backed vector database with numpy cosine similarity search."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from src.chunker import Chunk
from src.config import VECTOR_DB

EMBEDDING_DIM = 768  # nomic-embed-text


def _ensure_db(db_path: Path = VECTOR_DB) -> sqlite3.Connection:
    """Create the DB and table if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            contact TEXT,
            start_time REAL,
            end_time REAL,
            text TEXT NOT NULL,
            message_count INTEGER,
            embedding BLOB,
            created_at REAL DEFAULT (unixepoch())
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedup
        ON chunks(source, contact, start_time)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_contact ON chunks(contact)
    """)
    # Migration: add metadata column for existing DBs
    try:
        conn.execute("ALTER TABLE chunks ADD COLUMN metadata TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    return conn


def insert_chunk(chunk: Chunk, embedding: list[float], db_path: Path = VECTOR_DB) -> int:
    """Insert a chunk with its embedding. Returns the row ID."""
    conn = _ensure_db(db_path)
    try:
        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
        meta_json = json.dumps(chunk.metadata) if chunk.metadata else None
        cursor = conn.execute(
            """
            INSERT INTO chunks (source, contact, start_time, end_time, text, message_count, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, contact, start_time) DO UPDATE SET
                end_time = excluded.end_time,
                text = excluded.text,
                message_count = excluded.message_count,
                embedding = excluded.embedding,
                metadata = excluded.metadata,
                created_at = unixepoch()
            """,
            (
                chunk.source,
                chunk.contact,
                chunk.start_time.timestamp(),
                chunk.end_time.timestamp(),
                chunk.text,
                chunk.message_count,
                emb_blob,
                meta_json,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def search(
    query_embedding: list[float],
    top_k: int = 5,
    source: str | None = None,
    db_path: Path = VECTOR_DB,
) -> list[dict]:
    """Find the top-k most similar chunks by cosine similarity."""
    top_k = max(1, min(top_k, 50))
    conn = _ensure_db(db_path)
    try:
        where = "WHERE embedding IS NOT NULL"
        params: list = []
        if source:
            where += " AND source = ?"
            params.append(source)

        rows = conn.execute(
            f"SELECT id, source, contact, start_time, end_time, text, message_count, embedding, metadata FROM chunks {where}",
            params,
        ).fetchall()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scored = []
        for row in rows:
            emb = np.frombuffer(row[7], dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            similarity = float(np.dot(query_vec, emb) / (query_norm * emb_norm))
            scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, row in scored[:top_k]:
            results.append({
                "id": row[0],
                "source": row[1],
                "contact": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "text": row[5],
                "message_count": row[6],
                "similarity": sim,
                "metadata": json.loads(row[8]) if row[8] else {},
            })
        return results
    finally:
        conn.close()


def fetch_by_ids(chunk_ids: list[int], db_path: Path = VECTOR_DB) -> list[dict]:
    """Fetch chunks by their row IDs. Returns them in the same dict format as search()."""
    if not chunk_ids:
        return []
    conn = _ensure_db(db_path)
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT id, source, contact, start_time, end_time, text, message_count, metadata "
            f"FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return [
            {
                "id": r[0],
                "source": r[1],
                "contact": r[2],
                "start_time": r[3],
                "end_time": r[4],
                "text": r[5],
                "message_count": r[6],
                "similarity": 0.0,  # not from a search, no score
                "metadata": json.loads(r[7]) if r[7] else {},
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_stats(db_path: Path = VECTOR_DB) -> dict:
    """Return basic stats about the vector DB."""
    if not db_path.exists():
        return {"total_chunks": 0, "by_source": {}, "db_size_mb": 0}

    conn = _ensure_db(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sources = conn.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source"
        ).fetchall()
        db_size = db_path.stat().st_size / (1024 * 1024)
        return {
            "total_chunks": total,
            "by_source": {s: c for s, c in sources},
            "db_size_mb": round(db_size, 2),
        }
    finally:
        conn.close()
