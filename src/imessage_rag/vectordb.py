"""SQLite-backed vector database with numpy cosine similarity search."""

import heapq
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from imessage_rag.chunker import Chunk
from imessage_rag.config import EMBED_DIMENSIONS, VECTOR_DB

EMBEDDING_DIM = EMBED_DIMENSIONS or 768

_UPSERT_CHUNK_SQL = """
    INSERT INTO chunks (thread_key, contact, start_time, end_time, text, message_count, embedding, embedding_norm, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(thread_key, start_time) DO UPDATE SET
        contact = excluded.contact,
        end_time = excluded.end_time,
        text = excluded.text,
        message_count = excluded.message_count,
        embedding = excluded.embedding,
        embedding_norm = excluded.embedding_norm,
        metadata = excluded.metadata,
        created_at = unixepoch()
"""

_FTS_TOKEN = re.compile(r"[\w']+")


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
            embedding_norm REAL,
            metadata      TEXT,
            created_at    REAL    DEFAULT (unixepoch())
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    if "embedding_norm" not in columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN embedding_norm REAL")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedup "
        "ON chunks(thread_key, start_time)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_contact ON chunks(contact)"
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(chunk_id UNINDEXED, contact, text, participants)
        """
    )
    conn.commit()
    return conn


def insert_chunk(chunk: Chunk, embedding: list[float], db_path: Path = VECTOR_DB) -> int:
    """Insert a chunk with its embedding. Returns the row ID."""
    conn = _ensure_db(db_path)
    try:
        conn.execute(_UPSERT_CHUNK_SQL, _chunk_row(chunk, embedding))
        row_id = _chunk_id(conn, chunk)
        _sync_fts(conn, [(row_id, chunk)])
        conn.commit()
        return row_id
    finally:
        conn.close()


def _chunk_row(chunk: Chunk, embedding: list[float]) -> tuple:
    emb_array = np.array(embedding, dtype=np.float32)
    emb_blob = emb_array.tobytes()
    emb_norm = float(np.linalg.norm(emb_array))
    meta_json = json.dumps(chunk.metadata) if chunk.metadata else None
    return (
        chunk.thread_key,
        chunk.contact,
        chunk.start_time.timestamp(),
        chunk.end_time.timestamp(),
        chunk.text,
        chunk.message_count,
        emb_blob,
        emb_norm,
        meta_json,
    )


def _chunk_id(conn: sqlite3.Connection, chunk: Chunk) -> int:
    row = conn.execute(
        "SELECT id FROM chunks WHERE thread_key = ? AND start_time = ?",
        (chunk.thread_key, chunk.start_time.timestamp()),
    ).fetchone()
    if row is None:
        raise sqlite3.OperationalError("Inserted chunk row was not found")
    return int(row[0])


def _fts_participants(chunk: Chunk) -> str:
    participants = chunk.metadata.get("participants", []) if chunk.metadata else []
    if isinstance(participants, list):
        return " ".join(str(participant) for participant in participants)
    return ""


def _sync_fts(conn: sqlite3.Connection, rows: list[tuple[int, Chunk]]) -> None:
    if not rows:
        return
    conn.executemany(
        "DELETE FROM chunks_fts WHERE chunk_id = ?",
        [(row_id,) for row_id, _ in rows],
    )
    conn.executemany(
        """
        INSERT INTO chunks_fts (chunk_id, contact, text, participants)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                row_id,
                chunk.contact,
                chunk.embedding_text or chunk.text,
                _fts_participants(chunk),
            )
            for row_id, chunk in rows
        ],
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
        fts_rows = [(_chunk_id(conn, chunk), chunk) for chunk in chunks]
        _sync_fts(conn, fts_rows)
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


def _row_to_result(row, similarity: float, metadata_index: int = 8) -> dict:
    return {
        "id": row[0],
        "contact": row[1],
        "thread_key": row[2],
        "start_time": row[3],
        "end_time": row[4],
        "text": row[5],
        "message_count": row[6],
        "similarity": similarity,
        "metadata": json.loads(row[metadata_index]) if row[metadata_index] else {},
    }


def _fts_query(query: str) -> str:
    terms = []
    for token in _FTS_TOKEN.findall(query):
        token = token.strip("'")
        if len(token) < 2:
            continue
        terms.append('"' + token.replace('"', '""') + '"')
    return " OR ".join(terms[:12])


def keyword_search(
    query: str,
    top_k: int = 5,
    db_path: Path = VECTOR_DB,
) -> list[dict]:
    """Find chunks with exact-ish keyword matches via SQLite FTS5."""
    top_k = max(1, min(top_k, 50))
    if not db_path.exists():
        return []

    match_query = _fts_query(query)
    if not match_query:
        return []

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT
                c.id, c.contact, c.thread_key, c.start_time, c.end_time,
                c.text, c.message_count, c.embedding, c.metadata,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    results = []
    for index, row in enumerate(rows):
        result = _row_to_result(row, max(0.01, 0.8 - index * 0.03))
        result["retrieval"] = "keyword"
        results.append(result)
    return results


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
            "SELECT id, contact, thread_key, start_time, end_time, message_count, embedding, embedding_norm, metadata "
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
            row_norms = []
            valid_rows = []
            for row in rows:
                emb = np.frombuffer(row[6], dtype=np.float32)
                if emb.shape == query_vec.shape:
                    vectors.append(emb)
                    row_norms.append(row[7] or np.linalg.norm(emb))
                    valid_rows.append(row)
            if not vectors:
                continue

            matrix = np.vstack(vectors)
            norms = np.array(row_norms, dtype=np.float32)
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
        if not scored:
            return []

        ids = [row[0] for _, _, row in scored]
        placeholders = ",".join("?" for _ in ids)
        text_rows = conn.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        text_by_id = {row[0]: row[1] for row in text_rows}

        results = []
        for sim, _, row in scored:
            text = text_by_id.get(row[0])
            if text is None:
                continue
            results.append(
                {
                    "id": row[0],
                    "contact": row[1],
                    "thread_key": row[2],
                    "start_time": row[3],
                    "end_time": row[4],
                    "text": text,
                    "message_count": row[5],
                    "similarity": sim,
                    "metadata": json.loads(row[8]) if row[8] else {},
                    "retrieval": "vector",
                }
            )
        return results
    finally:
        conn.close()


def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 5,
    db_path: Path = VECTOR_DB,
) -> list[dict]:
    """Merge vector and keyword retrieval so exact names/places are not lost."""
    top_k = max(1, min(top_k, 50))
    vector_results = search(query_embedding, top_k=top_k * 2, db_path=db_path)
    keyword_results = keyword_search(query, top_k=top_k * 2, db_path=db_path)

    merged: dict[int, dict] = {}
    order = 0
    for result in [*keyword_results, *vector_results]:
        order += 1
        existing = merged.get(result["id"])
        if existing is None:
            result["_merge_order"] = order
            merged[result["id"]] = result
            continue
        existing["similarity"] = max(existing["similarity"], result["similarity"])
        modes = {
            mode
            for mode in (
                existing.get("retrieval"),
                result.get("retrieval"),
            )
            if mode
        }
        if modes:
            existing["retrieval"] = "+".join(sorted(modes))

    results = sorted(
        merged.values(),
        key=lambda row: (row["similarity"], -row["_merge_order"]),
        reverse=True,
    )[:top_k]
    for result in results:
        result.pop("_merge_order", None)
    return results


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
        by_id = {
            r[0]: {
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
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]
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
