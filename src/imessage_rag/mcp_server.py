"""Minimal stdio MCP server exposing local retrieval tools."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TextIO

SERVER_NAME = "imessage-rag"
SERVER_VERSION = "0.1.0"
MAX_TOP_K = 20


@dataclass
class JsonRpcError(Exception):
    code: int
    message: str
    data: Any = None


def _isoformat(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _format_search_text(results: list[dict]) -> str:
    if not results:
        return "No matching chunks found."

    blocks = []
    for idx, row in enumerate(results, 1):
        start = row.get("start_time") or "unknown"
        end = row.get("end_time") or "unknown"
        blocks.append(
            f"{idx}. {row['contact']} | "
            f"{start} -> {end} | "
            f"{row['message_count']} messages | "
            f"similarity {row['similarity']:.3f}\n"
            f"{row['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _search_messages(arguments: dict[str, Any]) -> dict[str, Any]:
    query = (arguments.get("query") or "").strip()
    if not query:
        raise JsonRpcError(-32602, "search_messages requires a non-empty 'query'")

    top_k_raw = arguments.get("top_k", 5)
    try:
        top_k = int(top_k_raw)
    except (TypeError, ValueError) as exc:
        raise JsonRpcError(-32602, "'top_k' must be an integer") from exc
    top_k = max(1, min(top_k, MAX_TOP_K))

    from imessage_rag.embed import get_embedding
    from imessage_rag.vectordb import search

    query_embedding = get_embedding(query)
    results = search(query_embedding, top_k=top_k)
    safe_results = [
        {
            "id": row["id"],
            "contact": row["contact"],
            "start_time": _isoformat(row["start_time"]),
            "end_time": _isoformat(row["end_time"]),
            "message_count": row["message_count"],
            "similarity": round(row["similarity"], 3),
            "text": row["text"],
            "metadata": row.get("metadata", {}),
        }
        for row in results
    ]

    return {
        "content": [{"type": "text", "text": _format_search_text(safe_results)}],
        "structuredContent": {
            "query": query,
            "top_k": top_k,
            "results": safe_results,
        },
    }


def _get_chunk(arguments: dict[str, Any]) -> dict[str, Any]:
    chunk_id_raw = arguments.get("chunk_id")
    try:
        chunk_id = int(chunk_id_raw)
    except (TypeError, ValueError) as exc:
        raise JsonRpcError(-32602, "get_chunk requires integer 'chunk_id'") from exc

    from imessage_rag.vectordb import fetch_by_ids

    results = fetch_by_ids([chunk_id])
    if not results:
        raise JsonRpcError(-32602, f"Chunk {chunk_id} was not found")

    row = results[0]
    chunk = {
        "id": row["id"],
        "contact": row["contact"],
        "start_time": _isoformat(row["start_time"]),
        "end_time": _isoformat(row["end_time"]),
        "message_count": row["message_count"],
        "text": row["text"],
        "metadata": row.get("metadata", {}),
    }
    text = (
        f"{chunk['contact']} | "
        f"{chunk['start_time']} -> {chunk['end_time']} | "
        f"{chunk['message_count']} messages\n\n"
        f"{chunk['text']}"
    )
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": chunk,
    }


def _get_stats(_: dict[str, Any]) -> dict[str, Any]:
    from imessage_rag.vectordb import get_stats

    stats = get_stats()
    text = (
        f"Total chunks: {stats['total_chunks']}\n"
        f"DB size: {stats['db_size_mb']} MB"
    )
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": stats,
    }


TOOLS = {
    "search_messages": {
        "description": (
            "Semantic search over the local imessage-rag database. "
            "Use this first to find relevant iMessage conversation chunks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"Maximum number of chunks to return (1-{MAX_TOP_K}).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": MAX_TOP_K,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "get_chunk": {
        "description": "Fetch the full text and metadata for a specific chunk by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "integer",
                    "description": "Chunk ID returned by search_messages.",
                }
            },
            "required": ["chunk_id"],
            "additionalProperties": False,
        },
    },
    "get_stats": {
        "description": "Return chunk counts and DB size for the local vector database.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


def _make_response(message_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _make_error(message_id: Any, error: JsonRpcError) -> dict[str, Any]:
    payload = {"code": error.code, "message": error.message}
    if error.data is not None:
        payload["data"] = error.data
    return {"jsonrpc": "2.0", "id": message_id, "error": payload}


def _handle_request(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    params = params or {}

    if method == "initialize":
        protocol_version = params.get("protocolVersion", "2025-06-18")
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": name,
                    "description": spec["description"],
                    "inputSchema": spec["inputSchema"],
                }
                for name, spec in TOOLS.items()
            ]
        }

    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            raise JsonRpcError(-32602, f"Unknown tool '{name}'")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(-32602, "'arguments' must be an object")
        if name == "search_messages":
            return _search_messages(arguments)
        if name == "get_chunk":
            return _get_chunk(arguments)
        if name == "get_stats":
            return _get_stats(arguments)
        raise JsonRpcError(-32602, f"Unknown tool '{name}'")

    if method == "ping":
        return {}

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return {}

    raise JsonRpcError(-32601, f"Method '{method}' not found")


def _write_message(outstream: TextIO, payload: dict[str, Any]) -> None:
    outstream.write(json.dumps(payload) + "\n")
    outstream.flush()


def serve(instream: TextIO = sys.stdin, outstream: TextIO = sys.stdout) -> int:
    for line in instream:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_message(
                outstream,
                _make_error(None, JsonRpcError(-32700, "Parse error", str(exc))),
            )
            continue

        if not isinstance(request, dict):
            _write_message(
                outstream,
                _make_error(None, JsonRpcError(-32600, "Invalid Request")),
            )
            continue

        message_id = request.get("id")
        method = request.get("method")
        params = request.get("params")

        if not method:
            _write_message(
                outstream,
                _make_error(message_id, JsonRpcError(-32600, "Invalid Request")),
            )
            continue

        try:
            result = _handle_request(method, params)
        except JsonRpcError as exc:
            if message_id is not None:
                _write_message(outstream, _make_error(message_id, exc))
            continue
        except Exception as exc:  # pragma: no cover - defensive server boundary
            if message_id is not None:
                _write_message(
                    outstream,
                    _make_error(message_id, JsonRpcError(-32603, "Internal error", str(exc))),
                )
            continue

        if message_id is not None:
            _write_message(outstream, _make_response(message_id, result))

    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
