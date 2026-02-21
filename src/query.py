"""Semantic search over the vector DB and answer generation via Ollama."""

import json
from datetime import datetime, timezone
from typing import Generator

import requests

from src.config import GENERATION_MODEL, OLLAMA_URL
from src.embed import get_embedding
from src.vectordb import search


def retrieve(query: str, top_k: int = 5, source: str | None = None) -> list[dict]:
    """Embed the query and return the top-k matching chunks."""
    query_embedding = get_embedding(query)
    return search(query_embedding, top_k=top_k, source=source)


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    for i, r in enumerate(results, 1):
        start = datetime.fromtimestamp(r["start_time"], tz=timezone.utc)
        end = datetime.fromtimestamp(r["end_time"], tz=timezone.utc)
        header = (
            f"[Chunk {i} | {r['source']} | {r['contact']} | "
            f"{start.strftime('%Y-%m-%d %H:%M')}–{end.strftime('%H:%M')} | "
            f"{r['message_count']} messages | similarity: {r['similarity']:.3f}]"
        )
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    return (
        "You are a helpful assistant answering questions about the user's personal "
        "messages. Use ONLY the conversation excerpts provided below to answer. "
        "If the answer isn't in the excerpts, say so. Be concise.\n\n"
        f"--- CONVERSATION EXCERPTS ---\n{context}\n"
        f"--- END EXCERPTS ---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def stream_answer(
    query: str, top_k: int = 5, source: str | None = None
) -> Generator[dict, None, None]:
    """Retrieve chunks and stream an answer as event dicts.

    Yields dicts with:
      {"type": "sources", "data": [list of result dicts]}
      {"type": "token",   "data": "text fragment"}
      {"type": "done",    "data": ""}
      {"type": "error",   "data": "error message"}
    """
    try:
        results = retrieve(query, top_k=top_k, source=source)
    except Exception as e:
        yield {"type": "error", "data": f"Retrieval failed: {e}"}
        return

    if not results:
        yield {"type": "sources", "data": []}
        yield {"type": "error", "data": "No matching chunks found. Have you run 'ingest' yet?"}
        return

    # Strip embedding blobs before sending to client
    safe_results = []
    for r in results:
        safe_results.append({
            "contact": r["contact"],
            "source": r["source"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "message_count": r["message_count"],
            "similarity": round(r["similarity"], 3),
            "text": r["text"][:300],
        })
    yield {"type": "sources", "data": safe_results}

    context = _format_context(results)
    prompt = _build_prompt(query, context)

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GENERATION_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        yield {"type": "error", "data": f"Generation failed: {e}"}
        return

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response", "")
        if token:
            yield {"type": "token", "data": token}
        if data.get("done"):
            break

    yield {"type": "done", "data": ""}


def reformulate_query(
    user_msg: str, history: list[dict],
) -> str:
    """Rewrite a follow-up question as a standalone search query.

    Uses Ollama to combine conversation context with the new message so that
    vector-search retrieval works on follow-ups like "tell me more about that".
    If there's no history, returns the original message unchanged.
    """
    if not history:
        return user_msg

    # Build a compact summary of the last few turns (keep token budget small)
    recent = history[-6:]  # last 3 exchanges max
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

    prompt = (
        "Given the conversation below, rewrite the latest user message as a "
        "standalone search query that captures the full intent. Output ONLY the "
        "rewritten query, nothing else.\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Latest message: {user_msg}\n\n"
        "Standalone search query:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GENERATION_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip()
        return rewritten if rewritten else user_msg
    except Exception:
        return user_msg


def stream_answer_chat(
    user_msg: str,
    history: list[dict],
    top_k: int = 5,
    source: str | None = None,
) -> Generator[dict, None, None]:
    """Multi-turn chat: reformulate → retrieve → stream answer with history.

    Args:
        user_msg: The latest user message.
        history: Prior turns as [{"role": "user"|"assistant", "content": "..."}].
        top_k: Number of chunks to retrieve.
        source: Optional source filter.

    Yields the same event dict format as stream_answer().
    """
    # Step 1 — reformulate follow-up into standalone retrieval query
    search_query = reformulate_query(user_msg, history)

    # Step 2 — retrieve
    try:
        results = retrieve(search_query, top_k=top_k, source=source)
    except Exception as e:
        yield {"type": "error", "data": f"Retrieval failed: {e}"}
        return

    if not results:
        yield {"type": "sources", "data": []}
        yield {"type": "error", "data": "No matching chunks found. Have you run 'ingest' yet?"}
        return

    safe_results = []
    for r in results:
        safe_results.append({
            "contact": r["contact"],
            "source": r["source"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "message_count": r["message_count"],
            "similarity": round(r["similarity"], 3),
            "text": r["text"][:300],
        })
    yield {"type": "sources", "data": safe_results}

    # Step 3 — build messages array for Ollama /api/chat
    context = _format_context(results)

    system_msg = (
        "You are a helpful assistant answering questions about the user's personal "
        "messages. Use ONLY the conversation excerpts provided below to answer. "
        "If the answer isn't in the excerpts, say so. Be concise and specific.\n\n"
        f"--- CONVERSATION EXCERPTS ---\n{context}\n"
        f"--- END EXCERPTS ---"
    )

    messages = [{"role": "system", "content": system_msg}]

    # Include recent history so the model has conversational context
    for turn in history[-8:]:  # last 4 exchanges max to stay within context
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_msg})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": GENERATION_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        yield {"type": "error", "data": f"Generation failed: {e}"}
        return

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            yield {"type": "token", "data": token}
        if data.get("done"):
            break

    yield {"type": "done", "data": ""}


def generate_answer(query: str, top_k: int = 5, source: str | None = None) -> None:
    """Retrieve relevant chunks and stream an LLM-generated answer to stdout."""
    for event in stream_answer(query, top_k=top_k, source=source):
        if event["type"] == "sources":
            if not event["data"]:
                print("No matching chunks found. Have you run 'ingest' yet?")
                return
            contacts = sorted({r["contact"] for r in event["data"]})
            print(f"Found {len(event['data'])} relevant chunks from: {', '.join(contacts)}")
            print()
        elif event["type"] == "token":
            print(event["data"], end="", flush=True)
        elif event["type"] == "error":
            print(event["data"])
            return
        elif event["type"] == "done":
            print()  # final newline
