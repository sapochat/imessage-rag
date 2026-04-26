"""Semantic search over the vector DB and LLM-powered answer generation."""

from datetime import datetime, timezone
from typing import Generator

from imessage_rag.generate import generate_once, stream_chat
from imessage_rag.embed import get_embedding
from imessage_rag.vectordb import fetch_by_ids, hybrid_search

_SYSTEM_PROMPT = (
    "You are a search assistant for the user's iMessage history. "
    "Your ONLY job is to find and quote relevant parts from the excerpts below.\n\n"
    "RULES:\n"
    "- ONLY use information from the excerpts. NEVER use your own knowledge.\n"
    "- Quote or paraphrase the actual messages. Include who said it and when.\n"
    "- If the excerpts contain nothing relevant, say \"Nothing found in your "
    "messages about this.\" Do NOT explain the topic yourself.\n"
    "- Do NOT define terms, give background info, or answer from general knowledge."
)

_MAX_CONTEXT_CHARS_PER_CHUNK = 4_000
_MAX_CONTEXT_CHARS_TOTAL = 18_000
_MAX_HISTORY_TURNS = 12
_MAX_CONTEXT_CHUNKS = 20


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[excerpt truncated]"


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Embed the query and return hybrid vector/keyword matches."""
    query_embedding = get_embedding(query)
    return hybrid_search(query, query_embedding, top_k=top_k)


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    remaining = _MAX_CONTEXT_CHARS_TOTAL
    for i, r in enumerate(results, 1):
        if remaining <= 0:
            break
        start = datetime.fromtimestamp(r["start_time"], tz=timezone.utc)
        end = datetime.fromtimestamp(r["end_time"], tz=timezone.utc)
        header = (
            f"[Chunk {i} | {r['contact']} | "
            f"{start.strftime('%Y-%m-%d %H:%M')}–{end.strftime('%H:%M')} | "
            f"{r['message_count']} messages | similarity: {r['similarity']:.3f} | "
            f"{r.get('retrieval', 'unknown')}]"
        )
        text_limit = min(_MAX_CONTEXT_CHARS_PER_CHUNK, remaining)
        text = _clip_text(r["text"], text_limit)
        parts.append(f"{header}\n{text}")
        remaining -= len(text)
    return "\n\n---\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"--- CONVERSATION EXCERPTS ---\n{context}\n"
        f"--- END EXCERPTS ---\n\n"
        f"Question: {query}\n\n"
        "Answer (cite only from excerpts above):"
    )


def _safe_result(r: dict, extras: dict | None = None) -> dict:
    base = {
        "id": r["id"],
        "contact": r["contact"],
        "start_time": r["start_time"],
        "end_time": r["end_time"],
        "message_count": r["message_count"],
        "similarity": round(r["similarity"], 3),
        "retrieval": r.get("retrieval", "unknown"),
        "text": r["text"][:300],
        "metadata": r.get("metadata", {}),
    }
    if extras:
        base.update(extras)
    return base


def stream_answer(
    query: str, top_k: int = 5,
) -> Generator[dict, None, None]:
    """Retrieve chunks and stream an answer as event dicts.

    Yields dicts with:
      {"type": "sources", "data": [list of result dicts]}
      {"type": "token",   "data": "text fragment"}
      {"type": "done",    "data": ""}
      {"type": "error",   "data": "error message"}
    """
    try:
        results = retrieve(query, top_k=top_k)
    except Exception as e:
        yield {"type": "error", "data": f"Retrieval failed: {e}"}
        return

    if not results:
        yield {"type": "sources", "data": []}
        yield {"type": "error", "data": "No matching chunks found. Have you run 'ingest' yet?"}
        return

    yield {"type": "sources", "data": [_safe_result(r) for r in results]}

    context = _format_context(results)
    prompt = _build_prompt(query, context)
    messages = [{"role": "user", "content": prompt}]

    try:
        for token in stream_chat(messages):
            yield {"type": "token", "data": token}
    except Exception as e:
        yield {"type": "error", "data": f"Generation failed: {e}"}
        return

    yield {"type": "done", "data": ""}


def reformulate_query(
    user_msg: str, history: list[dict],
) -> str:
    """Rewrite a follow-up question as a standalone search query."""
    if not history:
        return user_msg

    recent = history[-6:]
    convo = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:200]}"
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
        rewritten = generate_once(prompt)
        return rewritten if rewritten else user_msg
    except Exception:
        return user_msg


def stream_answer_chat(
    user_msg: str,
    history: list[dict],
    top_k: int = 5,
    prior_chunk_ids: list[int] | None = None,
) -> Generator[dict, None, None]:
    """Multi-turn chat: reformulate → retrieve → merge prior chunks → stream."""
    history = history[-_MAX_HISTORY_TURNS:]

    search_query = reformulate_query(user_msg, history)

    try:
        new_results = retrieve(search_query, top_k=top_k)
    except Exception as e:
        yield {"type": "error", "data": f"Retrieval failed: {e}"}
        return
    new_results = new_results[:_MAX_CONTEXT_CHUNKS]

    new_ids = {r["id"] for r in new_results}
    prior_ids_to_fetch = []
    for cid in reversed(prior_chunk_ids or []):
        if cid in new_ids or cid in prior_ids_to_fetch:
            continue
        prior_ids_to_fetch.append(cid)
        if len(prior_ids_to_fetch) >= _MAX_CONTEXT_CHUNKS:
            break
    prior_ids_to_fetch.reverse()

    prior_results = []
    if prior_ids_to_fetch:
        try:
            prior_results = fetch_by_ids(prior_ids_to_fetch)
        except Exception:
            pass

    remaining_prior_slots = max(0, _MAX_CONTEXT_CHUNKS - len(new_results))
    selected_prior_results = prior_results[-remaining_prior_slots:] if remaining_prior_slots else []
    all_results = new_results + selected_prior_results

    if not all_results:
        yield {"type": "sources", "data": []}
        yield {"type": "error", "data": "No matching chunks found. Have you run 'ingest' yet?"}
        return

    yield {
        "type": "sources",
        "data": [_safe_result(r, {"is_new": r["id"] in new_ids}) for r in all_results],
    }

    context = _format_context(all_results)
    system_msg = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"--- CONVERSATION EXCERPTS ---\n{context}\n"
        f"--- END EXCERPTS ---"
    )
    messages = [{"role": "system", "content": system_msg}]
    for turn in history[-8:]:
        role = turn.get("role")
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": str(turn.get("content", ""))[:4_000]})
    messages.append({"role": "user", "content": user_msg})

    try:
        for token in stream_chat(messages):
            yield {"type": "token", "data": token}
    except Exception as e:
        yield {"type": "error", "data": f"Generation failed: {e}"}
        return

    yield {"type": "done", "data": ""}


def generate_answer(query: str, top_k: int = 5) -> None:
    """Retrieve relevant chunks and stream an LLM-generated answer to stdout."""
    for event in stream_answer(query, top_k=top_k):
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
            print()
