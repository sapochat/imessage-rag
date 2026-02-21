"""Query routes — search page, SSE streaming, and multi-turn chat."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.query import retrieve, stream_answer, stream_answer_chat
from src.vectordb import fetch_by_ids
from src.web.app import templates

router = APIRouter()

MAX_TOP_K = 50


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    source: str | None = None
    prior_chunk_ids: list[int] = []


@router.get("/")
async def query_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/api/query/stream")
async def query_stream(q: str, top_k: int = 5, source: str | None = None):
    """SSE endpoint that streams answer tokens (single-shot, backwards compat)."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    if source == "":
        source = None

    def event_generator():
        for event in stream_answer(q, top_k=top_k, source=source):
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE endpoint for multi-turn chat with conversation history."""
    source = req.source if req.source else None

    def event_generator():
        for event in stream_answer_chat(
            req.query, req.history, top_k=req.top_k, source=source,
            prior_chunk_ids=req.prior_chunk_ids,
        ):
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/chunk/{chunk_id}")
async def chunk_detail(chunk_id: int):
    """Return full chunk data including text and metadata."""
    results = fetch_by_ids([chunk_id])
    if not results:
        return JSONResponse({"detail": "Chunk not found"}, status_code=404)
    return results[0]


@router.get("/api/query/retrieve")
async def query_retrieve(q: str, top_k: int = 5, source: str | None = None):
    """JSON endpoint returning raw retrieved chunks."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    if source == "":
        source = None
    results = retrieve(q, top_k=top_k, source=source)
    return {"results": results}
