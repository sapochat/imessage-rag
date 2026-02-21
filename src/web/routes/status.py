"""Status routes — dashboard with vector DB stats and Ollama health."""

import requests as http_requests

from fastapi import APIRouter, Request

from src.config import OLLAMA_URL, GENERATION_MODEL, EMBED_MODEL
from src.vectordb import get_stats
from src.web.app import templates

router = APIRouter()


def _check_ollama() -> dict:
    """Check Ollama connectivity and available models."""
    try:
        resp = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {
            "status": "online",
            "models": models,
            "has_embed": any(EMBED_MODEL in m for m in models),
            "has_gen": any(GENERATION_MODEL in m for m in models),
        }
    except Exception as e:
        return {"status": "offline", "error": str(e), "models": [], "has_embed": False, "has_gen": False}


@router.get("/status")
async def status_page(request: Request):
    stats = get_stats()
    ollama = _check_ollama()
    return templates.TemplateResponse(
        "status.html", {"request": request, "stats": stats, "ollama": ollama}
    )


@router.get("/status/api/refresh")
async def status_refresh(request: Request):
    stats = get_stats()
    ollama = _check_ollama()
    return templates.TemplateResponse(
        "partials/status_cards.html", {"request": request, "stats": stats, "ollama": ollama}
    )
