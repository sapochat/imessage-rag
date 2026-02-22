"""Status routes — dashboard with vector DB stats and Ollama health."""

import requests as http_requests

from fastapi import APIRouter, Request

from src import settings
from src.config import EMBED_MODEL, OLLAMA_URL
from src.vectordb import get_stats
from src.web.app import templates

router = APIRouter()


def _check_ollama() -> dict:
    """Check Ollama connectivity and available models."""
    gen_backend = settings.get_generation_backend()
    gen_model = settings.get_generation_model()
    try:
        resp = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        gen_via_ollama = gen_backend == "ollama"
        return {
            "status": "online",
            "models": models,
            "has_embed": any(EMBED_MODEL in m for m in models),
            "has_gen": any(gen_model in m for m in models) if gen_via_ollama else None,
        }
    except Exception as e:
        return {"status": "offline", "error": str(e), "models": [], "has_embed": False, "has_gen": None}


def _check_generation_backend() -> dict:
    """Check the configured generation backend."""
    gen_backend = settings.get_generation_backend()
    gen_model = settings.get_generation_model()
    gen_api_url = settings.get_generation_api_url()

    info = {"backend": gen_backend, "model": gen_model}

    if gen_backend == "ollama":
        info["status"] = "via_ollama"
        return info

    # OpenAI-compatible backend — probe /models endpoint
    try:
        resp = http_requests.get(f"{gen_api_url}/models", timeout=5)
        resp.raise_for_status()
        info["status"] = "online"
    except Exception as e:
        info["status"] = "offline"
        info["error"] = str(e)

    return info


@router.get("/status")
async def status_page(request: Request):
    stats = get_stats()
    ollama = _check_ollama()
    generation = _check_generation_backend()
    return templates.TemplateResponse(
        "status.html", {"request": request, "stats": stats, "ollama": ollama, "generation": generation}
    )


@router.get("/status/api/refresh")
async def status_refresh(request: Request):
    stats = get_stats()
    ollama = _check_ollama()
    generation = _check_generation_backend()
    return templates.TemplateResponse(
        "partials/status_cards.html", {"request": request, "stats": stats, "ollama": ollama, "generation": generation}
    )
