"""Settings routes — configure generation backend from the web UI."""

import requests as http_requests

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src import settings
from src.config import OLLAMA_URL
from src.web.app import templates

router = APIRouter()


def _test_backend(backend: str, model: str, api_url: str, api_key: str) -> str:
    """Probe the configured backend and return a status message."""
    try:
        if backend == "ollama":
            resp = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(model in m for m in models):
                return f"Ollama online — model '{model}' ready."
            return f"Ollama online — but model '{model}' not found. Available: {', '.join(models)}"
        else:
            # OpenAI-compatible — probe /models
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = http_requests.get(f"{api_url}/models", headers=headers, timeout=5)
            resp.raise_for_status()
            return f"Backend online at {api_url}."
    except Exception as e:
        return f"Connection failed: {e}"


@router.get("/settings")
async def settings_page(request: Request):
    current = settings.get_all()
    return templates.TemplateResponse(
        "settings.html", {"request": request, "current": current}
    )


@router.post("/settings/save")
async def settings_save(request: Request):
    form = await request.form()

    backend = form.get("generation_backend", "ollama").strip().lower()
    model = form.get("generation_model", "").strip()
    api_url = form.get("generation_api_url", "").strip()
    api_key = form.get("generation_api_key", "").strip()

    data: dict = {}
    if backend:
        data["generation_backend"] = backend
    if model:
        data["generation_model"] = model
    if api_url:
        data["generation_api_url"] = api_url
    # Empty API key = keep existing (user doesn't re-enter every time)
    if api_key:
        data["generation_api_key"] = api_key

    try:
        settings.save(data)
    except ValueError as e:
        return templates.TemplateResponse(
            "partials/settings_result.html",
            {"request": request, "success": False, "message": str(e)},
        )

    # Test connection with the effective settings
    effective = settings.get_all()
    test_msg = _test_backend(
        effective["generation_backend"],
        effective["generation_model"],
        effective["generation_api_url"],
        effective["generation_api_key"],
    )

    is_ok = "online" in test_msg.lower() or "ready" in test_msg.lower()
    message = f"Saved. {test_msg}"

    return templates.TemplateResponse(
        "partials/settings_result.html",
        {"request": request, "success": is_ok, "message": message},
    )
