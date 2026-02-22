"""Generation abstraction — route to Ollama or OpenAI-compatible backends."""

import json
from typing import Generator

import requests

from src import settings
from src.config import OLLAMA_URL


def _openai_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def stream_chat(messages: list[dict], model: str | None = None) -> Generator[str, None, None]:
    """Stream tokens from a chat completion. Yields token strings.

    Args:
        messages: OpenAI-style messages list [{"role": ..., "content": ...}].
        model: Model name to use. Defaults to settings/env value.
    """
    model = model or settings.get_generation_model()
    backend = settings.get_generation_backend()

    if backend == "openai":
        yield from _stream_chat_openai(messages, model)
    else:
        yield from _stream_chat_ollama(messages, model)


def _stream_chat_ollama(messages: list[dict], model: str) -> Generator[str, None, None]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            yield token
        if data.get("done"):
            break


def _stream_chat_openai(messages: list[dict], model: str) -> Generator[str, None, None]:
    api_url = settings.get_generation_api_url()
    api_key = settings.get_generation_api_key()

    resp = requests.post(
        f"{api_url}/chat/completions",
        headers=_openai_headers(api_key),
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.startswith("data: "):
            continue
        payload = text[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        data = json.loads(payload)
        choices = data.get("choices", [])
        if not choices:
            continue
        token = choices[0].get("delta", {}).get("content", "")
        if token:
            yield token


def generate_once(prompt: str, model: str | None = None) -> str:
    """Single non-streaming generation. Returns the full response text.

    For the OpenAI backend (which may be streaming-only), we accumulate
    streamed tokens.
    """
    model = model or settings.get_generation_model()
    backend = settings.get_generation_backend()

    if backend == "openai":
        return _generate_once_openai(prompt, model)
    else:
        return _generate_once_ollama(prompt, model)


def _generate_once_ollama(prompt: str, model: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _generate_once_openai(prompt: str, model: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    tokens = list(_stream_chat_openai(messages, model))
    return "".join(tokens).strip()
