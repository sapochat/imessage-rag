"""Generate embeddings via Ollama's local API."""

import re
import time

import requests

from src.config import EMBED_MODEL, OLLAMA_URL

# nomic-embed-text has an 8192 token context window; ~4 chars/token is a safe estimate
_MAX_CHARS = 30_000

# Unicode object replacement char that iMessage inserts for attachments
_ATTACHMENT_PLACEHOLDER = re.compile(r"\ufffc")


def _clean(text: str) -> str:
    """Strip characters that cause Ollama to choke."""
    text = _ATTACHMENT_PLACEHOLDER.sub("", text)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS]
    return text


def get_embedding(text: str, retries: int = 2) -> list[float]:
    """Get embedding vector for a single text string.

    Retries on transient 500s with a short back-off.
    """
    text = _clean(text)

    for attempt in range(1 + retries):
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["embedding"]
        if resp.status_code >= 500 and attempt < retries:
            time.sleep(1 * (attempt + 1))
            continue
        resp.raise_for_status()
