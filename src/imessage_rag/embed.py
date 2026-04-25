"""Generate embeddings via Ollama's local API."""

import re
import time

import requests

from imessage_rag.config import EMBED_DIMENSIONS, EMBED_MAX_CHARS, EMBED_MODEL, OLLAMA_URL

# nomic-embed-text has an 8192 token context window; ~4 chars/token is a safe estimate
_MAX_CHARS = EMBED_MAX_CHARS
_FALLBACK_CHAR_LIMITS = tuple(
    limit for limit in (20_000, 12_000, 8_000, 4_000) if limit < _MAX_CHARS
)

# Unicode object replacement char that iMessage inserts for attachments
_ATTACHMENT_PLACEHOLDER = re.compile(r"\ufffc")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def _clean(text: str, max_chars: int = _MAX_CHARS) -> str:
    """Strip characters that cause Ollama to choke."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ATTACHMENT_PLACEHOLDER.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _candidate_prompts(text: str) -> list[str]:
    """Build progressively smaller prompts for recovery from model-side 500s."""
    prompts: list[str] = []
    for limit in (_MAX_CHARS, *_FALLBACK_CHAR_LIMITS):
        candidate = _clean(text, max_chars=limit)
        if candidate and candidate not in prompts:
            prompts.append(candidate)
    return prompts


def _post_embedding(input_value: str | list[str]) -> requests.Response:
    payload = {"model": EMBED_MODEL, "input": input_value}
    if EMBED_DIMENSIONS is not None:
        payload["dimensions"] = EMBED_DIMENSIONS
    timeout = 120 if isinstance(input_value, str) else 300
    return requests.post(f"{OLLAMA_URL}/api/embed", json=payload, timeout=timeout)


def get_embeddings(texts: list[str], retries: int = 1) -> list[list[float]]:
    """Get embeddings for a batch of texts with one Ollama request."""
    if not texts:
        return []

    prompts = [_clean(text) for text in texts]
    if any(not prompt for prompt in prompts):
        raise ValueError("Cannot embed empty text after cleaning.")

    last_error: Exception | None = None
    for attempt in range(1 + retries):
        resp = _post_embedding(prompts)
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings") or []
            if len(embeddings) != len(prompts):
                raise ValueError(
                    "Ollama returned an unexpected number of embeddings "
                    f"({len(embeddings)} for {len(prompts)} prompts)."
                )
            return embeddings
        if resp.status_code >= 500:
            last_error = requests.HTTPError(
                f"{resp.status_code} Server Error for url: {resp.url} "
                f"(batch_size={len(prompts)})",
                response=resp,
            )
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
                continue
            break
        resp.raise_for_status()

    assert last_error is not None
    raise last_error


def get_embedding(text: str, retries: int = 1) -> list[float]:
    """Get embedding vector for a single text string.

    Retries on transient 500s, then falls back to shorter sanitized prompts
    for chunks that trip Ollama's embedding endpoint.
    """
    prompts = _candidate_prompts(text)
    if not prompts:
        raise ValueError("Cannot embed empty text after cleaning.")

    last_error: Exception | None = None

    for prompt_index, prompt in enumerate(prompts):
        attempts = 1 + retries if prompt_index == 0 else 1
        for attempt in range(attempts):
            resp = _post_embedding(prompt)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings") or []
                if embeddings:
                    return embeddings[0]
                raise ValueError("Ollama returned no embeddings.")
            if resp.status_code >= 500:
                last_error = requests.HTTPError(
                    f"{resp.status_code} Server Error for url: {resp.url} "
                    f"(prompt_len={len(prompt)}, fallback={prompt_index})",
                    response=resp,
                )
                if attempt < attempts - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                break
            resp.raise_for_status()

        if prompt_index < len(prompts) - 1:
            time.sleep(0.25)

    assert last_error is not None
    raise last_error
