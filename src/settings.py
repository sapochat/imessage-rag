"""Persistent settings store — generation config saved to ~/.personal-rag/settings.json.

Env vars in .env serve as fallback defaults. Saved UI settings take precedence.
Only generation-related settings are mutable here; embedding/DB/chunking stay in .env.
"""

import json
import os
from pathlib import Path

from src.config import (
    GENERATION_API_KEY,
    GENERATION_API_URL,
    GENERATION_BACKEND,
    GENERATION_MODEL,
    _validate_localhost,
)

_SETTINGS_PATH = Path(os.path.expanduser("~/.personal-rag/settings.json"))

_cache: dict | None = None
_cache_mtime: float = 0.0


def _load() -> dict:
    """Read settings.json with mtime-based cache to avoid redundant disk reads."""
    global _cache, _cache_mtime

    if not _SETTINGS_PATH.exists():
        _cache = {}
        _cache_mtime = 0.0
        return _cache

    mtime = _SETTINGS_PATH.stat().st_mtime
    if _cache is not None and mtime == _cache_mtime:
        return _cache

    _cache = json.loads(_SETTINGS_PATH.read_text())
    _cache_mtime = mtime
    return _cache


def save(data: dict) -> None:
    """Validate and persist settings. Merges with existing values."""
    # Validate localhost constraint on API URL if provided
    api_url = data.get("generation_api_url", "")
    if api_url:
        _validate_localhost(api_url)

    existing = _load()
    existing.update(data)

    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(existing, indent=2))
    _SETTINGS_PATH.chmod(0o600)

    # Bust cache so next read picks up changes
    global _cache, _cache_mtime
    _cache = existing
    _cache_mtime = _SETTINGS_PATH.stat().st_mtime


def get_generation_backend() -> str:
    return _load().get("generation_backend") or GENERATION_BACKEND


def get_generation_model() -> str:
    return _load().get("generation_model") or GENERATION_MODEL


def get_generation_api_url() -> str:
    return _load().get("generation_api_url") or GENERATION_API_URL


def get_generation_api_key() -> str:
    return _load().get("generation_api_key") or GENERATION_API_KEY


def get_all() -> dict:
    """Return all effective generation settings (saved values with env fallbacks)."""
    return {
        "generation_backend": get_generation_backend(),
        "generation_model": get_generation_model(),
        "generation_api_url": get_generation_api_url(),
        "generation_api_key": get_generation_api_key(),
    }
