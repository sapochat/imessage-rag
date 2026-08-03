import os
import json
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

# Load .env from repo root (two parents up from this file: src/imessage_rag/).
if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


_SETTINGS_PATH = Path(os.path.expanduser("~/.imessage-rag/settings.json"))


def _load_saved_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            return json.loads(_SETTINGS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {}


_SAVED_SETTINGS = _load_saved_settings()


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def _validate_localhost(url: str) -> str:
    """Ensure an HTTP(S) URL resolves only to loopback addresses."""
    parsed = urlparse(url)
    hostname = parsed.hostname

    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or "\\" in parsed.netloc
    ):
        raise ValueError(
            "URL must be an HTTP(S) localhost URL, such as "
            "'http://localhost:11434'."
        )

    try:
        port = parsed.port
    except ValueError:
        raise ValueError(
            "URL must be an HTTP(S) localhost URL, such as "
            "'http://localhost:11434'."
        )

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None

    if literal is not None:
        if not literal.is_loopback:
            raise ValueError(
                f"URL must point to localhost, but '{hostname}' is not loopback. "
                f"This system never sends data off-machine."
            )
        address = literal.compressed
    else:
        try:
            addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            raise ValueError(
                f"Hostname '{hostname}' cannot be resolved. "
                f"Only localhost URLs are allowed."
            )

        addresses = list(dict.fromkeys(
            str(addr_info[4][0])
            for addr_info in addr_infos
            if addr_info[4]
        ))
        if not addresses or not all(
            ipaddress.ip_address(addr).is_loopback for addr in addresses
        ):
            resolved = ", ".join(sorted(addresses)) or "no addresses"
            raise ValueError(
                f"URL must point to localhost, but '{hostname}' resolves to {resolved}. "
                f"This system never sends data off-machine."
            )

        # Pin validated hostnames to a loopback literal so DNS cannot change
        # between validation and the eventual HTTP request. Prefer IPv4 when
        # available because most local Ollama installations bind there.
        address = next((addr for addr in addresses if ":" not in addr), addresses[0])

    literal_host = f"[{address}]" if ":" in address else address
    netloc = literal_host if port is None else f"{literal_host}:{port}"
    return parsed._replace(netloc=netloc).geturl()


# iMessage
IMESSAGE_DB = _expand(os.getenv("IMESSAGE_DB", "~/Library/Messages/chat.db"))

# Contacts
_contacts_enabled = os.getenv("CONTACTS_ENABLED", "1").strip().lower()
CONTACTS_ENABLED = _contacts_enabled not in {"0", "false", "no", "off"}
_contacts_db = os.getenv("CONTACTS_DB", "").strip()
CONTACTS_DB = _expand(_contacts_db) if _contacts_db else None

# Ollama (always used for embeddings)
OLLAMA_URL = _validate_localhost(os.getenv("OLLAMA_URL", "http://localhost:11434"))
EMBED_PROFILES = {
    "fast": {
        "model": "qwen3-embedding:0.6b",
        "dimensions": 1024,
        "batch_size": 64,
        "workers": 2,
        "max_chars": 8000,
    },
    "full": {
        "model": "qwen3-embedding:8b",
        "dimensions": 4096,
        "batch_size": 32,
        "workers": 2,
        "max_chars": 12000,
    },
}
EMBED_PROFILE = (
    os.getenv("EMBED_PROFILE")
    or _SAVED_SETTINGS.get("embed_profile")
    or "custom"
).strip().lower()
_active_embed_profile = EMBED_PROFILES.get(EMBED_PROFILE, {})
EMBED_MODEL = _active_embed_profile.get("model") or os.getenv("EMBED_MODEL", "nomic-embed-text")
_embed_dimensions = os.getenv("EMBED_DIMENSIONS", "").strip()
if _active_embed_profile:
    EMBED_DIMENSIONS = _active_embed_profile["dimensions"]
elif _embed_dimensions:
    EMBED_DIMENSIONS = int(_embed_dimensions)
else:
    EMBED_DIMENSIONS = 768
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma4")

# Generation backend — "ollama" (default) or "openai" (local OpenAI-compatible proxy)
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "ollama").lower()
_gen_api_url = os.getenv("GENERATION_API_URL", "")
GENERATION_API_URL = _validate_localhost(_gen_api_url) if _gen_api_url else ""
GENERATION_API_KEY = os.getenv("GENERATION_API_KEY", "")

# Vector DB
VECTOR_DB = _expand(os.getenv("VECTOR_DB", "~/.imessage-rag/vectors.db"))

# Chunking
CHUNK_WINDOW_HOURS = int(os.getenv("CHUNK_WINDOW_HOURS", "4"))
CHUNK_MAX_MESSAGES = int(os.getenv("CHUNK_MAX_MESSAGES", "80"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", _active_embed_profile.get("batch_size", 32)))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", _active_embed_profile.get("workers", 2)))
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", _active_embed_profile.get("max_chars", 12000)))

# Auth
AUTH_TOKEN_PATH = _expand("~/.imessage-rag/auth_token")

# Apple Core Data epoch offset (seconds between 1970-01-01 and 2001-01-01)
APPLE_EPOCH_OFFSET = 978307200
