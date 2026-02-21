import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


_LOCALHOST_ADDRS = {"127.0.0.1", "::1"}


def _validate_localhost(url: str) -> str:
    """Ensure a URL points to localhost. Raises ValueError otherwise."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if hostname in ("localhost", "127.0.0.1", "::1"):
        return url

    # Resolve hostname to check if it points to a loopback address
    try:
        addr = socket.gethostbyname(hostname)
    except socket.gaierror:
        raise ValueError(
            f"OLLAMA_URL hostname '{hostname}' cannot be resolved. "
            f"Only localhost URLs are allowed (e.g. http://localhost:11434)."
        )

    if addr not in _LOCALHOST_ADDRS:
        raise ValueError(
            f"OLLAMA_URL must point to localhost, but '{hostname}' resolves to {addr}. "
            f"This system never sends data off-machine. "
            f"Use http://localhost:11434 or http://127.0.0.1:11434."
        )

    return url


# iMessage
IMESSAGE_DB = _expand(os.getenv("IMESSAGE_DB", "~/Library/Messages/chat.db"))

# Apple Mail
MAIL_DIR = _expand(os.getenv("MAIL_DIR", "~/Library/Mail/V10"))

# Ollama
OLLAMA_URL = _validate_localhost(os.getenv("OLLAMA_URL", "http://localhost:11434"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma3:4b")

# Vector DB
VECTOR_DB = _expand(os.getenv("VECTOR_DB", "~/.personal-rag/vectors.db"))

# Chunking
CHUNK_WINDOW_HOURS = int(os.getenv("CHUNK_WINDOW_HOURS", "4"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))

# Auth
AUTH_TOKEN_PATH = _expand("~/.personal-rag/auth_token")

# Apple Core Data epoch offset (seconds between 1970-01-01 and 2001-01-01)
APPLE_EPOCH_OFFSET = 978307200
