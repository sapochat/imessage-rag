import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


# iMessage
IMESSAGE_DB = _expand(os.getenv("IMESSAGE_DB", "~/Library/Messages/chat.db"))

# Apple Mail
MAIL_DIR = _expand(os.getenv("MAIL_DIR", "~/Library/Mail/V10"))

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma3:4b")

# Vector DB
VECTOR_DB = _expand(os.getenv("VECTOR_DB", "~/.personal-rag/vectors.db"))

# Chunking
CHUNK_WINDOW_HOURS = int(os.getenv("CHUNK_WINDOW_HOURS", "4"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))

# Apple Core Data epoch offset (seconds between 1970-01-01 and 2001-01-01)
APPLE_EPOCH_OFFSET = 978307200
