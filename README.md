# Personal RAG

Local-only RAG system that indexes your iMessage and Apple Mail for semantic search and AI-powered Q&A. Everything runs on localhost — no data ever leaves your machine.

## How it works

1. **Ingest** — Streams messages from iMessage's SQLite DB (Apple Mail support planned)
2. **Chunk** — Groups iMessages by contact + 4-hour time window
3. **Embed** — Generates vectors via Ollama (`nomic-embed-text`) and stores in a local SQLite DB
4. **Query** — Semantic search over embeddings, then streams an answer from Gemma 3 4B
5. **Multi-turn chat** — Follow-up questions carry full context: prior chunks are accumulated across turns (capped at 20) so the model always sees the raw conversations, not just previous answers

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- [Ollama](https://ollama.com) running locally with two models pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull gemma3:4b
  ```
- Terminal with **Full Disk Access** (System Settings > Privacy & Security) to read `chat.db` and Mail directories

## Setup

```bash
git clone <repo-url> && cd personal-rag
pip install -e .
```

Optionally create a `.env` file to override defaults:

```
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
GENERATION_MODEL=gemma3:4b
VECTOR_DB=~/.personal-rag/vectors.db
CHUNK_WINDOW_HOURS=4
```

## Usage

### CLI

```bash
# Ingest recent messages (start here)
python3 cli.py ingest --source imessage --since 30d

# Full historical backfill (run overnight)
python3 cli.py ingest --source imessage

# Query
python3 cli.py query "What restaurant did Sarah recommend?"

# Retrieve raw chunks without LLM generation
python3 cli.py query "meeting notes" --retrieve-only --top-k 10

# Check DB stats
python3 cli.py status
```

### Web UI

```bash
python3 cli.py serve
# Prints a URL with an auth token — open it in your browser
# e.g. http://127.0.0.1:5391?token=<generated-token>
```

Three pages:

- **Query** — Multi-turn chat interface with streaming responses. Ask a question, then follow up naturally ("tell me more about that", "what else did they say"). Retrieved chunks accumulate across turns so you can drill deeper without losing context.
- **Ingest** — Start ingestion jobs, watch progress live, cancel if needed.
- **Status** — Chunk counts, DB size, Ollama connectivity and model availability.

## Architecture

```
iMessage (chat.db) ── extract → chunk → embed (Ollama) → SQLite vector DB
                                                               │
                                                               ▼
                      CLI / Web UI ── query → semantic search + Gemma 3 → answer
```

All processing is local. The only network calls are to `localhost:11434` (Ollama).

## Performance

On Apple Silicon (M1/M2/M3):

| Operation | Speed |
|-----------|-------|
| Embedding | ~50-100 chunks/min |
| Full backfill | Several hours (run overnight) |
| Query retrieval | <100ms |
| Answer generation | 5-20s |
| Disk usage | ~2-5 GB for text + embeddings |

## Privacy

This system exists because your messages are private. By design:

- Zero external API calls — everything runs through Ollama on localhost
- No telemetry, no analytics, no cloud services
- Vector DB stored locally at `~/.personal-rag/vectors.db`
- Source data is never copied — only extracted text and embeddings are stored

## Tech stack

Python, FastAPI, HTMX, Jinja2, SQLite, numpy, Ollama, Gemma 3 4B, nomic-embed-text
