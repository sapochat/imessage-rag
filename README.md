# Personal RAG

Local-only RAG system that indexes your iMessage and Apple Mail for semantic search and AI-powered Q&A. Everything runs on localhost — no data ever leaves your machine.

## How it works

1. **Ingest** — Streams messages from iMessage's SQLite DB and parses Apple Mail `.emlx` files
2. **Chunk** — Groups iMessages by conversation thread + 4-hour time window; one chunk per email with header metadata
3. **Embed** — Generates vectors via Ollama (`qwen3-embedding:8b` by default) and stores in a local SQLite DB
4. **Query** — Semantic search over embeddings, then streams an answer from Gemma 3 4B
5. **Multi-turn chat** — Follow-up questions carry full context: prior chunks are accumulated across turns (capped at 20) so the model always sees the raw conversations, not just previous answers

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- [Ollama](https://ollama.com) running locally with two models pulled:
  ```
  ollama pull qwen3-embedding:8b
  ollama pull gemma3:4b
  ```
- Terminal with **Full Disk Access** (System Settings > Privacy & Security) to read `chat.db` and Mail directories

## Setup

### Fastest setup

From the repo root:

```bash
./scripts/setup.sh
source .venv/bin/activate
```

That creates the virtual environment, installs dependencies, and installs the
`personal-rag` and `personal-rag-mcp` commands into `.venv/bin/`.

### Manual setup

```bash
git clone https://github.com/vishoo7/local-private-rag.git
cd local-private-rag
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

### Optional `.env`

You do not need a `.env` file to get started. The defaults are fine.

If you want to pin the DB path or embedding size, create a `.env` file in the
repo root:

```env
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=qwen3-embedding:8b
# Optional: request a smaller output dimension from qwen3-embedding:8b
# EMBED_DIMENSIONS=1024
GENERATION_MODEL=gemma3:4b
VECTOR_DB=~/.personal-rag/vectors.db
CHUNK_WINDOW_HOURS=4
```

## First-time run

### 1. Activate the virtual environment

Run this every time you open a new terminal:

```bash
cd /path/to/local-private-rag
source .venv/bin/activate
```

### 2. Make sure Ollama is ready

```bash
ollama pull qwen3-embedding:8b
ollama pull gemma3:4b
personal-rag doctor
```

If `personal-rag doctor` says Ollama is offline, start Ollama first and rerun
the command.

### 3. Start with a clean database

Use this when you want to rebuild from scratch:

```bash
personal-rag reset-db --yes
```

### 4. Ingest messages

One person only:

```bash
personal-rag ingest --source imessage --contact +15551234567
```

One exact group thread:

```bash
personal-rag ingest --source imessage --participants +15551234567,+15557654321
```

Important:

- `--contact` means only 1:1 chats with that number
- `--participants` means the exact remote participant set for a group chat
- your own number is implicit, so do not include your own number in `--participants`

### 5. Check what was loaded

```bash
personal-rag status
```

### 6. Ask questions

```bash
personal-rag query "What did they say about dinner?"
```

Show raw matching chunks only:

```bash
personal-rag query "dinner plans" --retrieve-only --top-k 10
```

## Common commands

```bash
# Show active DB/model config
personal-rag config

# Check whether the local services and models are ready
personal-rag doctor

# Reset the DB and start fresh
personal-rag reset-db --yes

# Recent-only ingest
personal-rag ingest --source imessage --since 30d
personal-rag ingest --source email --since 30d

# Full historical ingest
personal-rag ingest --source imessage
personal-rag ingest --source email
```

### Web UI

```bash
personal-rag serve
# Prints a URL with an auth token — open it in your browser
# e.g. http://127.0.0.1:5391?token=<generated-token>
```

Three pages:

- **Query** — Multi-turn chat interface with streaming responses. Ask a question, then follow up naturally ("tell me more about that", "what else did they say"). Retrieved chunks accumulate across turns so you can drill deeper without losing context.
- **Ingest** — Start ingestion jobs, watch progress live, cancel if needed.
- **Status** — Chunk counts, DB size, Ollama connectivity and model availability.

### MCP for LM Studio

This repo also includes a minimal stdio MCP server so other local LLM apps can
query the vector DB without using this app's built-in answer generation.

The MCP server exposes three tools:

- `search_messages` — semantic search over the local vector DB
- `get_chunk` — fetch a full chunk by ID
- `get_stats` — basic DB counts and size

### LM Studio setup

Use the virtualenv binary directly. This avoids PATH problems.

Example LM Studio `mcp.json` entry:

```json
{
  "mcpServers": {
    "personal-rag": {
      "command": "/absolute/path/to/local-private-rag/.venv/bin/personal-rag-mcp",
      "args": []
    }
  }
}
```

If that still gives import errors, use Python directly:

```json
{
  "mcpServers": {
    "personal-rag": {
      "command": "/absolute/path/to/local-private-rag/.venv/bin/python3",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/local-private-rag"
      }
    }
  }
}
```

Before debugging LM Studio, you can test the MCP command yourself:

```bash
source .venv/bin/activate
.venv/bin/personal-rag-mcp
```

If that command starts and just waits, it is working.

LM Studio can then call `search_messages`, inspect the returned chunk IDs, and
use `get_chunk` when it needs the full conversation text.

## If something breaks

Try these in order:

```bash
source .venv/bin/activate
personal-rag doctor
personal-rag config
personal-rag reset-db --yes
```

If the CLI command itself is broken, use the module form from the repo root:

```bash
PYTHONPATH=$PWD python3 -m src.cli doctor
PYTHONPATH=$PWD python3 -m src.cli ingest --source imessage --contact +15551234567
```

## Architecture

```
iMessage (chat.db) ─┐
                    ├─ extract → chunk → embed (Ollama) → SQLite vector DB
Apple Mail (.emlx) ─┘                                          │
                                                               ▼
                      CLI / Web UI ── query → semantic search + Gemma 3 → answer
```

All processing is local. Network calls go only to localhost — Ollama for embeddings, and optionally an OpenAI-compatible proxy (e.g. maple.ai at `127.0.0.1:8080`) for generation.

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

- Zero external API calls — everything runs on localhost (Ollama + optional local proxy)
- No telemetry, no analytics, no cloud services
- Vector DB stored locally at `~/.personal-rag/vectors.db`
- Source data is never copied — only extracted text and embeddings are stored

## Caveats

The ingestors read directly from Apple's internal data formats — iMessage's `chat.db` SQLite schema and Apple Mail's `.emlx` file structure. These are undocumented, private formats that Apple can change in any macOS update. If ingestion breaks after an OS upgrade, the likely culprit is a schema or format change in one of these sources.

If you use `--contact` for iMessage ingestion, only 1:1 chats with that handle are added during that ingest run. If you use `--participants`, only chats whose remote participant set exactly matches the provided handles are added. Existing chunks already stored in the vector DB are not deleted, so use a fresh `VECTOR_DB` path or rebuild the DB if you want search limited strictly to one contact or one group thread.

If you change `EMBED_MODEL` or `EMBED_DIMENSIONS`, rebuild or switch to a fresh `VECTOR_DB`. Retrieval only works when stored chunk embeddings and query embeddings were generated with the same embedding configuration.

## Tech stack

Python, FastAPI, HTMX, Jinja2, SQLite, numpy, Ollama, Gemma 3 4B, qwen3-embedding:8b
