# imessage-rag

Local-only semantic search and Q&A over your iMessage history. Everything runs on your machine — no data ever leaves. macOS on Apple Silicon only.

## What it does

1. **Ingests** your iMessages from `~/Library/Messages/chat.db` without copying the raw data.
2. **Resolves Contacts** locally so chunks use names instead of raw handles when macOS allows Contacts access.
3. **Chunks** messages into conversation windows by contact and time.
4. **Embeds** each chunk locally with Ollama and stores vectors in SQLite.
5. **Retrieves + generates** — type a natural-language question, get an answer grounded in the actual messages, with sources.

Three ways to use it:

- **CLI**: `imessage-rag query "what restaurant did sarah recommend?"`
- **Web UI**: `imessage-rag serve` opens a local chat interface with streaming responses and multi-turn follow-ups.
- **MCP server**: expose your iMessage index as a tool to LM Studio or any MCP client.

## Requirements

- macOS on Apple Silicon
- Python 3.11+
- [Ollama](https://ollama.com) running locally, with:
  ```bash
  ollama pull nomic-embed-text
  ollama pull gemma3:4b
  ```
- Terminal must have **Full Disk Access** (System Settings → Privacy & Security → Full Disk Access) so it can read `chat.db`.

## Install

```bash
git clone https://github.com/<your-handle>/imessage-rag.git
cd imessage-rag
./scripts/setup.sh
source .venv/bin/activate
```

That creates the virtualenv, installs the package in editable mode, and puts `imessage-rag` and `imessage-rag-mcp` on your PATH.

## Quickstart

```bash
# 1. Confirm Ollama and models are reachable
imessage-rag doctor

# 2. Confirm local Contacts can be read without printing contact data
imessage-rag contacts

# 3. Ingest the last 30 days of messages with one person
imessage-rag ingest --contact +15551234567 --since 30d

# 4. Check what loaded
imessage-rag status

# 5. Ask a question
imessage-rag query "what did we decide about the trip?"
```

## CLI reference

| Command | What it does |
|---|---|
| `ingest --contact <handle> [--since 30d]` | Ingest 1:1 messages with one contact. Handle can be phone or email. |
| `ingest --participants +15551234567,+15557654321` | Ingest one exact group thread (your own number is implicit). |
| `ingest --since 30d` | Ingest all contacts, last 30 days. Omit `--since` for full history. |
| `query "<question>"` | Retrieve + generate an answer. |
| `query "<question>" --retrieve-only --top-k 10` | Show raw matching chunks with no LLM. |
| `status` | Chunk count and DB size. |
| `doctor` | Check Ollama, models, and active config. |
| `contacts` | Check local Contacts resolution counts without printing contact data. |
| `embed-profile show` | Show the active embedding profile/model/dimensions. |
| `embed-profile fast` | Switch to `nomic-embed-text` for faster lower-cost ingestion. |
| `embed-profile full` | Switch to `qwen3-embedding:8b` for higher-quality retrieval. |
| `config` | Print active settings and paths. |
| `reset-db --yes` | Delete the vector DB and start over. |
| `serve [--port 5391]` | Start the web UI on localhost with an auth token. |

## Full-history first run

For an all-time iMessage ingest, run the preflights first:

```bash
imessage-rag doctor
imessage-rag contacts
imessage-rag status
```

Then start from a fresh DB if you do not need the current index:

```bash
imessage-rag reset-db --yes
imessage-rag ingest
```

Keep the Terminal session open. Full history can take hours because each chunk is embedded locally through Ollama. If `imessage-rag contacts` reports zero contacts or read errors, grant Terminal Full Disk Access and Contacts access in macOS privacy settings, then re-run the command before ingesting.

## Web UI

```bash
imessage-rag serve
# Prints a URL with a generated auth token — open it
```

Three tabs:

- **Query** — multi-turn chat with streaming. Retrieved chunks accumulate across turns so follow-ups see the same raw context.
- **Ingest** — kick off ingest jobs, watch progress, cancel.
- **Status** — chunk counts, DB size, Ollama health.

## MCP server (for LM Studio and friends)

Expose your iMessage index as an MCP tool. Three tools are provided:

- `search_messages` — semantic search
- `get_chunk` — fetch a chunk by ID
- `get_stats` — DB stats

LM Studio `mcp.json` example:

```json
{
  "mcpServers": {
    "imessage-rag": {
      "command": "/absolute/path/to/imessage-rag/.venv/bin/imessage-rag-mcp",
      "args": []
    }
  }
}
```

## Configuration

All config is environment variables. Defaults work out of the box. See [`.env.example`](.env.example) for the full list. The most common overrides:

| Variable | Default | Purpose |
|---|---|---|
| `EMBED_PROFILE` | `custom` | `fast` uses `nomic-embed-text`; `full` uses `qwen3-embedding:8b`; `custom` uses explicit env values. Prefer `imessage-rag embed-profile fast|full`. |
| `EMBED_MODEL` | `nomic-embed-text` | Custom embedding model used when `EMBED_PROFILE=custom`. |
| `EMBED_DIMENSIONS` | `768` | Custom embedding dimensions used when `EMBED_PROFILE=custom`. |
| `GENERATION_MODEL` | `gemma3:4b` | Any local Ollama chat model. |
| `VECTOR_DB` | `~/.imessage-rag/vectors.db` | Where embeddings live. |
| `CONTACTS_ENABLED` | `1` | Resolve handles through local macOS Contacts during ingest. |
| `CONTACTS_DB` | auto-discover | Optional explicit AddressBook `.abcddb` path. |
| `CHUNK_WINDOW_HOURS` | `4` | How to group messages into conversation chunks. |
| `EMBED_BATCH_SIZE` | `32` | Number of chunks to send to Ollama per embedding request. Lower this if Ollama runs out of memory. |
| `EMBED_WORKERS` | `2` | Number of concurrent embedding requests. Raise cautiously; local models often get slower if overloaded. |
| `EMBED_MAX_CHARS` | `12000` | Max characters from each chunk sent to the embedding model. Lower values speed ingest but preserve less context. |

Switching `EMBED_PROFILE`, `EMBED_MODEL`, or `EMBED_DIMENSIONS` means existing vectors are incompatible — `imessage-rag reset-db --yes` and re-ingest.

## Privacy

This project exists because your iMessages are private. By design:

- Zero external API calls. `OLLAMA_URL` is validated to resolve to loopback at startup.
- No telemetry, no analytics.
- Source data is never copied — only extracted text and embeddings are stored.
- Contacts are read locally and used only to label chunks before local embedding.
- The vector DB lives under your home directory and is git-ignored by default.

## Caveats

- `chat.db` is Apple's internal SQLite schema. If Apple changes it in a macOS update, ingestion may break. The schema has been stable since ~Big Sur, but this is a real risk of the tool.
- Retrieval quality depends heavily on the embedding model. `nomic-embed-text` is the default for speed; `qwen3-embedding:8b` is noticeably better on some queries but takes ~9GB and 3-5× longer to ingest.
- Contact resolution depends on macOS Contacts/AddressBook permissions. If access is denied, ingestion falls back to raw phone numbers and email addresses.

## License

MIT — see [LICENSE](LICENSE).
