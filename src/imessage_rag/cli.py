"""CLI entry point for imessage-rag."""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone


def parse_since(value: str) -> datetime:
    """Parse a relative time like '30d', '7d', '24h' into a UTC datetime."""
    unit = value[-1].lower()
    amount = int(value[:-1])
    now = datetime.now(tz=timezone.utc)
    if unit == "d":
        return now - timedelta(days=amount)
    elif unit == "h":
        return now - timedelta(hours=amount)
    else:
        raise ValueError(f"Unknown time unit '{unit}'. Use 'd' (days) or 'h' (hours).")


def parse_participants(value: str) -> list[str]:
    """Parse a comma-separated participant list."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _print_kv(label: str, value: str) -> None:
    print(f"{label:<18} {value}")


def cmd_ingest(args: argparse.Namespace) -> None:
    since = parse_since(args.since) if args.since else None
    participants = parse_participants(args.participants) if getattr(args, "participants", None) else None

    if getattr(args, "contact", None) and participants:
        print("Use either --contact or --participants, not both.")
        sys.exit(2)

    _ingest_imessage(since, getattr(args, "contact", None), participants)


def _ingest_imessage(
    since: datetime | None,
    contact: str | None = None,
    participants: list[str] | None = None,
) -> None:
    from imessage_rag.chunker import chunk_imessages
    from imessage_rag.embed import get_embedding
    from imessage_rag.ingest import extract_messages
    from imessage_rag.vectordb import insert_chunk

    since_str = since.strftime("%Y-%m-%d") if since else "all time"
    if participants:
        target = f" for participants {', '.join(participants)}"
    else:
        target = f" for {contact}" if contact else ""
    print(f"Extracting iMessages since {since_str}{target}...")

    messages = extract_messages(since=since, contact=contact, participants=participants)
    chunks = chunk_imessages(messages)

    total_chunks = 0
    total_messages = 0
    start = time.time()

    for chunk in chunks:
        total_chunks += 1
        total_messages += chunk.message_count

        if total_chunks % 10 == 0:
            elapsed = time.time() - start
            rate = total_chunks / elapsed if elapsed > 0 else 0
            print(
                f"  Chunked: {total_chunks} chunks ({total_messages} messages) "
                f"[{rate:.1f} chunks/s]",
                end="\r",
            )

        try:
            embedding = get_embedding(chunk.text)
        except Exception as e:
            print(f"\n  Warning: embedding failed for chunk ({chunk.contact}, "
                  f"{chunk.start_time.strftime('%Y-%m-%d %H:%M')}): {e}")
            continue

        insert_chunk(chunk, embedding)

    elapsed = time.time() - start
    print(f"\nDone. {total_chunks} chunks from {total_messages} messages "
          f"in {elapsed:.1f}s")


def cmd_query(args: argparse.Namespace) -> None:
    from imessage_rag.query import generate_answer, retrieve

    top_k = getattr(args, "top_k", 5)

    if args.retrieve_only:
        results = retrieve(args.question, top_k=top_k)
        if not results:
            print("No matching chunks found.")
            return
        for i, r in enumerate(results, 1):
            start = datetime.fromtimestamp(r["start_time"], tz=timezone.utc)
            print(f"\n--- Result {i} (similarity: {r['similarity']:.3f}) ---")
            print(f"Contact: {r['contact']}  |  {start.strftime('%Y-%m-%d %H:%M')}  |  {r['message_count']} msgs")
            print(r["text"][:500])
    else:
        generate_answer(args.question, top_k=top_k)


def cmd_serve(args: argparse.Namespace) -> None:
    from imessage_rag.web.app import run
    run(port=args.port)


def cmd_status(args: argparse.Namespace) -> None:
    from imessage_rag.vectordb import get_stats

    stats = get_stats()
    if stats["total_chunks"] == 0:
        print("Vector DB is empty. Run 'ingest' first.")
        return
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"DB size: {stats['db_size_mb']:.2f} MB")


def cmd_config(args: argparse.Namespace) -> None:
    from imessage_rag.config import EMBED_DIMENSIONS, EMBED_MODEL, OLLAMA_URL, VECTOR_DB
    from imessage_rag.settings import get_generation_backend, get_generation_model

    _print_kv("Vector DB", str(VECTOR_DB))
    _print_kv("DB exists", "yes" if VECTOR_DB.exists() else "no")
    _print_kv("Embed model", EMBED_MODEL)
    _print_kv("Embed dims", str(EMBED_DIMENSIONS or 768))
    _print_kv("Generation", f"{get_generation_backend()} / {get_generation_model()}")
    _print_kv("Ollama URL", OLLAMA_URL)


def cmd_reset_db(args: argparse.Namespace) -> None:
    from imessage_rag.config import VECTOR_DB

    db_path = VECTOR_DB
    if not db_path.exists():
        print(f"Vector DB does not exist: {db_path}")
        return

    if not args.yes:
        reply = input(f"Delete vector DB at {db_path}? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Cancelled.")
            return

    db_path.unlink()
    print(f"Deleted {db_path}")


def cmd_doctor(args: argparse.Namespace) -> None:
    import requests

    from imessage_rag.config import EMBED_DIMENSIONS, EMBED_MODEL, OLLAMA_URL, VECTOR_DB
    from imessage_rag.settings import get_generation_backend, get_generation_model

    print("imessage-rag doctor")
    print()
    _print_kv("Vector DB", str(VECTOR_DB))
    _print_kv("DB exists", "yes" if VECTOR_DB.exists() else "no")
    _print_kv("Embed model", EMBED_MODEL)
    _print_kv("Embed dims", str(EMBED_DIMENSIONS or 768))
    _print_kv("Generation", f"{get_generation_backend()} / {get_generation_model()}")
    _print_kv("Ollama URL", OLLAMA_URL)
    print()

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        print("Ollama: online")
        print(f"Embed model loaded: {'yes' if any(EMBED_MODEL in m for m in models) else 'no'}")
        print(
            f"Generation model loaded: "
            f"{'yes' if any(get_generation_model() in m for m in models) else 'no'}"
        )
    except Exception as exc:
        print(f"Ollama: error - {exc}")
        return

    print()
    print("Next steps")
    if not VECTOR_DB.exists():
        print("  1. Reset or create a fresh DB if needed: `imessage-rag reset-db --yes`")
        print("  2. Ingest iMessages: `imessage-rag ingest --contact +15551234567`")
    else:
        print("  1. Check current data: `imessage-rag status`")
        print("  2. Query it: `imessage-rag query \"your question\"`")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="imessage-rag — local semantic search over iMessage history"
    )
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest messages into the vector DB")
    p_ingest.add_argument(
        "--since",
        help="Only ingest messages from this far back (e.g. 30d, 24h)",
    )
    p_ingest.add_argument(
        "--contact",
        help="Ingest 1:1 chats with a single handle (phone or email).",
    )
    p_ingest.add_argument(
        "--participants",
        help="Ingest one exact group thread as a comma-separated participant set.",
    )

    p_query = sub.add_parser("query", help="Search your messages with natural language")
    p_query.add_argument("question", help="Your question or search query")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of chunks to retrieve (default: 5)",
    )
    p_query.add_argument(
        "--retrieve-only",
        action="store_true",
        dest="retrieve_only",
        help="Show raw retrieved chunks without LLM generation",
    )

    sub.add_parser("status", help="Show vector DB statistics")
    sub.add_parser("config", help="Show the active DB/model configuration")
    sub.add_parser("doctor", help="Check Ollama connectivity and active config")

    p_reset = sub.add_parser("reset-db", help="Delete the current vector DB")
    p_reset.add_argument(
        "--yes",
        action="store_true",
        help="Delete without prompting",
    )

    p_serve = sub.add_parser("serve", help="Start the web UI")
    p_serve.add_argument(
        "--port",
        type=int,
        default=5391,
        help="Port to listen on (default: 5391)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "reset-db":
        cmd_reset_db(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
