"""CLI entry point for imessage-rag."""

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _imessage_db_readable(db_path) -> tuple[bool, str | None]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return False, type(exc).__name__
    return True, None


def _embed_and_insert_batch(chunks: list, warn=None) -> tuple[int, int, int]:
    """Embed and store a chunk batch. Returns (inserted, skipped, messages_inserted)."""
    if not chunks:
        return 0, 0, 0

    from imessage_rag.embed import EmbeddingConfigError, get_embedding, get_embeddings
    from imessage_rag.vectordb import insert_chunk, insert_chunks

    try:
        embeddings = get_embeddings(
            [chunk.embedding_text or chunk.text for chunk in chunks]
        )
        insert_chunks(chunks, embeddings)
        return len(chunks), 0, sum(chunk.message_count for chunk in chunks)
    except EmbeddingConfigError:
        raise
    except Exception as batch_error:
        if warn is not None:
            warn(
                "batch embedding failed "
                f"({type(batch_error).__name__}); falling back to single chunks"
            )

    inserted = 0
    skipped = 0
    inserted_messages = 0
    for chunk in chunks:
        try:
            embedding = get_embedding(chunk.embedding_text or chunk.text)
            insert_chunk(chunk, embedding)
            inserted += 1
            inserted_messages += chunk.message_count
        except EmbeddingConfigError:
            raise
        except Exception as exc:
            skipped += 1
            if warn is not None:
                warn(
                    "skipped one chunk after embedding failure "
                    f"({type(exc).__name__}, start={chunk.start_time.strftime('%Y-%m-%d %H:%M')})"
                )
    return inserted, skipped, inserted_messages


def _embed_and_insert_batches(
    batches: list[list],
    workers: int,
    warn=None,
) -> tuple[int, int, int]:
    """Embed and store multiple batches, optionally concurrently."""
    batches = [batch for batch in batches if batch]
    if not batches:
        return 0, 0, 0

    workers = max(1, min(workers, len(batches)))
    if workers == 1:
        totals = [_embed_and_insert_batch(batch, warn=warn) for batch in batches]
    else:
        totals = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_embed_and_insert_batch, batch, warn)
                for batch in batches
            ]
            for future in as_completed(futures):
                totals.append(future.result())

    inserted = sum(total[0] for total in totals)
    skipped = sum(total[1] for total in totals)
    inserted_messages = sum(total[2] for total in totals)
    return inserted, skipped, inserted_messages


def cmd_ingest(args: argparse.Namespace) -> None:
    since = parse_since(args.since) if args.since else None
    participants = parse_participants(args.participants) if getattr(args, "participants", None) else None

    if getattr(args, "contact", None) and participants:
        print("Use either --contact or --participants, not both.")
        sys.exit(2)

    _ingest_imessage(
        since,
        getattr(args, "contact", None),
        participants,
        reindex=getattr(args, "reindex", False),
    )


def _ingest_imessage(
    since: datetime | None,
    contact: str | None = None,
    participants: list[str] | None = None,
    reindex: bool = False,
) -> None:
    from imessage_rag.chunker import chunk_imessages
    from imessage_rag.config import EMBED_BATCH_SIZE, EMBED_WORKERS
    from imessage_rag.embed import EmbeddingConfigError
    from imessage_rag.ingest import extract_messages
    from imessage_rag.vectordb import filter_new_chunks

    since_str = since.strftime("%Y-%m-%d") if since else "all time"
    if participants:
        target = f" for participants {', '.join(participants)}"
    else:
        target = f" for {contact}" if contact else ""
    print(f"Extracting iMessages since {since_str}{target}...")

    messages = extract_messages(since=since, contact=contact, participants=participants)
    chunks = chunk_imessages(messages)
    batch_size = max(1, EMBED_BATCH_SIZE)
    workers = max(1, EMBED_WORKERS)
    group_size = batch_size * workers

    total_chunks = 0
    total_messages = 0
    inserted_chunks = 0
    skipped_chunks = 0
    skipped_existing = 0
    start = time.time()
    batch = []
    batch_group = []

    def warn(message: str) -> None:
        print(f"\n  Warning: {message}")

    def flush() -> None:
        nonlocal inserted_chunks, skipped_chunks, skipped_existing, batch_group
        candidate_batches = batch_group
        batch_group = []
        if not reindex:
            filtered_batches = []
            for candidate_batch in candidate_batches:
                new_batch = filter_new_chunks(candidate_batch)
                skipped_existing += len(candidate_batch) - len(new_batch)
                if new_batch:
                    filtered_batches.append(new_batch)
            candidate_batches = filtered_batches

        inserted, skipped, _ = _embed_and_insert_batches(
            candidate_batches,
            workers=workers,
            warn=warn,
        )
        inserted_chunks += inserted
        skipped_chunks += skipped

        elapsed = time.time() - start
        rate = inserted_chunks / elapsed if elapsed > 0 else 0
        print(
            f"  Embedded: {inserted_chunks}/{total_chunks} chunks "
            f"({total_messages} messages, skipped {skipped_chunks}, existing {skipped_existing}) "
            f"[{rate:.1f} chunks/s]",
            end="\r",
        )

    for chunk in chunks:
        total_chunks += 1
        total_messages += chunk.message_count
        batch.append(chunk)

        if len(batch) >= batch_size:
            batch_group.append(batch)
            batch = []

        if sum(len(group_batch) for group_batch in batch_group) >= group_size:
            try:
                flush()
            except EmbeddingConfigError as exc:
                print(f"\nEmbedding configuration error: {exc}")
                sys.exit(2)

    if batch:
        batch_group.append(batch)
    if batch_group:
        try:
            flush()
        except EmbeddingConfigError as exc:
            print(f"\nEmbedding configuration error: {exc}")
            sys.exit(2)

    elapsed = time.time() - start
    print(
        f"\nDone. {inserted_chunks}/{total_chunks} chunks from {total_messages} "
        f"messages in {elapsed:.1f}s (skipped {skipped_chunks}, existing {skipped_existing})"
    )


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
            print(
                f"\n--- Result {i} "
                f"(similarity: {r['similarity']:.3f}, "
                f"retrieval: {r.get('retrieval', 'unknown')}) ---"
            )
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
    from imessage_rag.config import CONTACTS_ENABLED, CHUNK_MAX_MESSAGES, EMBED_BATCH_SIZE, EMBED_DIMENSIONS, EMBED_MAX_CHARS, EMBED_MODEL, EMBED_PROFILE, EMBED_WORKERS, IMESSAGE_DB, OLLAMA_URL, VECTOR_DB
    from imessage_rag.settings import get_generation_backend, get_generation_model

    readable, _ = _imessage_db_readable(IMESSAGE_DB)
    _print_kv("iMessage DB", str(IMESSAGE_DB))
    _print_kv("iMessage read", "yes" if readable else "no")
    _print_kv("Vector DB", str(VECTOR_DB))
    _print_kv("DB exists", "yes" if VECTOR_DB.exists() else "no")
    _print_kv("Contacts", "enabled" if CONTACTS_ENABLED else "disabled")
    _print_kv("Embed profile", EMBED_PROFILE)
    _print_kv("Embed model", EMBED_MODEL)
    _print_kv("Embed dims", str(EMBED_DIMENSIONS or 768))
    _print_kv("Embed batch", str(EMBED_BATCH_SIZE))
    _print_kv("Embed workers", str(EMBED_WORKERS))
    _print_kv("Embed max chars", str(EMBED_MAX_CHARS))
    _print_kv("Chunk max msgs", str(CHUNK_MAX_MESSAGES))
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

    from imessage_rag.config import CHUNK_MAX_MESSAGES, EMBED_BATCH_SIZE, EMBED_DIMENSIONS, EMBED_MAX_CHARS, EMBED_MODEL, EMBED_PROFILE, EMBED_WORKERS, IMESSAGE_DB, OLLAMA_URL, VECTOR_DB
    from imessage_rag.contacts import load_contacts
    from imessage_rag.settings import get_generation_backend, get_generation_model

    print("imessage-rag doctor")
    print()
    readable, read_error = _imessage_db_readable(IMESSAGE_DB)
    _print_kv("iMessage DB", str(IMESSAGE_DB))
    _print_kv(
        "iMessage read",
        "yes" if readable else f"no ({read_error or 'unknown error'})",
    )
    _print_kv("Vector DB", str(VECTOR_DB))
    _print_kv("DB exists", "yes" if VECTOR_DB.exists() else "no")
    _print_kv("Embed profile", EMBED_PROFILE)
    _print_kv("Embed model", EMBED_MODEL)
    _print_kv("Embed dims", str(EMBED_DIMENSIONS or 768))
    _print_kv("Embed batch", str(EMBED_BATCH_SIZE))
    _print_kv("Embed workers", str(EMBED_WORKERS))
    _print_kv("Embed max chars", str(EMBED_MAX_CHARS))
    _print_kv("Chunk max msgs", str(CHUNK_MAX_MESSAGES))
    _print_kv("Generation", f"{get_generation_backend()} / {get_generation_model()}")
    _print_kv("Ollama URL", OLLAMA_URL)
    resolver = load_contacts()
    _print_kv("Contacts", f"{resolver.contact_count} contacts / {resolver.handle_count} handles")
    if resolver.errors:
        _print_kv("Contacts errors", str(len(resolver.errors)))
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


def cmd_contacts(args: argparse.Namespace) -> None:
    from imessage_rag.contacts import default_contact_db_paths, load_contacts

    paths = [path for path in default_contact_db_paths() if path.exists()]
    resolver = load_contacts(paths)
    print("Contacts resolution")
    _print_kv("DBs found", str(len(paths)))
    _print_kv("Contacts", str(resolver.contact_count))
    _print_kv("Handles", str(resolver.handle_count))
    _print_kv("Errors", str(len(resolver.errors)))
    if resolver.errors:
        print()
        print("Errors are path/type only; no contact values are printed:")
        for error in resolver.errors:
            print(f"  {error}")


def cmd_embed_profile(args: argparse.Namespace) -> None:
    from imessage_rag import settings
    from imessage_rag.config import (
        EMBED_BATCH_SIZE,
        EMBED_DIMENSIONS,
        EMBED_MAX_CHARS,
        EMBED_MODEL,
        EMBED_PROFILE,
        EMBED_PROFILES,
        EMBED_WORKERS,
    )

    if args.profile == "show":
        _print_kv("Active profile", EMBED_PROFILE)
        _print_kv("Model", EMBED_MODEL)
        _print_kv("Dimensions", str(EMBED_DIMENSIONS))
        _print_kv("Batch", str(EMBED_BATCH_SIZE))
        _print_kv("Workers", str(EMBED_WORKERS))
        _print_kv("Max chars", str(EMBED_MAX_CHARS))
        return

    settings.save({"embed_profile": args.profile})

    if args.profile == "custom":
        print("Embedding profile set to custom. EMBED_MODEL/EMBED_DIMENSIONS env values will be used.")
        return

    profile = EMBED_PROFILES[args.profile]
    print(f"Embedding profile set to {args.profile}.")
    _print_kv("Model", profile["model"])
    _print_kv("Dimensions", str(profile["dimensions"]))
    _print_kv("Batch", str(profile["batch_size"]))
    _print_kv("Workers", str(profile["workers"]))
    _print_kv("Max chars", str(profile["max_chars"]))
    print()
    print("Embedding profile changes require a fresh vector DB and re-ingest.")
    print("Run: imessage-rag reset-db --yes && imessage-rag ingest --reindex")


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
    p_ingest.add_argument(
        "--reindex",
        action="store_true",
        help="Re-embed chunks even if they already exist in the vector DB.",
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
    sub.add_parser("contacts", help="Check local Contacts resolution without printing contact data")

    p_profile = sub.add_parser("embed-profile", help="Show or switch embedding profiles")
    p_profile.add_argument(
        "profile",
        choices=["show", "fast", "full", "custom"],
        help="fast=qwen 0.6b, full=qwen 8b, custom=env-controlled",
    )

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
    elif args.command == "contacts":
        cmd_contacts(args)
    elif args.command == "embed-profile":
        cmd_embed_profile(args)
    elif args.command == "reset-db":
        cmd_reset_db(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
