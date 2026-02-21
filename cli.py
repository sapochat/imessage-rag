"""CLI entry point for the personal RAG system."""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

from src.chunker import chunk_emails, chunk_imessages
from src.embed import get_embedding
from src.ingest.email import extract_emails
from src.ingest.imessage import extract_messages
from src.query import generate_answer, retrieve
from src.vectordb import get_stats, insert_chunk


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


def cmd_ingest(args: argparse.Namespace) -> None:
    since = parse_since(args.since) if args.since else None

    if args.source == "imessage":
        _ingest_imessage(since)
    elif args.source == "email":
        _ingest_email(since)
    else:
        print(f"Unknown source: {args.source}")
        sys.exit(1)


def _ingest_imessage(since: datetime | None) -> None:
    since_str = since.strftime("%Y-%m-%d") if since else "all time"
    print(f"Extracting iMessages since {since_str}...")

    messages = extract_messages(since=since)
    chunks = chunk_imessages(messages)

    total_chunks = 0
    total_messages = 0
    start = time.time()

    for chunk in chunks:
        total_chunks += 1
        total_messages += chunk.message_count

        # Progress update every 10 chunks
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


def _ingest_email(since: datetime | None) -> None:
    since_str = since.strftime("%Y-%m-%d") if since else "all time"
    print(f"Extracting emails since {since_str}...")

    emails = extract_emails(since=since)
    chunks = chunk_emails(emails)

    total_chunks = 0
    start = time.time()

    for chunk in chunks:
        total_chunks += 1

        if total_chunks % 10 == 0:
            elapsed = time.time() - start
            rate = total_chunks / elapsed if elapsed > 0 else 0
            print(
                f"  Processed: {total_chunks} emails [{rate:.1f} chunks/s]",
                end="\r",
            )

        try:
            embedding = get_embedding(chunk.text)
        except Exception as e:
            print(f"\n  Warning: embedding failed for email "
                  f"({chunk.contact}, {chunk.start_time.strftime('%Y-%m-%d %H:%M')}): {e}")
            continue

        insert_chunk(chunk, embedding)

    elapsed = time.time() - start
    print(f"\nDone. {total_chunks} email chunks in {elapsed:.1f}s")


def cmd_query(args: argparse.Namespace) -> None:
    source = getattr(args, "source", None)
    top_k = getattr(args, "top_k", 5)

    if args.retrieve_only:
        results = retrieve(args.question, top_k=top_k, source=source)
        if not results:
            print("No matching chunks found.")
            return
        for i, r in enumerate(results, 1):
            start = datetime.fromtimestamp(r["start_time"], tz=timezone.utc)
            print(f"\n--- Result {i} (similarity: {r['similarity']:.3f}) ---")
            print(f"Contact: {r['contact']}  |  {start.strftime('%Y-%m-%d %H:%M')}  |  {r['message_count']} msgs")
            print(r["text"][:500])
    else:
        generate_answer(args.question, top_k=top_k, source=source)


def cmd_serve(args: argparse.Namespace) -> None:
    from src.web.app import run
    run(port=args.port)


def cmd_status(args: argparse.Namespace) -> None:
    stats = get_stats()
    if stats["total_chunks"] == 0:
        print("Vector DB is empty. Run 'ingest' first.")
        return
    print(f"Total chunks: {stats['total_chunks']}")
    for source, count in stats["by_source"].items():
        print(f"  {source}: {count}")
    print(f"DB size: {stats['db_size_mb']:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personal RAG — local semantic search over iMessage and email"
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest messages into the vector DB")
    p_ingest.add_argument(
        "--source",
        required=True,
        choices=["imessage", "email"],
        help="Data source to ingest",
    )
    p_ingest.add_argument(
        "--since",
        help="Only ingest messages from this far back (e.g. 30d, 24h)",
    )

    # query
    p_query = sub.add_parser("query", help="Search your messages with natural language")
    p_query.add_argument("question", help="Your question or search query")
    p_query.add_argument(
        "--source",
        choices=["imessage", "email"],
        help="Restrict search to a specific source",
    )
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

    # status
    sub.add_parser("status", help="Show vector DB statistics")

    # serve
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
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
