"""Background task management for long-running ingestions."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class IngestTask:
    id: str
    source: str
    since: str | None
    status: TaskStatus = TaskStatus.PENDING
    chunks_processed: int = 0
    messages_processed: int = 0
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "source": self.source,
                "since": self.since,
                "status": self.status.value,
                "chunks_processed": self.chunks_processed,
                "messages_processed": self.messages_processed,
                "error": self.error,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            }


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, IngestTask] = {}
        self._lock = threading.Lock()

    def get(self, task_id: str) -> IngestTask | None:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[IngestTask]:
        return list(self._tasks.values())

    def has_running(self, source: str) -> bool:
        return any(
            t.source == source and t.status == TaskStatus.RUNNING
            for t in self._tasks.values()
        )

    def start_ingest(self, source: str, since: str | None) -> IngestTask:
        task = IngestTask(id=uuid.uuid4().hex[:8], source=source, since=since)
        with self._lock:
            self._tasks[task.id] = task

        thread = threading.Thread(
            target=self._run_ingest, args=(task,), daemon=True
        )
        thread.start()
        return task

    def _run_ingest(self, task: IngestTask) -> None:
        from src.chunker import chunk_emails, chunk_imessages
        from src.embed import get_embedding
        from src.ingest.email import extract_emails
        from src.ingest.imessage import extract_messages
        from src.vectordb import insert_chunk

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(tz=timezone.utc)

        try:
            since_dt = None
            if task.since:
                from cli import parse_since
                since_dt = parse_since(task.since)

            if task.source == "imessage":
                messages = extract_messages(since=since_dt)
                chunks = chunk_imessages(messages)
            elif task.source == "email":
                emails = extract_emails(since=since_dt)
                chunks = chunk_emails(emails)
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Unknown source '{task.source}'"
                task.finished_at = datetime.now(tz=timezone.utc)
                return

            for chunk in chunks:
                if task.cancel_requested:
                    task.status = TaskStatus.CANCELLED
                    task.finished_at = datetime.now(tz=timezone.utc)
                    return

                try:
                    embedding = get_embedding(chunk.text)
                except Exception:
                    continue

                insert_chunk(chunk, embedding)

                with task._lock:
                    task.chunks_processed += 1
                    task.messages_processed += chunk.message_count

            task.status = TaskStatus.DONE
            task.finished_at = datetime.now(tz=timezone.utc)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.finished_at = datetime.now(tz=timezone.utc)


# Singleton instance
task_manager = TaskManager()
