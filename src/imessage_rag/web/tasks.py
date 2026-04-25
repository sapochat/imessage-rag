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
    since: str | None
    contact: str | None = None
    participants: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    chunks_processed: int = 0
    messages_processed: int = 0
    chunks_existing: int = 0
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
                "since": self.since,
                "contact": self.contact,
                "participants": self.participants,
                "status": self.status.value,
                "chunks_processed": self.chunks_processed,
                "messages_processed": self.messages_processed,
                "chunks_existing": self.chunks_existing,
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

    def has_running(self) -> bool:
        return any(
            t.status == TaskStatus.RUNNING for t in self._tasks.values()
        )

    def start_ingest(
        self,
        since: str | None,
        contact: str | None = None,
        participants: str | None = None,
    ) -> IngestTask:
        task = IngestTask(
            id=uuid.uuid4().hex[:8],
            since=since,
            contact=contact,
            participants=participants,
        )
        with self._lock:
            self._tasks[task.id] = task

        thread = threading.Thread(
            target=self._run_ingest, args=(task,), daemon=True
        )
        thread.start()
        return task

    def _run_ingest(self, task: IngestTask) -> None:
        from imessage_rag.chunker import chunk_imessages
        from imessage_rag.cli import _embed_and_insert_batches, parse_participants, parse_since
        from imessage_rag.config import EMBED_BATCH_SIZE, EMBED_WORKERS
        from imessage_rag.ingest import extract_messages
        from imessage_rag.vectordb import filter_new_chunks

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(tz=timezone.utc)

        try:
            since_dt = parse_since(task.since) if task.since else None
            participant_list = parse_participants(task.participants) if task.participants else None

            messages = extract_messages(
                since=since_dt,
                contact=task.contact,
                participants=participant_list,
            )
            chunks = chunk_imessages(messages)
            batch_size = max(1, EMBED_BATCH_SIZE)
            workers = max(1, EMBED_WORKERS)
            group_size = batch_size * workers
            batch = []
            batch_group = []

            def flush_batch() -> None:
                nonlocal batch_group
                filtered_batches = []
                existing = 0
                for candidate_batch in batch_group:
                    new_batch = filter_new_chunks(candidate_batch)
                    existing += len(candidate_batch) - len(new_batch)
                    if new_batch:
                        filtered_batches.append(new_batch)
                batch_group = []

                inserted, _, inserted_messages = _embed_and_insert_batches(
                    filtered_batches,
                    workers=workers,
                )
                with task._lock:
                    task.chunks_processed += inserted
                    task.messages_processed += inserted_messages
                    task.chunks_existing += existing
                batch = []

            for chunk in chunks:
                if task.cancel_requested:
                    task.status = TaskStatus.CANCELLED
                    task.finished_at = datetime.now(tz=timezone.utc)
                    return

                batch.append(chunk)
                if len(batch) < batch_size:
                    continue

                batch_group.append(batch)
                batch = []

                if sum(len(group_batch) for group_batch in batch_group) >= group_size:
                    flush_batch()

            if batch:
                batch_group.append(batch)
            if batch_group and not task.cancel_requested:
                flush_batch()

            task.status = TaskStatus.DONE
            task.finished_at = datetime.now(tz=timezone.utc)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.finished_at = datetime.now(tz=timezone.utc)


task_manager = TaskManager()
