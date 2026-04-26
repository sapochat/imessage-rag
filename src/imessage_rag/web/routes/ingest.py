"""Ingest routes — start/monitor/cancel ingestion tasks."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from imessage_rag.web.app import templates
from imessage_rag.web.tasks import task_manager

router = APIRouter()


@router.get("/ingest")
async def ingest_page(request: Request):
    tasks = task_manager.all_tasks()
    return templates.TemplateResponse(
        "ingest.html", {"request": request, "tasks": tasks}
    )


@router.post("/ingest/start")
async def ingest_start(
    request: Request,
    since: str = Form(""),
    contact: str = Form(""),
    participants: str = Form(""),
):
    since_val = since.strip() or None
    contact_val = contact.strip() or None
    participants_val = participants.strip() or None

    if contact_val and participants_val:
        tasks = task_manager.all_tasks()
        return templates.TemplateResponse(
            "ingest.html",
            {
                "request": request,
                "tasks": tasks,
                "error": "Use either contact or group participants, not both.",
            },
        )

    if since_val:
        from imessage_rag.cli import parse_since

        try:
            parse_since(since_val)
        except ValueError as exc:
            tasks = task_manager.all_tasks()
            return templates.TemplateResponse(
                "ingest.html",
                {
                    "request": request,
                    "tasks": tasks,
                    "error": f"Invalid since value: {exc}",
                },
            )

    if task_manager.has_running():
        tasks = task_manager.all_tasks()
        return templates.TemplateResponse(
            "ingest.html",
            {"request": request, "tasks": tasks, "error": "An ingest is already running."},
        )

    try:
        task_manager.start_ingest(since_val, contact_val, participants_val)
    except RuntimeError as exc:
        tasks = task_manager.all_tasks()
        return templates.TemplateResponse(
            "ingest.html",
            {"request": request, "tasks": tasks, "error": str(exc)},
        )
    tasks = task_manager.all_tasks()
    return templates.TemplateResponse(
        "ingest.html", {"request": request, "tasks": tasks}
    )


@router.get("/ingest/progress/{task_id}")
async def ingest_progress(request: Request, task_id: str):
    task = task_manager.get(task_id)
    if not task:
        return HTMLResponse("<p>Task not found.</p>")
    return templates.TemplateResponse(
        "partials/ingest_progress.html", {"request": request, "task": task}
    )


@router.post("/ingest/cancel/{task_id}")
async def ingest_cancel(request: Request, task_id: str):
    task = task_manager.get(task_id)
    if task:
        task.request_cancel()
    return templates.TemplateResponse(
        "partials/ingest_progress.html", {"request": request, "task": task}
    )
