"""Ingest routes — start/monitor/cancel ingestion tasks."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from src.web.app import templates
from src.web.tasks import task_manager

router = APIRouter()


@router.get("/ingest")
async def ingest_page(request: Request):
    tasks = task_manager.all_tasks()
    return templates.TemplateResponse(
        "ingest.html", {"request": request, "tasks": tasks}
    )


@router.post("/ingest/start")
async def ingest_start(request: Request, source: str = Form(...), since: str = Form("")):
    since_val = since.strip() or None

    if task_manager.has_running(source):
        tasks = task_manager.all_tasks()
        return templates.TemplateResponse(
            "ingest.html",
            {"request": request, "tasks": tasks, "error": f"An ingest for '{source}' is already running."},
        )

    task = task_manager.start_ingest(source, since_val)
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
