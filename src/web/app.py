"""FastAPI application factory for the personal-rag web UI."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_WEB_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="Personal RAG", docs_url=None, redoc_url=None)

    app.mount(
        "/static",
        StaticFiles(directory=str(_WEB_DIR / "static")),
        name="static",
    )

    from src.web.routes.query import router as query_router
    from src.web.routes.ingest import router as ingest_router
    from src.web.routes.status import router as status_router

    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(status_router)

    return app


def run(host: str = "127.0.0.1", port: int = 5391) -> None:
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)
