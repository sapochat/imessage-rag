"""FastAPI application factory for the personal-rag web UI."""

import secrets
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import AUTH_TOKEN_PATH

_WEB_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))

_ALLOWED_ORIGINS = {"http://127.0.0.1", "http://localhost"}


def _get_or_create_token() -> str:
    """Read the auth token from disk, or generate and persist a new one."""
    AUTH_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AUTH_TOKEN_PATH.exists():
        token = AUTH_TOKEN_PATH.read_text().strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    AUTH_TOKEN_PATH.write_text(token)
    AUTH_TOKEN_PATH.chmod(0o600)
    return token


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a bearer token on all /api/ routes."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            # Check Authorization header
            auth = request.headers.get("authorization", "")
            if auth == f"Bearer {self.token}":
                return await call_next(request)
            # Check query parameter
            if request.query_params.get("token") == self.token:
                return await call_next(request)
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Reject POST requests from foreign origins."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin:
                parsed = urlparse(origin)
                origin_base = f"{parsed.scheme}://{parsed.hostname}"
                if origin_base not in _ALLOWED_ORIGINS:
                    return JSONResponse(
                        {"detail": "CSRF check failed: origin not allowed"},
                        status_code=403,
                    )
        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(title="Personal RAG", docs_url=None, redoc_url=None)

    token = _get_or_create_token()

    # Store token on app state and inject into all templates
    app.state.auth_token = token
    templates.env.globals["auth_token"] = token

    # Middleware is applied in reverse order — CSRF first, then auth
    app.add_middleware(AuthMiddleware, token=token)
    app.add_middleware(CSRFMiddleware)

    app.mount(
        "/static",
        StaticFiles(directory=str(_WEB_DIR / "static")),
        name="static",
    )

    from src.web.routes.query import router as query_router
    from src.web.routes.ingest import router as ingest_router
    from src.web.routes.status import router as status_router
    from src.web.routes.settings import router as settings_router

    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(status_router)
    app.include_router(settings_router)

    return app


def run(port: int = 5391) -> None:
    import uvicorn

    app = create_app()
    token = app.state.auth_token
    print(f"Auth token: {token}")
    print(f"Open: http://127.0.0.1:{port}?token={token}")
    uvicorn.run(app, host="127.0.0.1", port=port)
