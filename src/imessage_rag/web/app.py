"""FastAPI application factory for the imessage-rag web UI."""

import secrets
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from imessage_rag.config import AUTH_TOKEN_PATH

_WEB_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))

_AUTH_COOKIE = "imessage_rag_auth"
_PUBLIC_PREFIXES = ("/static/",)
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


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
    """Require a local auth token for all app routes except static assets."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    def _is_authorized(self, request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {self.token}":
            return True
        if request.cookies.get(_AUTH_COOKIE) == self.token:
            return True
        return request.query_params.get("token") == self.token

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        query_token = request.query_params.get("token")
        if query_token == self.token and request.method == "GET":
            response = RedirectResponse(
                url=str(request.url.remove_query_params(["token"])),
                status_code=303,
            )
            response.set_cookie(
                _AUTH_COOKIE,
                self.token,
                httponly=True,
                samesite="strict",
                secure=False,
            )
            return response

        if self._is_authorized(request):
            response = await call_next(request)
            if query_token == self.token:
                response.set_cookie(
                    _AUTH_COOKIE,
                    self.token,
                    httponly=True,
                    samesite="strict",
                    secure=False,
                )
            return response

        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return PlainTextResponse("Unauthorized. Start with the URL printed by `imessage-rag serve`.", status_code=401)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Reject POST requests from foreign origins."""

    @staticmethod
    def _origin_allowed(origin: str, request: Request) -> bool:
        parsed = urlparse(origin)
        if parsed.scheme != request.url.scheme:
            return False
        if parsed.hostname not in _LOOPBACK_HOSTS:
            return False
        return parsed.port == request.url.port

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin and not self._origin_allowed(origin, request):
                return JSONResponse(
                    {"detail": "CSRF check failed: origin not allowed"},
                    status_code=403,
                )
        return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(title="imessage-rag", docs_url=None, redoc_url=None)

    token = _get_or_create_token()

    # Store token on app state for the printed startup URL.
    app.state.auth_token = token

    # Middleware is applied in reverse order — CSRF first, then auth
    app.add_middleware(AuthMiddleware, token=token)
    app.add_middleware(CSRFMiddleware)

    app.mount(
        "/static",
        StaticFiles(directory=str(_WEB_DIR / "static")),
        name="static",
    )

    from imessage_rag.web.routes.query import router as query_router
    from imessage_rag.web.routes.ingest import router as ingest_router
    from imessage_rag.web.routes.status import router as status_router
    from imessage_rag.web.routes.settings import router as settings_router

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
