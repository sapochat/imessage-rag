"""Tests for web UI security and request hardening."""

import asyncio

from starlette.requests import Request
from starlette.responses import PlainTextResponse

from imessage_rag.web.app import AuthMiddleware, CSRFMiddleware
from imessage_rag.web.routes.settings import _validate_settings_form


def _request(
    path="/",
    method="GET",
    query_string=b"",
    headers=None,
    server=("127.0.0.1", 5391),
):
    raw_headers = [(b"host", f"{server[0]}:{server[1]}".encode())]
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode(), value.encode()))
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "query_string": query_string,
            "headers": raw_headers,
            "scheme": "http",
            "server": server,
            "client": ("127.0.0.1", 50000),
        }
    )


async def _ok_response(_request):
    return PlainTextResponse("ok")


class TestWebAuth:
    def test_pages_require_auth(self):
        middleware = AuthMiddleware(app=lambda scope, receive, send: None, token="test-token")
        req = _request()

        resp = asyncio.run(middleware.dispatch(req, _ok_response))

        assert resp.status_code == 401

    def test_token_query_sets_cookie_and_redirects(self):
        middleware = AuthMiddleware(app=lambda scope, receive, send: None, token="test-token")
        req = _request(query_string=b"token=test-token")

        resp = asyncio.run(middleware.dispatch(req, _ok_response))

        assert resp.status_code == 303
        assert str(resp.headers["location"]) == "http://127.0.0.1:5391/"
        assert "imessage_rag_auth=test-token" in resp.headers["set-cookie"]

    def test_cookie_allows_app_routes(self):
        middleware = AuthMiddleware(app=lambda scope, receive, send: None, token="test-token")
        req = _request(headers={"cookie": "imessage_rag_auth=test-token"})

        resp = asyncio.run(middleware.dispatch(req, _ok_response))

        assert resp.status_code == 200

    def test_foreign_origin_post_is_rejected(self):
        middleware = CSRFMiddleware(app=lambda scope, receive, send: None)
        req = _request(
            path="/settings/save",
            method="POST",
            headers={"origin": "http://evil.test"},
        )

        resp = asyncio.run(middleware.dispatch(req, _ok_response))

        assert resp.status_code == 403

    def test_same_origin_post_is_allowed(self):
        middleware = CSRFMiddleware(app=lambda scope, receive, send: None)
        req = _request(
            path="/settings/save",
            method="POST",
            headers={"origin": "http://127.0.0.1:5391"},
        )

        resp = asyncio.run(middleware.dispatch(req, _ok_response))

        assert resp.status_code == 200


class TestSettingsValidation:
    def test_rejects_unknown_backend(self):
        assert _validate_settings_form("external", "model", "") is not None

    def test_requires_model(self):
        assert _validate_settings_form("ollama", "", "") == "Model is required."

    def test_requires_openai_api_url(self):
        assert _validate_settings_form("openai", "model", "") is not None

    def test_accepts_valid_ollama_settings(self):
        assert _validate_settings_form("ollama", "gemma4", "") is None
