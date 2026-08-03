"""Tests for the fail-closed local HTTP transport."""

from imessage_rag import local_http


def test_session_ignores_environment_proxies():
    assert local_http._SESSION.trust_env is False


def test_request_disables_redirects_even_if_caller_enables_them(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return object()

    monkeypatch.setattr(local_http._SESSION, "request", fake_request)
    result = local_http.get("http://127.0.0.1:11434/api/tags", allow_redirects=True)

    assert result is not None
    assert captured == {
        "method": "GET",
        "url": "http://127.0.0.1:11434/api/tags",
        "allow_redirects": False,
    }