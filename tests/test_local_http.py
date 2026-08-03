"""Tests for the fail-closed local HTTP transport."""

import ipaddress
from urllib.parse import urlparse

import pytest

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
        "proxies": {"http": "", "https": "", "all": ""},
    }


def test_request_rejects_external_destinations_before_transport(monkeypatch):
    monkeypatch.setattr(
        local_http._SESSION,
        "request",
        lambda *args, **kwargs: pytest.fail("transport must not be called"),
    )

    with pytest.raises(ValueError, match="localhost"):
        local_http.get("https://example.com/private")


def test_request_pins_hostname_and_ignores_explicit_proxy(monkeypatch):
    captured = {}
    local_http._SESSION.proxies["all"] = "http://attacker.invalid:8080"

    monkeypatch.setattr(
        "imessage_rag.config.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 11434))],
    )

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return object()

    monkeypatch.setattr(local_http._SESSION, "request", fake_request)
    local_http.post(
        "http://model.local:11434/api/chat",
        proxies={"http": "http://attacker.invalid:8080"},
    )

    parsed = urlparse(captured["url"])
    assert parsed.hostname is not None
    assert ipaddress.ip_address(parsed.hostname).is_loopback
    assert local_http._SESSION.proxies == {}
    assert captured["proxies"] == {"http": "", "https": "", "all": ""}
    assert captured["allow_redirects"] is False