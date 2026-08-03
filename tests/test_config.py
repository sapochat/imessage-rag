"""Tests for configuration validation."""

import socket

import pytest

from imessage_rag.config import _validate_localhost


class TestValidateLocalhost:
    def test_accepts_localhost(self):
        assert _validate_localhost("http://localhost:11434") == "http://localhost:11434"

    def test_accepts_127_0_0_1(self):
        assert _validate_localhost("http://127.0.0.1:8080") == "http://127.0.0.1:8080"

    def test_accepts_ipv6_loopback(self):
        assert _validate_localhost("http://[::1]:5000") == "http://[::1]:5000"

    def test_rejects_external_hostname(self):
        with pytest.raises(ValueError, match="localhost"):
            _validate_localhost("http://api.openai.com/v1")

    def test_rejects_unresolvable_hostname(self):
        with pytest.raises(ValueError, match="cannot be resolved"):
            _validate_localhost("http://definitely-not-a-real-host-12345.invalid/api")

    def test_rejects_url_without_hostname(self):
        with pytest.raises(ValueError, match="HTTP\\(S\\) localhost URL"):
            _validate_localhost("localhost:11434")

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="HTTP\\(S\\) localhost URL"):
            _validate_localhost("ftp://localhost:11434")

    def test_rejects_invalid_port(self):
        with pytest.raises(ValueError, match="HTTP\\(S\\) localhost URL"):
            _validate_localhost("http://localhost:not-a-port")

    @pytest.mark.parametrize(
        "url",
        [
            "http://user@localhost:11434",
            "http://user:password@localhost:11434",
            "http://evil.example\\@localhost:11434",
        ],
    )
    def test_rejects_userinfo_and_backslash_authorities(self, url):
        with pytest.raises(ValueError, match="HTTP\\(S\\) localhost URL"):
            _validate_localhost(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:not-a-port",
            "http://[::1]:not-a-port",
            "http://127.0.0.1:99999",
        ],
    )
    def test_rejects_invalid_port_on_loopback_ip(self, url):
        with pytest.raises(ValueError, match="HTTP\\(S\\) localhost URL"):
            _validate_localhost(url)

    def test_rejects_hostname_with_mixed_loopback_and_external_addresses(self, monkeypatch):
        def fake_getaddrinfo(*args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 11434)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

        with pytest.raises(ValueError, match="93\\.184\\.216\\.34"):
            _validate_localhost("http://mixed-localhost.test:11434")

    def test_accepts_hostname_when_all_addresses_are_loopback(self, monkeypatch):
        def fake_getaddrinfo(*args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 11434)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

        assert (
            _validate_localhost("http://loopback-alias.test:11434")
            == "http://loopback-alias.test:11434"
        )


class TestExpandPath:
    def test_expand_tilde(self):
        from imessage_rag.config import _expand
        result = _expand("~/test")
        assert "~" not in str(result)
        assert str(result).endswith("/test")
