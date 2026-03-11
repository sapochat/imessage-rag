"""Tests for configuration validation."""

import pytest

from src.config import _validate_localhost


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


class TestExpandPath:
    def test_expand_tilde(self):
        from src.config import _expand
        result = _expand("~/test")
        assert "~" not in str(result)
        assert str(result).endswith("/test")
