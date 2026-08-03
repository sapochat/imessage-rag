"""HTTP client for validated local-only model endpoints."""

from __future__ import annotations

from typing import Any

import requests


_SESSION = requests.Session()
_SESSION.trust_env = False


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Send one local request without proxies or redirect traversal."""
    kwargs["allow_redirects"] = False
    return _SESSION.request(method, url, **kwargs)


def get(url: str, **kwargs: Any) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return request("POST", url, **kwargs)
