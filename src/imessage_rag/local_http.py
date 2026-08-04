"""HTTP client for validated local-only model endpoints."""

from __future__ import annotations

from typing import Any

import requests

from imessage_rag.config import _validate_localhost


_SESSION = requests.Session()
_SESSION.trust_env = False


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Send one local request without proxies or redirect traversal."""
    url = _validate_localhost(url)
    _SESSION.trust_env = False
    _SESSION.proxies.clear()
    kwargs["allow_redirects"] = False
    kwargs["proxies"] = {"http": "", "https": "", "all": ""}
    return _SESSION.request(method, url, **kwargs)


def get(url: str, **kwargs: Any) -> requests.Response:
    return request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return request("POST", url, **kwargs)
