"""Small dependency-free JSON HTTP transport for explicit provider adapters."""
from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(RuntimeError):
    """Base class for provider transport and protocol failures."""


class ProviderTimeout(ProviderError):
    """The provider did not respond within the configured timeout."""


class ProviderProtocolError(ProviderError):
    """A request or response does not fit the provider's documented contract."""


class ProviderHTTPError(ProviderError):
    """The provider returned a non-success HTTP status, kept as ``status``."""

    def __init__(self, status, message):
        self.status = status
        super().__init__(f"Provider returned HTTP {status}: {message}")


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def endpoint(base_url, path):
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url cannot contain credentials, query parameters or a fragment")
    joined_path = parsed.path.rstrip("/") + "/" + path.lstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, joined_path, "", ""))


def post_json(url, payload, *, api_key=None, timeout=30.0):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number")
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key is not None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a nonempty string or None")
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with build_opener(_RejectRedirects()).open(request, timeout=float(timeout)) as response:
            response_body = response.read()
    except HTTPError as exc:
        response_body = exc.read()
        message = response_body.decode("utf-8", errors="replace")[:1000]
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        raise ProviderHTTPError(exc.code, message) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderTimeout(f"Provider request timed out after {timeout:g} seconds") from exc
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise ProviderTimeout(f"Provider request timed out after {timeout:g} seconds") from exc
        raise ProviderError(f"Provider request failed: {exc.reason}") from exc
    try:
        decoded = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderProtocolError("Provider response is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ProviderProtocolError("Provider response must be a JSON object")
    return decoded
