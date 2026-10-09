"""Provider interface + shared HTTP helpers.

This package is the ONLY place in the project that talks to the network.
Each provider downloads raw data into data/raw/<provider>/<snapshot>/ and
normalizes it into the canonical tables defined in screener/schema.py.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..config import Config
from ..store import Snapshot, Store

log = logging.getLogger(__name__)


class Provider(ABC):
    name: str = "base"

    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.settings = cfg.provider_settings(self.name)

    @abstractmethod
    def refresh(self, **options: Any) -> Snapshot:
        """Download (or generate) data, write canonical tables, return the new snapshot."""


class RateLimiter:
    """Simple token-spacing limiter: at most `per_second` calls per second."""

    def __init__(self, per_second: float):
        self.min_interval = 1.0 / max(per_second, 0.001)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class ApiError(RuntimeError):
    """HTTP error with any API key removed from the message (tracebacks get pasted into chats and
    issue trackers). `status_code` and a short `body` are kept for diagnostics."""

    def __init__(self, message: str, status_code: int = 0, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def redact(text: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Replace secret query values (any parameter whose name contains 'key' or 'token') with ***."""
    out = str(text)
    for k, v in (params or {}).items():
        if v and any(w in k.lower() for w in ("key", "token")):
            out = out.replace(str(v), "***")
    import re

    return re.sub(r"((?:api_?key|token)=)[^&\s'\"]+", r"\1***", out, flags=re.I)


def http_get(session, url: str, params: Optional[Dict[str, Any]] = None, *,
             limiter: Optional[RateLimiter] = None, retries: int = 5,
             timeout: float = 60.0, stream: bool = False, allow_redirects: bool = True):
    """GET with rate limiting and exponential backoff on 429/5xx/network errors.
    Client errors (400/401/403/404) are raised at once as ApiError; messages never contain the key."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        if limiter:
            limiter.wait()
        try:
            r = session.get(url, params=params, timeout=timeout, stream=stream,
                            allow_redirects=allow_redirects)
        except Exception as e:  # network errors
            last_err = e
            time.sleep(min(60.0, 2.0 ** attempt))
            continue
        code = getattr(r, "status_code", 200) or 200
        if code == 429 or code >= 500:
            wait = min(60.0, 2.0 ** attempt)
            log.warning("HTTP %s from %s — retrying in %.0fs", code, url, wait)
            last_err = ApiError(f"HTTP {code} from {url}", code)
            time.sleep(wait)
            continue
        if code >= 400:
            try:
                body = (r.text or "")[:300]
            except Exception:
                body = ""
            raise ApiError(redact(f"HTTP {code} from {url}: {body}".strip(), params), code,
                           redact(body, params))
        return r
    raise ApiError(redact(f"GET {url} failed after {retries} attempts: {last_err}", params),
                   getattr(last_err, "status_code", 0))


def get_provider(name: str, cfg: Config, store: Store) -> Provider:
    from .sec import SecProvider
    from .sharadar import SharadarProvider
    from .synthetic import SyntheticProvider

    registry = {p.name: p for p in (SyntheticProvider, SecProvider, SharadarProvider)}
    if name not in registry:
        raise KeyError(f"Unknown provider '{name}'. Choose from: {', '.join(registry)}")
    return registry[name](cfg, store)
