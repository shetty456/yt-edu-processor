from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any

_store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
_locks: dict[str, asyncio.Lock] = {}


def _get(key: str) -> Any | None:
    if entry := _store.get(key):
        val, exp = entry
        if time.monotonic() < exp:
            return val
        del _store[key]
    return None


def _set(key: str, val: Any, ttl: int) -> None:
    from app.config import get_settings
    s = get_settings()
    if len(_store) >= s.cache_max_size:
        oldest = sorted(_store.items(), key=lambda x: x[1][1])
        for k, _ in oldest[: max(1, s.cache_max_size // 10)]:
            _store.pop(k, None)
    _store[key] = (val, time.monotonic() + ttl)


async def get_or_acquire(key: str) -> tuple[Any | None, asyncio.Lock | None]:
    """
    Returns (cached_value, None) on hit.
    Returns (None, lock) on miss — caller must call lock.release() after storing result.
    Uses double-checked locking to handle concurrent requests for same key.
    """
    val = _get(key)
    if val is not None:
        return val, None
    lock = _locks.setdefault(key, asyncio.Lock())
    await lock.acquire()
    val = _get(key)  # re-check after acquiring
    if val is not None:
        lock.release()
        return val, None
    return None, lock  # caller holds the lock


def pdf_key(data: bytes) -> str:
    return "pdf:" + hashlib.sha256(data).hexdigest()


def yt_key(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/|/v/)([A-Za-z0-9_-]{11})", url)
    return ("yt:" + m.group(1)) if m else None
