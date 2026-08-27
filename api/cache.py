"""Small process-local TTL cache used only to shield external read stores."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

_entries: dict[tuple[Any, ...], tuple[float, Any]] = {}
_lock = threading.Lock()


def get_or_set(key: tuple[Any, ...], ttl_seconds: int, loader: Callable[[], Any]) -> Any:
    now = time.monotonic()
    with _lock:
        cached = _entries.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]

    value = loader()
    with _lock:
        _entries[key] = (time.monotonic() + ttl_seconds, value)
    return value


def invalidate_read_cache() -> None:
    with _lock:
        _entries.clear()
