"""Per-request RPC trace buffer on agents."""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

MAX_EVENTS = 256


class TraceBuffer:
    def __init__(self, max_events: int = MAX_EVENTS):
        self._lock = threading.Lock()
        self._events: collections.deque[dict[str, Any]] = collections.deque(maxlen=max_events)
        self._job_id: str | None = None

    def set_job(self, job_id: str | None) -> None:
        with self._lock:
            self._job_id = job_id

    @property
    def job_id(self) -> str | None:
        with self._lock:
            return self._job_id

    def record(self, kind: str, node_id: str, layer: int, expert: int,
               rpc_us: float | None = None, local: bool = False) -> None:
        with self._lock:
            job_id = self._job_id
        if not job_id:
            return
        event = {
            "ts": time.time(),
            "job_id": job_id,
            "kind": kind,
            "node_id": node_id,
            "layer": layer,
            "expert": expert,
            "local": local,
        }
        if rpc_us is not None:
            event["rpc_us"] = round(rpc_us, 1)
        with self._lock:
            self._events.append(event)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
            return out
