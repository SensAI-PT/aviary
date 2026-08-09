"""In-memory cluster job tracker (Spark-style request visibility on the master)."""

from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClusterJob:
    job_id: str
    node_id: str
    node_endpoint: str
    path: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    http_status: int | None = None
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        duration = (self.ended_at or now) - self.started_at
        return {
            "job_id": self.job_id,
            "node_id": self.node_id,
            "node_endpoint": self.node_endpoint,
            "path": self.path,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": round(duration, 3),
            "http_status": self.http_status,
            "error": self.error,
            "trace": list(self.trace),
        }


class JobTracker:
    def __init__(self, history: int = 100):
        self._lock = threading.Lock()
        self._active: dict[str, ClusterJob] = {}
        self._history: collections.deque[ClusterJob] = collections.deque(maxlen=history)

    def start(self, node_id: str, node_endpoint: str, path: str) -> ClusterJob:
        job = ClusterJob(job_id=str(uuid.uuid4()), node_id=node_id,
                         node_endpoint=node_endpoint, path=path)
        with self._lock:
            self._active[job.job_id] = job
        return job

    def append_trace(self, job_id: str, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self._lock:
            job = self._active.get(job_id)
            if job is None:
                for past in self._history:
                    if past.job_id == job_id:
                        job = past
                        break
            if job is None:
                return
            job.trace.extend(events)

    def finish(self, job_id: str, http_status: int | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._active.pop(job_id, None)
            if job is None:
                return
            job.ended_at = time.time()
            job.http_status = http_status
            job.error = error
            job.status = "failed" if error or (http_status and http_status >= 400) else "completed"
            self._history.appendleft(job)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = [j.snapshot() for j in self._active.values()]
            history = [j.snapshot() for j in self._history]
        return {
            "active": sorted(active, key=lambda j: j["started_at"], reverse=True),
            "history": history,
            "active_count": len(active),
            "completed_count": sum(1 for j in history if j["status"] == "completed"),
            "failed_count": sum(1 for j in history if j["status"] == "failed"),
        }
