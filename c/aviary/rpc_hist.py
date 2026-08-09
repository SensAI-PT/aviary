"""Rolling RPC latency histogram for the cluster dashboard."""

from __future__ import annotations

import collections
import statistics
import threading
import time
from typing import Any

BUCKETS_US = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000)
WINDOW_SEC = 60.0


class RpcHistogram:
    def __init__(self, window_sec: float = WINDOW_SEC):
        self._window_sec = window_sec
        self._lock = threading.Lock()
        self._samples: collections.deque[tuple[float, float]] = collections.deque()

    def record(self, us: float, ts: float | None = None) -> None:
        now = ts if ts is not None else time.time()
        with self._lock:
            self._samples.append((now, us))
            cutoff = now - self._window_sec
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            vals = [us for _, us in self._samples]
        if not vals:
            return {"count": 0, "buckets": [], "p50_us": 0, "p95_us": 0, "window_sec": self._window_sec}
        vals.sort()
        counts = [sum(1 for v in vals if (BUCKETS_US[i - 1] if i else 0) < v <= lim)
                  for i, lim in enumerate(BUCKETS_US)]
        over = sum(1 for v in vals if v > BUCKETS_US[-1])
        return {
            "count": len(vals),
            "buckets": [{"max_us": lim, "count": counts[i]} for i, lim in enumerate(BUCKETS_US)],
            "over_max": over,
            "p50_us": round(statistics.median(vals), 1),
            "p95_us": round(vals[int(0.95 * (len(vals) - 1))], 1),
            "window_sec": self._window_sec,
        }
