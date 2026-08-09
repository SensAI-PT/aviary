"""Best-effort cross-node weight shard prefetch (Aviary Phase 4)."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PREFETCH_SEC = float(os.environ.get("AVIARY_PREFETCH_SEC", "10"))
DEFAULT_PREFETCH_MAX = int(os.environ.get("AVIARY_PREFETCH_MAX", "2"))


def list_local_shards(model_path: str | Path) -> set[str]:
    root = Path(model_path)
    return {p.name for p in root.glob("*.safetensors")} if root.is_dir() else set()


def fetch_peer_shards(peer_endpoint: str, api_key: str = "") -> set[str]:
    url = f"{peer_endpoint.rstrip('/')}/cluster/shards"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return set(body.get("files") or [])
    except (OSError, json.JSONDecodeError, urllib.error.HTTPError):
        return set()


def download_shard(peer_endpoint: str, name: str, dest: Path, api_key: str = "") -> bool:
    url = f"{peer_endpoint.rstrip('/')}/cluster/shard?name={urllib.request.quote(name)}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.replace(dest)
        return True
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


class PrefetchDaemon:
    def __init__(self, model_path: str, placement_path: Path, master_url: str,
                 node_id: str, api_key: str = "", interval_sec: float = DEFAULT_PREFETCH_SEC,
                 max_inflight: int = DEFAULT_PREFETCH_MAX):
        self.model_path = Path(model_path)
        self.placement_path = placement_path
        self.master_url = master_url.rstrip("/")
        self.node_id = node_id
        self.api_key = api_key
        self.interval_sec = interval_sec
        self.max_inflight = max_inflight
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="aviary-prefetch", daemon=True)
        self._inflight = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _peer_endpoints(self) -> list[str]:
        url = f"{self.master_url}/cluster/nodes"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return [n["endpoint"] for n in body.get("nodes", [])
                    if n.get("status") == "healthy" and n.get("node_id") != self.node_id]
        except (OSError, json.JSONDecodeError, urllib.error.HTTPError):
            return []

    def _tick(self) -> None:
        local = list_local_shards(self.model_path)
        peers = self._peer_endpoints()
        if not peers:
            return
        for peer in peers:
            remote = fetch_peer_shards(peer, self.api_key)
            missing = sorted(remote - local)
            for name in missing[: self.max_inflight]:
                with self._lock:
                    if self._inflight >= self.max_inflight:
                        break
                    self._inflight += 1
                dest = self.model_path / name
                try:
                    if download_shard(peer, name, dest, self.api_key):
                        local.add(name)
                        print(f"[aviary-prefetch] fetched {name} from {peer}", flush=True)
                finally:
                    with self._lock:
                        self._inflight -= 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self._tick()
            except Exception as error:
                print(f"[aviary-prefetch] error: {error}", flush=True)
