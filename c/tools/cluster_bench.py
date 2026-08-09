#!/usr/bin/env python3
"""Aviary cluster throughput and Phase 2 measurement harness."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _post(url: str, body: dict[str, Any], api_key: str = "", timeout: float = 120.0) -> tuple[int, float, bytes]:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "*/*"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            return resp.status, time.perf_counter() - t0, payload
    except urllib.error.HTTPError as err:
        return err.code, time.perf_counter() - t0, err.read()


def _health(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/cluster/health", timeout=timeout) as resp:
            return resp.status == 200
    except OSError:
        return False


def run_concurrent(base_url: str, n: int, workers: int, model: str, api_key: str,
                   max_tokens: int, smoke: bool) -> dict[str, Any]:
    if smoke and not _health(base_url):
        return {"skipped": True, "reason": "master not reachable (--smoke)"}
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    latencies: list[float] = []
    errors = 0

    def one(_: int) -> float | None:
        status, elapsed, _ = _post(url, body, api_key)
        if status != 200:
            return None
        return elapsed

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(n)))
    total = time.perf_counter() - t0
    latencies = [x for x in results if x is not None]
    errors = n - len(latencies)
    out: dict[str, Any] = {
        "requests": n,
        "workers": workers,
        "ok": len(latencies),
        "errors": errors,
        "elapsed_sec": round(total, 3),
        "rps": round(len(latencies) / total, 3) if total > 0 else 0.0,
    }
    if latencies:
        latencies.sort()
        out["p50_ms"] = round(statistics.median(latencies) * 1000, 1)
        out["p95_ms"] = round(latencies[int(0.95 * (len(latencies) - 1))] * 1000, 1)
    return out


def measure_rtt(expert_host: str, expert_port: int, layer: int = 0, eid: int = 0,
                samples: int = 10) -> dict[str, Any]:
    import socket
    import struct

    latencies: list[float] = []
    hidden = 128
    raw = struct.pack(f"{hidden}f", *([0.0] * hidden))
    for i in range(samples):
        try:
            conn = socket.create_connection((expert_host, expert_port), timeout=1.0)
            t0 = time.perf_counter()
            conn.sendall(f"EXEC_EXPERT bench{i} {layer} {eid} {len(raw)}\n".encode() + raw + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
            latencies.append((time.perf_counter() - t0) * 1e6)
            conn.close()
        except OSError:
            pass
    if not latencies:
        return {"samples": 0, "error": "no successful RPC probes"}
    latencies.sort()
    return {
        "samples": len(latencies),
        "p50_us": round(statistics.median(latencies), 1),
        "p95_us": round(latencies[int(0.95 * (len(latencies) - 1))], 1),
        "min_us": round(min(latencies), 1),
        "max_us": round(max(latencies), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aviary cluster benchmark harness")
    parser.add_argument("--url", default="http://127.0.0.1:9000", help="Master HTTP base URL")
    parser.add_argument("--model", default="hy3-colibri")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Skip if master unreachable")
    parser.add_argument("--rtt-host", default="", help="Expert RPC host for RTT probe")
    parser.add_argument("--rtt-port", type=int, default=9003)
    parser.add_argument("--markdown", action="store_true", help="Print markdown summary")
    args = parser.parse_args()

    bench = run_concurrent(args.url, args.requests, args.workers, args.model, args.api_key,
                           args.max_tokens, args.smoke)
    report: dict[str, Any] = {"throughput": bench}
    if args.rtt_host:
        report["rtt"] = measure_rtt(args.rtt_host, args.rtt_port)

    if args.markdown:
        print("## Aviary cluster bench\n")
        if bench.get("skipped"):
            print(f"- Skipped: {bench.get('reason')}\n")
            return 0
        print(f"- Requests: {bench['ok']}/{bench['requests']} ok, {bench['rps']} req/s")
        if "p50_ms" in bench:
            print(f"- Latency: p50={bench['p50_ms']}ms p95={bench['p95_ms']}ms")
        if "rtt" in report:
            rtt = report["rtt"]
            if rtt.get("samples"):
                print(f"- Expert RPC RTT: p50={rtt['p50_us']}µs p95={rtt['p95_us']}µs "
                      f"({rtt['samples']} samples)")
    else:
        json.dump(report, sys.stdout, indent=2)
        print()
    return 0 if bench.get("skipped") or bench.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
