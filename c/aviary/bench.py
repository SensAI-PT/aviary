"""Master-side cluster benchmark runner (EPA harness)."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aviary.jobs import JobTracker
from aviary.placement import PlacementScheduler
from aviary.protocol import placement_frame, reset_usage_frame
from aviary.registry import NodeRegistry

DEFAULT_PROMPT = "Say hello in one word."
DEFAULT_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "bench"

PRESET_STEPS: dict[str, list[dict[str, Any]]] = {
    "cold_sequential": [{"step": "cold_sequential", "wipe_usage": True, "workers": 1}],
    "warm_sequential": [{"step": "warm_sequential", "wipe_usage": False, "workers": 1}],
    "concurrent": [{"step": "concurrent", "wipe_usage": False, "workers": None}],
    "suite": [
        {"step": "cold_sequential", "wipe_usage": True, "workers": 1},
        {"step": "warm_sequential", "wipe_usage": False, "workers": 1},
        {"step": "concurrent", "wipe_usage": False, "workers": None},
    ],
}


def bench_dir() -> Path:
    return Path(os.environ.get("AVIARY_BENCH_DIR", str(DEFAULT_BENCH_DIR)))


def percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    idx = int(pct * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def hop_mix(traces: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {"local": 0, "remote": 0, "fallback": 0}
    for job in traces:
        for ev in job.get("trace") or []:
            kind = str(ev.get("kind") or "").lower()
            if kind in counts:
                counts[kind] += 1
    total = sum(counts.values()) or 1
    return {k: round(v / total * 100.0, 1) for k, v in counts.items()}


def primary_node_mix(jobs: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for job in jobs:
        nid = str(job.get("node_id") or "")
        if nid:
            counts[nid] = counts.get(nid, 0) + 1
    total = sum(counts.values()) or 1
    return {k: round(v / total * 100.0, 1) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def donor_primary_warnings(nodes: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> list[str]:
    """Warn when a donor_only node served chat as primary (coordinator violation)."""
    roles = {n["node_id"]: n.get("coordinator_eligible", True) for n in nodes}
    reasons = {n["node_id"]: n.get("donor_only_reason", "") for n in nodes}
    return [
        f"donor_only node {nid[:8]} was primary {pct}% ({reasons.get(nid, 'donor_only')})"
        for nid, pct in primary_node_mix(jobs).items()
        if pct > 0 and not roles.get(nid, True)
    ]


def cluster_snapshot(registry: NodeRegistry) -> dict[str, Any]:
    """Hardware + tier budgets reported by agents at bench time."""
    snap = registry.snapshot()
    nodes = []
    for n in snap.get("nodes") or []:
        hw = n.get("hwinfo") or {}
        tiers = n.get("tiers") or {}
        tc = n.get("tiers_config") or {}
        nodes.append({
            "node_id": n.get("node_id"),
            "host": n.get("host"),
            "endpoint": n.get("endpoint"),
            "model_id": n.get("model_id"),
            "arch": n.get("arch"),
            "status": n.get("status"),
            "ram_gb": tiers.get("ram_gb"),
            "vram_gb": tiers.get("vram_gb"),
            "ram_flag": n.get("ram_config_gb") or tc.get("ram_gb"),
            "vram_flag": n.get("vram_config_gb") or tc.get("vram_gb"),
            "ram_occ": n.get("ram_occ_gb") if n.get("ram_occ_gb") is not None else tiers.get("ram_gb"),
            "vram_occ": n.get("vram_occ_gb") if n.get("vram_occ_gb") is not None else tiers.get("vram_gb"),
            "gpu_tier": n.get("gpu_tier"),
            "coordinator_eligible": n.get("coordinator_eligible", True),
            "donor_only_reason": n.get("donor_only_reason", ""),
            "disk_gb": tiers.get("disk"),
            "ram_total_gb": hw.get("ram_total_gb"),
            "ram_avail_gb": hw.get("ram_avail_gb"),
            "vram_total_gb": hw.get("vram_total_gb"),
            "cores": hw.get("cores"),
            "cpu": hw.get("cpu"),
            "gpu": hw.get("gpu"),
            "gpus": hw.get("gpus"),
        })
    return {
        "cohort": snap.get("cohort") or {},
        "nodes": nodes,
        "healthy": snap.get("healthy"),
        "total": snap.get("total"),
        "coordinator_id": snap.get("coordinator_id", ""),
    }


def compute_epa(p50_cold: float | None, p50_warm: float | None) -> float | None:
    if not p50_cold or not p50_warm or p50_warm <= 0:
        return None
    return round(p50_cold / p50_warm - 1.0, 3)


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {}
    vals = sorted(latencies)
    return {
        "p50_sec": round(statistics.median(vals), 3),
        "p95_sec": round(percentile(vals, 0.95) or vals[-1], 3),
        "min_sec": round(vals[0], 3),
        "max_sec": round(vals[-1], 3),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["## Aviary cluster bench", ""]
    meta = result.get("meta") or {}
    lines += [
        f"- Preset: **{meta.get('preset', '?')}**",
        f"- Requests: {meta.get('requests', '?')} · workers: {meta.get('workers', '?')} · max_tokens: {meta.get('max_tokens', '?')}",
        f"- Wipe usage: {meta.get('wipe_usage')} · local-only: {meta.get('local_only')}",
        f"- Finished: {result.get('finished_at', '')}",
        "",
    ]
    cluster = result.get("cluster") or {}
    cohort = cluster.get("cohort") or {}
    if cohort.get("model_id") or cohort.get("arch"):
        lines.append(f"- Cohort: **{cohort.get('model_id') or '?'}** · {cohort.get('arch') or '?'}")
        lines.append("")
    nodes = cluster.get("nodes") or []
    if nodes:
        lines += [
            "### Cluster config", "",
            "| node | host | RAM flag | VRAM flag | RAM occ | VRAM occ | coord |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for n in nodes:
            nid = str(n.get("node_id") or "")[:8]
            host = n.get("host") or "?"
            ram_f = n.get("ram_flag")
            vram_f = n.get("vram_flag")
            ram_o = n.get("ram_occ")
            vram_o = n.get("vram_occ")
            coord = "yes" if n.get("coordinator_eligible", True) else f"no ({n.get('donor_only_reason') or '?'})"
            lines.append(
                f"| `{nid}` | {host} | {ram_f if ram_f is not None else '—'} GB | "
                f"{vram_f if vram_f is not None else '—'} GB | {ram_o if ram_o is not None else '—'} GB | "
                f"{vram_o if vram_o is not None else '—'} GB | {coord} |"
            )
        lines.append("")
    score = result.get("scoreboard") or {}
    if score:
        lines += ["### Scoreboard", "", "| metric | value |", "|---|---|"]
        for key in ("p50_cold_sec", "p50_warm_sec", "p95_cold_sec", "p95_warm_sec", "epa", "rps_at_w"):
            if key in score and score[key] is not None:
                label = key.replace("_", " ")
                lines.append(f"| {label} | {score[key]} |")
        hops = score.get("hop_mix_pct") or {}
        if hops:
            lines.append(f"| hop mix | local {hops.get('local', 0)}% · remote {hops.get('remote', 0)}% · fallback {hops.get('fallback', 0)}% |")
        prim = score.get("primary_node_mix_pct") or {}
        if prim:
            mix = ", ".join(f"{nid[:8]} {pct}%" for nid, pct in list(prim.items())[:4])
            lines.append(f"| primary nodes | {mix} |")
        warnings = score.get("donor_primary_warnings") or []
        if warnings:
            lines.append(f"| **warnings** | {'; '.join(warnings)} |")
        lines.append("")
    for step in result.get("steps") or []:
        lat = step.get("latency") or {}
        lines += [
            f"### {step.get('step', 'step')}",
            "",
            f"- ok {step.get('ok', 0)}/{step.get('requests', 0)} · errors {step.get('errors', 0)}",
        ]
        if lat.get("p50_sec") is not None:
            lines.append(f"- wall: p50={lat['p50_sec']}s p95={lat.get('p95_sec')}s")
        if step.get("rps") is not None:
            lines.append(f"- throughput: {step['rps']} req/s")
        hops = step.get("hop_mix_pct") or {}
        if hops:
            lines.append(f"- hops: local {hops.get('local', 0)}% · remote {hops.get('remote', 0)}% · fallback {hops.get('fallback', 0)}%")
        placement = step.get("placement") or {}
        blocks = placement.get("planned_blocks") or placement.get("blocks")
        prefs = placement.get("node_tier_prefs") or {}
        if blocks:
            lines.append(f"- planned blocks: {len(blocks) if isinstance(blocks, list) else 'see json'}")
        if prefs:
            pref_txt = ", ".join(f"{nid[:8]} tier≤{p.get('max_tier', '?')}" for nid, p in list(prefs.items())[:4])
            lines.append(f"- node tier prefs: {pref_txt}")
        lines.append("")
    if result.get("last_error"):
        lines += [f"_Last error: {result['last_error']}_", ""]
    return "\n".join(lines).rstrip() + "\n"


def build_scoreboard(steps: list[dict[str, Any]], cluster_nodes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    by_name = {s["step"]: s for s in steps}
    cold = by_name.get("cold_sequential") or {}
    warm = by_name.get("warm_sequential") or {}
    conc = by_name.get("concurrent") or {}
    cold_lat = cold.get("latency") or {}
    warm_lat = warm.get("latency") or {}
    p50_cold = cold_lat.get("p50_sec")
    p50_warm = warm_lat.get("p50_sec")
    all_jobs = [j for s in steps for j in (s.get("jobs") or [])]
    warnings = donor_primary_warnings(cluster_nodes or [], all_jobs)
    return {
        "p50_cold_sec": p50_cold,
        "p50_warm_sec": p50_warm,
        "p95_cold_sec": cold_lat.get("p95_sec"),
        "p95_warm_sec": warm_lat.get("p95_sec"),
        "epa": compute_epa(p50_cold, p50_warm),
        "rps_at_w": conc.get("rps"),
        "hop_mix_pct": hop_mix(all_jobs),
        "primary_node_mix_pct": primary_node_mix(all_jobs),
        "donor_primary_warnings": warnings,
    }


@dataclass
class BenchConfig:
    preset: str = "cold_sequential"
    wipe_usage: bool | None = None
    local_only: bool = False
    requests: int = 8
    workers: int = 4
    max_tokens: int = 32
    prompt: str = DEFAULT_PROMPT


@dataclass
class BenchProgress:
    status: str = "idle"
    preset: str = ""
    step: str = ""
    chat_index: int = 0
    chat_total: int = 0
    last_error: str = ""


class BenchRunner:
    def __init__(self, registry: NodeRegistry, scheduler: PlacementScheduler,
                 jobs: JobTracker, model_id: str = "", api_key: str | None = None):
        self.registry = registry
        self.scheduler = scheduler
        self.jobs = jobs
        self.model_id = model_id
        self.api_key = api_key
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._progress = BenchProgress()
        self._last_result: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            prog = self._progress
            return {
                "status": prog.status,
                "progress": {
                    "preset": prog.preset,
                    "step": prog.step,
                    "chat_index": prog.chat_index,
                    "chat_total": prog.chat_total,
                    "last_error": prog.last_error,
                },
                "last_result": self._last_result,
            }

    def start(self, cfg: BenchConfig) -> None:
        with self._lock:
            if self._progress.status == "running":
                raise RuntimeError("bench already running")
            self._progress = BenchProgress(status="running", preset=cfg.preset)
            self._thread = threading.Thread(target=self._run, args=(cfg,), daemon=True, name="aviary-bench")
            self._thread.start()

    def _set_progress(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._progress, k, v)

    def _finish(self, result: dict[str, Any] | None = None, error: str = "") -> None:
        with self._lock:
            self._progress.status = "failed" if error else "idle"
            self._progress.last_error = error
            if result is not None:
                self._last_result = result

    def _pick_node(self):
        plan = self.scheduler.last_plan
        hot: set[tuple[int, int]] = set()
        if plan and plan.experts:
            hot = {tuple(map(int, k.split(":"))) for k in plan.experts}
        node = self.registry.pick_with_affinity(hot or None)
        if node is None:
            raise RuntimeError("no healthy agents registered")
        return node

    def _proxy_chat(self, body: bytes) -> tuple[int, float, str | None, str | None]:
        node = self._pick_node()
        job = self.jobs.start(node.node_id, node.endpoint, "/v1/chat/completions")
        self.registry.increment_inflight(node.node_id, 1)
        headers = {"Content-Type": "application/json", "Accept": "*/*", "X-Aviary-Job-Id": job.job_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        parsed = urlsplit(node.endpoint)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=600)
        t0 = time.perf_counter()
        status = 0
        try:
            conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            resp.read()
            resp.close()
            self.jobs.finish(job.job_id, http_status=status)
            return status, time.perf_counter() - t0, job.job_id, node.node_id
        except OSError as err:
            self.jobs.finish(job.job_id, error=str(err))
            raise
        finally:
            conn.close()
            self.registry.increment_inflight(node.node_id, -1)

    def _chat_body(self, cfg: BenchConfig) -> bytes:
        model = self.model_id or os.environ.get("COLI_MODEL_ID", "colibri")
        return json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": cfg.prompt}],
            "max_tokens": cfg.max_tokens,
            "stream": False,
        }).encode("utf-8")

    def _jobs_for_ids(self, job_ids: list[str]) -> list[dict[str, Any]]:
        snap = self.jobs.snapshot()
        by_id = {j["job_id"]: j for j in snap["active"] + snap["history"]}
        return [by_id[jid] for jid in job_ids if jid in by_id]

    def _wipe_usage(self) -> None:
        self.scheduler.clear_usage()
        self.registry.clear_usage()
        for nid in self.registry.control_connections():
            try:
                self.registry.control_send(nid, reset_usage_frame().encode("utf-8"))
            except OSError as err:
                print(f"[aviary-bench] RESET_USAGE to {nid} failed: {err}")

    def _push_local_only(self) -> None:
        snap = self.registry.snapshot()
        nodes = snap.get("nodes") or []
        plan = self.scheduler.last_plan or self.scheduler.recompute(nodes)
        for node in nodes:
            if node.get("status") != "healthy":
                continue
            nid = node["node_id"]
            if nid not in self.registry.control_connections():
                continue
            peers = {n["node_id"]: f"{n.get('host', '127.0.0.1')}:{int(n.get('expert_port', 9003))}"
                     for n in nodes if n.get("status") == "healthy" and n["node_id"] != nid}
            payload = {
                "node_id": nid,
                "peers": peers,
                "experts": {},
                "layer_caps": plan.layer_caps if plan else {},
                "blocks": plan.planned_blocks if plan else [],
            }
            self.registry.control_send(nid, placement_frame(payload).encode("utf-8"))

    def _run_step(self, cfg: BenchConfig, step: dict[str, Any]) -> dict[str, Any]:
        name = step["step"]
        wipe = step.get("wipe_usage", False)
        workers = step.get("workers") if step.get("workers") is not None else cfg.workers
        workers = max(1, int(workers or 1))
        self._set_progress(step=name, chat_index=0, chat_total=cfg.requests, last_error="")
        if wipe:
            self._wipe_usage()
            time.sleep(0.5)
        if cfg.local_only:
            self._push_local_only()
            time.sleep(0.5)
        body = self._chat_body(cfg)
        latencies: list[float] = []
        job_ids: list[str] = []
        errors = 0

        def one_chat(_: int) -> tuple[float | None, str | None]:
            try:
                status, elapsed, jid, _nid = self._proxy_chat(body)
                if status != 200:
                    return None, jid
                return elapsed, jid
            except OSError as err:
                return None, str(err)

        t0 = time.perf_counter()
        if workers <= 1:
            for i in range(cfg.requests):
                self._set_progress(chat_index=i + 1)
                elapsed, meta = one_chat(i)
                if elapsed is None:
                    errors += 1
                    if meta and not meta.startswith("{"):
                        self._set_progress(last_error=str(meta))
                else:
                    latencies.append(elapsed)
                    if meta:
                        job_ids.append(meta)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(one_chat, i): i for i in range(cfg.requests)}
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    done += 1
                    self._set_progress(chat_index=done)
                    elapsed, meta = fut.result()
                    if elapsed is None:
                        errors += 1
                        if meta and len(meta) > 8:
                            self._set_progress(last_error=str(meta))
                    else:
                        latencies.append(elapsed)
                        if meta:
                            job_ids.append(meta)
        elapsed_total = time.perf_counter() - t0
        node_ids = [n["node_id"] for n in self.registry.snapshot().get("nodes", [])]
        placement = self.scheduler.snapshot(node_ids)
        jobs = self._jobs_for_ids(job_ids)
        lat_summary = summarize_latencies(latencies)
        out: dict[str, Any] = {
            "step": name,
            "wipe_usage": wipe,
            "workers": workers,
            "requests": cfg.requests,
            "ok": len(latencies),
            "errors": errors,
            "elapsed_sec": round(elapsed_total, 3),
            "latency": lat_summary,
            "hop_mix_pct": hop_mix(jobs),
            "primary_node_mix_pct": primary_node_mix(jobs),
            "placement": placement,
            "jobs": jobs,
        }
        if workers > 1 and elapsed_total > 0:
            out["rps"] = round(len(latencies) / elapsed_total, 3)
        return out

    def _run(self, cfg: BenchConfig) -> None:
        try:
            steps_plan = PRESET_STEPS.get(cfg.preset)
            if not steps_plan:
                raise ValueError(f"unknown preset {cfg.preset!r}")
            if cfg.preset in ("cold_sequential",) and cfg.wipe_usage is None:
                cfg.wipe_usage = True
            elif cfg.wipe_usage is None:
                cfg.wipe_usage = False
            if len(steps_plan) == 1 and cfg.wipe_usage is not None:
                steps_plan = [{**steps_plan[0], "wipe_usage": cfg.wipe_usage}]
            finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            steps = [self._run_step(cfg, step) for step in steps_plan]
            cluster = cluster_snapshot(self.registry)
            scoreboard = build_scoreboard(steps, cluster.get("nodes") or []) if cfg.preset == "suite" or len(steps) > 1 else {}
            if not scoreboard and steps:
                step = steps[-1]
                warnings = donor_primary_warnings(cluster.get("nodes") or [], step.get("jobs") or [])
                scoreboard = {
                    "p50_cold_sec": (step.get("latency") or {}).get("p50_sec") if step["step"] == "cold_sequential" else None,
                    "p50_warm_sec": (step.get("latency") or {}).get("p50_sec") if step["step"] == "warm_sequential" else None,
                    "epa": None,
                    "rps_at_w": step.get("rps"),
                    "hop_mix_pct": step.get("hop_mix_pct"),
                    "primary_node_mix_pct": step.get("primary_node_mix_pct"),
                    "donor_primary_warnings": warnings,
                }
                if step["step"] == "cold_sequential":
                    scoreboard["p95_cold_sec"] = (step.get("latency") or {}).get("p95_sec")
                if step["step"] == "warm_sequential":
                    scoreboard["p95_warm_sec"] = (step.get("latency") or {}).get("p95_sec")
            result = {
                "finished_at": finished_at,
                "cluster": cluster,
                "meta": {
                    "preset": cfg.preset,
                    "wipe_usage": cfg.wipe_usage,
                    "local_only": cfg.local_only,
                    "requests": cfg.requests,
                    "workers": cfg.workers,
                    "max_tokens": cfg.max_tokens,
                    "prompt": cfg.prompt,
                },
                "steps": steps,
                "scoreboard": scoreboard,
                "last_error": self._progress.last_error,
            }
            result["markdown"] = render_markdown(result)
            self._persist(result)
            self._finish(result)
        except Exception as err:
            self._finish(error=str(err))

    def _persist(self, result: dict[str, Any]) -> None:
        out_dir = bench_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = result.get("finished_at", "").replace(":", "").replace("-", "")
        preset = (result.get("meta") or {}).get("preset", "bench")
        json_text = json.dumps(result, indent=2)
        md_text = render_markdown(result)
        (out_dir / "latest.json").write_text(json_text, encoding="utf-8")
        (out_dir / "latest.md").write_text(md_text, encoding="utf-8")
        if stamp:
            (out_dir / f"{stamp}-{preset}.json").write_text(json_text, encoding="utf-8")
            (out_dir / f"{stamp}-{preset}.md").write_text(md_text, encoding="utf-8")

    @staticmethod
    def read_latest() -> dict[str, Any] | None:
        path = bench_dir() / "latest.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
