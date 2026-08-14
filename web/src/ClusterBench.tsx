import { useCallback, useEffect, useMemo, useState } from "react"
import { FlaskConical, Loader2 } from "lucide-react"

import {
  getClusterBench,
  startClusterBench,
  type ClusterBenchResponse,
  type ClusterBenchStartRequest,
} from "@/lib/api"
import { cn } from "@/lib/utils"

type BenchPreset = ClusterBenchStartRequest["preset"]

const PRESETS: { id: BenchPreset; labelKey: string; wipeDefault: boolean }[] = [
  { id: "cold_sequential", labelKey: "cluster.bench.preset.cold", wipeDefault: true },
  { id: "warm_sequential", labelKey: "cluster.bench.preset.warm", wipeDefault: false },
  { id: "concurrent", labelKey: "cluster.bench.preset.concurrent", wipeDefault: false },
  { id: "suite", labelKey: "cluster.bench.preset.suite", wipeDefault: true },
]

export function ClusterBenchPanel({
  baseUrl,
  apiKey,
  t,
}: {
  baseUrl: string
  apiKey: string
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const [bench, setBench] = useState<ClusterBenchResponse | null>(null)
  const [error, setError] = useState("")
  const [busy, setBusy] = useState(false)
  const [wipeUsage, setWipeUsage] = useState(true)
  const [localOnly, setLocalOnly] = useState(false)
  const [requests, setRequests] = useState(8)
  const [workers, setWorkers] = useState(4)
  const [maxTokens, setMaxTokens] = useState(32)
  const [copied, setCopied] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const snap = await getClusterBench(baseUrl, apiKey)
      setBench(snap)
      setError("")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [baseUrl, apiKey])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    const interval = bench?.status === "running" ? 1000 : 5000
    const id = window.setInterval(() => void refresh(), interval)
    return () => window.clearInterval(id)
  }, [bench?.status, refresh])

  const running = bench?.status === "running"
  const last = bench?.last_result
  const score = last?.scoreboard
  const markdown = useMemo(() => last?.markdown || "", [last?.markdown])

  const runPreset = async (preset: BenchPreset, wipeOverride?: boolean) => {
    const meta = PRESETS.find((p) => p.id === preset)
    setBusy(true)
    setError("")
    try {
      await startClusterBench(baseUrl, apiKey, {
        preset,
        wipe_usage: wipeOverride ?? (preset === "suite" ? undefined : wipeUsage),
        local_only: localOnly,
        requests,
        workers,
        max_tokens: maxTokens,
      })
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const copyMarkdown = async () => {
    if (!markdown) return
    await navigator.clipboard.writeText(markdown)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="cluster-spark-body">
      <section className="cluster-panel cluster-bench-panel">
        <header className="cluster-panel-head">
          <h3><FlaskConical className="size-4" /> {t("cluster.bench.title")}</h3>
          <p className="cluster-panel-sub">{t("cluster.bench.subtitle")}</p>
        </header>

        <div className="cluster-bench-presets">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              className="cluster-bench-preset"
              disabled={running || busy}
              onClick={() => {
                if (p.id !== "suite") setWipeUsage(p.wipeDefault)
                void runPreset(p.id, p.id === "suite" ? undefined : p.wipeDefault)
              }}
            >
              {t(p.labelKey)}
            </button>
          ))}
        </div>

        <div className="cluster-bench-form">
          <label className="cluster-bench-toggle">
            <input type="checkbox" checked={wipeUsage} onChange={(e) => setWipeUsage(e.target.checked)} disabled={running} />
            {t("cluster.bench.wipeUsage")}
          </label>
          <label className="cluster-bench-toggle">
            <input type="checkbox" checked={localOnly} onChange={(e) => setLocalOnly(e.target.checked)} disabled={running} />
            {t("cluster.bench.localOnly")}
          </label>
          <label>
            {t("cluster.bench.requests")}
            <input type="number" min={1} max={64} value={requests} onChange={(e) => setRequests(Number(e.target.value))} disabled={running} />
          </label>
          <label>
            {t("cluster.bench.workers")}
            <input type="number" min={1} max={32} value={workers} onChange={(e) => setWorkers(Number(e.target.value))} disabled={running} />
          </label>
          <label>
            {t("cluster.bench.maxTokens")}
            <input type="number" min={1} max={512} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} disabled={running} />
          </label>
        </div>

        <p className="cluster-bench-note">{t("cluster.bench.usageNote")}</p>

        {(running || bench?.progress?.last_error) && (
          <div className={cn("cluster-bench-progress", running && "live")}>
            {running ? <Loader2 className="size-4 animate-spin" /> : null}
            <span>
              {running
                ? t("cluster.bench.progress", {
                    step: bench?.progress?.step || bench?.progress?.preset || "…",
                    i: bench?.progress?.chat_index ?? 0,
                    n: bench?.progress?.chat_total ?? 0,
                  })
                : bench?.progress?.last_error}
            </span>
          </div>
        )}

        {error ? <p className="cluster-bench-error">{error}</p> : null}
      </section>

      {last ? (
        <section className="cluster-panel">
          <header className="cluster-panel-head cluster-bench-results-head">
            <h3>{t("cluster.bench.results")}</h3>
            {markdown ? (
              <button type="button" className="cluster-bench-copy" onClick={() => void copyMarkdown()}>
                {copied ? t("cluster.bench.copied") : t("cluster.bench.copyMarkdown")}
              </button>
            ) : null}
          </header>

          {score ? (
            <table className="cluster-table compact">
              <thead>
                <tr>
                  <th>{t("cluster.bench.metric")}</th>
                  <th>{t("cluster.bench.value")}</th>
                </tr>
              </thead>
              <tbody>
                {score.p50_cold_sec != null ? <tr><td>p50 cold (s)</td><td>{score.p50_cold_sec}</td></tr> : null}
                {score.p50_warm_sec != null ? <tr><td>p50 warm (s)</td><td>{score.p50_warm_sec}</td></tr> : null}
                {score.p95_cold_sec != null ? <tr><td>p95 cold (s)</td><td>{score.p95_cold_sec}</td></tr> : null}
                {score.p95_warm_sec != null ? <tr><td>p95 warm (s)</td><td>{score.p95_warm_sec}</td></tr> : null}
                {score.epa != null ? <tr><td>EPA</td><td>{score.epa}</td></tr> : null}
                {score.rps_at_w != null ? <tr><td>RPS@W</td><td>{score.rps_at_w}</td></tr> : null}
                {score.hop_mix_pct ? (
                  <tr>
                    <td>{t("cluster.bench.hopMix")}</td>
                    <td>
                      local {score.hop_mix_pct.local}% · remote {score.hop_mix_pct.remote}% · fallback {score.hop_mix_pct.fallback}%
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          ) : null}

          {last.steps?.length ? (
            <table className="cluster-table compact">
              <thead>
                <tr>
                  <th>{t("cluster.bench.step")}</th>
                  <th>{t("cluster.bench.ok")}</th>
                  <th>p50 (s)</th>
                  <th>p95 (s)</th>
                  <th>RPS</th>
                </tr>
              </thead>
              <tbody>
                {last.steps.map((step) => (
                  <tr key={step.step}>
                    <td>{step.step}</td>
                    <td>{step.ok}/{step.requests}</td>
                    <td>{step.latency?.p50_sec ?? "—"}</td>
                    <td>{step.latency?.p95_sec ?? "—"}</td>
                    <td>{step.rps ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}

          {markdown ? (
            <pre className="cluster-bench-markdown">{markdown}</pre>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}
