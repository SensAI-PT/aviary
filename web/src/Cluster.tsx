import { useEffect, useMemo, useState } from "react"
import {
  Activity,
  ArrowRight,
  ChevronRight,
  Cpu,
  HardDrive,
  Layers,
  MemoryStick,
  Network,
  Server,
  Timer,
  Zap,
} from "lucide-react"

import {
  getClusterOverview,
  type ClusterJob,
  type ClusterNode,
  type ClusterOverviewResponse,
  type ClusterPlacementResponse,
} from "@/lib/api"
import { ClusterHeatmapSection, ExpertHeatmapModal } from "./ClusterHeatmap"
import { Badge } from "@/components/ui/badge"
import { useLocale } from "./i18n"
import { cn } from "@/lib/utils"

type ClusterTab = "overview" | "jobs" | "executors" | "placement" | "rpc"

function shortId(id: string) {
  return id.slice(0, 8)
}

function formatDuration(sec: number) {
  if (sec < 1) return `${Math.round(sec * 1000)}ms`
  if (sec < 60) return `${sec.toFixed(1)}s`
  return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
}

function statusClass(status: string) {
  if (status === "healthy" || status === "completed") return "badge-live"
  if (status === "running") return "badge-speed"
  if (status === "stale") return "badge-warn"
  return ""
}

function layerBlocksForNode(placement: ClusterPlacementResponse | null, nodeId: string) {
  return placement?.blocks?.[nodeId] || []
}

function expertsForNode(placement: ClusterPlacementResponse | null, nodeId: string) {
  if (!placement?.experts) return []
  return Object.entries(placement.experts)
    .filter(([, nid]) => nid === nodeId)
    .map(([key, nid]) => ({ key, tier: placement.expert_tiers?.[key] ?? 0, nodeId: nid }))
    .sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }))
}

function remoteExpertsForNode(placement: ClusterPlacementResponse | null, nodeId: string) {
  if (!placement?.experts) return []
  return Object.entries(placement.experts)
    .filter(([, nid]) => nid !== nodeId)
    .map(([key, nid]) => ({ key, tier: placement.expert_tiers?.[key] ?? 0, nodeId: nid }))
}

export function Cluster({ baseUrl, apiKey, connected }: { baseUrl: string; apiKey: string; connected: boolean }) {
  const { t } = useLocale()
  const [overview, setOverview] = useState<ClusterOverviewResponse | null>(null)
  const [error, setError] = useState("")
  const [isMaster, setIsMaster] = useState(false)
  const [tab, setTab] = useState<ClusterTab>("overview")
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      try {
        const next = await getClusterOverview(baseUrl, apiKey)
        if (disposed) return
        setOverview(next)
        setIsMaster(true)
        setError("")
        setSelectedNodeId((cur) => cur || next.nodes.nodes[0]?.node_id || null)
      } catch {
        if (!disposed) {
          setIsMaster(false)
          setError(t("cluster.notMaster"))
        }
      }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [baseUrl, apiKey, connected, t])

  const nodes = overview?.nodes.nodes || []
  const placement = overview?.placement || null
  const jobs = overview?.jobs
  const selectedNode = nodes.find((n) => n.node_id === selectedNodeId) || null
  const selectedJob = [...(jobs?.active || []), ...(jobs?.history || [])].find((j) => j.job_id === selectedJobId) || null

  const allJobs = useMemo(() => {
    const active = jobs?.active || []
    const history = jobs?.history || []
    return [...active, ...history].sort((a, b) => b.started_at - a.started_at)
  }, [jobs])

  if (!connected) {
    return <div className="empty-state"><p>{t("cluster.connectFirst")}</p></div>
  }

  if (!isMaster) {
    return <div className="empty-state"><p>{error || t("cluster.notMaster")}</p></div>
  }

  const tabs: { id: ClusterTab; label: string; count?: number }[] = [
    { id: "overview", label: t("cluster.tab.overview") },
    { id: "jobs", label: t("cluster.tab.jobs"), count: jobs?.active_count },
    { id: "executors", label: t("cluster.tab.executors"), count: nodes.length },
    { id: "placement", label: t("cluster.tab.placement"), count: Object.keys(placement?.experts || {}).length },
    { id: "rpc", label: t("cluster.tab.rpc") },
  ]

  return (
    <div className="cluster-spark">
      <header className="cluster-spark-head">
        <div>
          <span className="eyebrow">{t("cluster.sparkTitle")}</span>
          <h2>{overview?.nodes.cohort?.model_id || t("cluster.sparkSubtitle")}</h2>
        </div>
        <div className="cluster-spark-tiles">
          <SummaryTile icon={Server} label={t("cluster.nodes", { n: nodes.length })} value={`${overview?.nodes.healthy ?? 0} ${t("cluster.healthyLabel")}`} />
          <SummaryTile icon={Activity} label={t("cluster.tab.jobs")} value={String(jobs?.active_count ?? 0)} live={!!jobs?.active_count} />
          <SummaryTile icon={Layers} label={t("cluster.hotExperts")} value={String(Object.keys(placement?.experts || {}).length)} />
          <SummaryTile icon={Timer} label={t("cluster.placementAge")} value={placement?.computed_at ? `${Math.round(Date.now() / 1000 - placement.computed_at)}s` : "—"} />
        </div>
      </header>

      <nav className="cluster-spark-nav">
        {tabs.map((item) => (
          <button key={item.id} className={cn(tab === item.id && "active")} onClick={() => setTab(item.id)}>
            {item.label}
            {item.count != null && item.count > 0 ? <span className="cluster-nav-count">{item.count}</span> : null}
          </button>
        ))}
      </nav>

      {tab === "overview" && (
        <OverviewPanel nodes={nodes} jobs={allJobs} placement={placement} onSelectNode={setSelectedNodeId} onSelectJob={(id) => { setSelectedJobId(id); setTab("jobs") }} onGoExecutors={() => setTab("executors")} t={t} />
      )}

      {tab === "jobs" && (
        <JobsPanel jobs={allJobs} nodes={nodes} selectedJobId={selectedJobId} onSelectJob={setSelectedJobId} onSelectNode={(id) => { setSelectedNodeId(id); setTab("executors") }} t={t} />
      )}

      {tab === "executors" && (
        <ExecutorsPanel nodes={nodes} placement={placement} selectedNode={selectedNode} selectedJob={selectedJob} onSelectNode={setSelectedNodeId} baseUrl={baseUrl} apiKey={apiKey} connected={connected} t={t} />
      )}

      {tab === "placement" && (
        <PlacementPanel nodes={nodes} placement={placement} selectedNodeId={selectedNodeId} onSelectNode={setSelectedNodeId} t={t} />
      )}

      {tab === "rpc" && (
        <RpcPanel nodes={nodes} placement={placement} t={t} />
      )}
    </div>
  )
}

function SummaryTile({ icon: Icon, label, value, live }: { icon: typeof Server; label: string; value: string; live?: boolean }) {
  return (
    <div className={cn("cluster-spark-tile", live && "live")}>
      <Icon className="size-4" />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  )
}

function OverviewPanel({ nodes, jobs, placement, onSelectNode, onSelectJob, onGoExecutors, t }: {
  nodes: ClusterNode[]
  jobs: ClusterJob[]
  placement: ClusterPlacementResponse | null
  onSelectNode: (id: string) => void
  onSelectJob: (id: string) => void
  onGoExecutors: () => void
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const running = jobs.filter((j) => j.status === "running")
  return (
    <div className="cluster-spark-body">
      <section className="cluster-panel">
        <header className="cluster-panel-head">
          <h3><Activity className="size-4" /> {t("cluster.runningJobs")}</h3>
          <button type="button" className="cluster-link" onClick={() => onSelectJob(running[0]?.job_id || jobs[0]?.job_id || "")}>{t("cluster.viewAllJobs")} <ChevronRight className="size-3" /></button>
        </header>
        {running.length ? (
          <div className="cluster-job-cards">
            {running.map((job) => (
              <button key={job.job_id} type="button" className="cluster-job-card running" onClick={() => onSelectJob(job.job_id)}>
                <div className="cluster-job-card-top">
                  <Badge className="badge-speed"><Zap className="size-3" /> {t("cluster.jobRunning")}</Badge>
                  <code>{shortId(job.job_id)}</code>
                </div>
                <div className="cluster-job-card-route">
                  <span>{job.path}</span>
                  <ArrowRight className="size-3" />
                  <span className="cluster-node-link" onClick={(e) => { e.stopPropagation(); onSelectNode(job.node_id) }}>{shortId(job.node_id)}</span>
                </div>
                <footer>{formatDuration(job.duration_sec)} · {job.node_endpoint}</footer>
              </button>
            ))}
          </div>
        ) : (
          <p className="cluster-empty">{t("cluster.noRunningJobs")}</p>
        )}
      </section>

      <section className="cluster-panel">
        <header className="cluster-panel-head">
          <h3><Server className="size-4" /> {t("cluster.tab.executors")}</h3>
          <button type="button" className="cluster-link" onClick={onGoExecutors}>{t("cluster.viewExecutors")} <ChevronRight className="size-3" /></button>
        </header>
        <div className="cluster-executor-strip">
          {nodes.map((node) => (
            <button key={node.node_id} type="button" className={cn("cluster-executor-chip", node.status)} onClick={() => onSelectNode(node.node_id)}>
              <strong>{shortId(node.node_id)}</strong>
              <span>{node.inflight} {t("cluster.inflightShort")}</span>
              <Badge className={statusClass(node.status)}>{node.status}</Badge>
            </button>
          ))}
        </div>
      </section>

      <section className="cluster-panel">
        <header className="cluster-panel-head">
          <h3><Layers className="size-4" /> {t("cluster.topUsage")}</h3>
        </header>
        <div className="cluster-usage-list">
          {(placement?.usage_top || []).slice(0, 12).map((u) => {
            const key = `${u.layer}:${u.expert}`
            const owner = placement?.experts?.[key]
            return (
              <div key={key} className="cluster-usage-row">
                <code>L{u.layer} E{u.expert}</code>
                <span>{u.count.toLocaleString()} routes</span>
                <span className="cluster-usage-owner">{owner ? `→ ${shortId(owner)}` : "—"}</span>
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}

function JobsPanel({ jobs, nodes, selectedJobId, onSelectJob, onSelectNode, t }: {
  jobs: ClusterJob[]
  nodes: ClusterNode[]
  selectedJobId: string | null
  onSelectJob: (id: string) => void
  onSelectNode: (id: string) => void
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const selected = jobs.find((j) => j.job_id === selectedJobId)
  const nodeMap = Object.fromEntries(nodes.map((n) => [n.node_id, n]))
  return (
    <div className="cluster-spark-split">
      <div className="cluster-spark-main">
        <table className="cluster-table">
          <thead>
            <tr>
              <th>{t("cluster.col.status")}</th>
              <th>{t("cluster.col.jobId")}</th>
              <th>{t("cluster.col.path")}</th>
              <th>{t("cluster.col.executor")}</th>
              <th>{t("cluster.col.duration")}</th>
            </tr>
          </thead>
          <tbody>
            {jobs.length ? jobs.map((job) => (
              <tr key={job.job_id} className={cn(selectedJobId === job.job_id && "selected", job.status === "running" && "running")} onClick={() => onSelectJob(job.job_id)}>
                <td><Badge className={statusClass(job.status)}>{job.status}</Badge></td>
                <td><code>{shortId(job.job_id)}</code></td>
                <td>{job.path}</td>
                <td><button type="button" className="cluster-inline-link" onClick={(e) => { e.stopPropagation(); onSelectNode(job.node_id) }}>{shortId(job.node_id)}</button></td>
                <td>{formatDuration(job.duration_sec)}</td>
              </tr>
            )) : (
              <tr><td colSpan={5} className="cluster-empty">{t("cluster.noJobsYet")}</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <aside className="cluster-spark-detail">
        <h3>{t("cluster.jobDetail")}</h3>
        {selected ? (
          <dl className="cluster-kv">
            <dt>{t("cluster.col.jobId")}</dt><dd><code>{selected.job_id}</code></dd>
            <dt>{t("cluster.col.status")}</dt><dd><Badge className={statusClass(selected.status)}>{selected.status}</Badge></dd>
            <dt>{t("cluster.col.path")}</dt><dd>{selected.path}</dd>
            <dt>{t("cluster.col.executor")}</dt><dd><button type="button" className="cluster-inline-link" onClick={() => onSelectNode(selected.node_id)}>{shortId(selected.node_id)}</button></dd>
            <dt>{t("cluster.col.endpoint")}</dt><dd>{selected.node_endpoint}</dd>
            <dt>{t("cluster.col.duration")}</dt><dd>{formatDuration(selected.duration_sec)}</dd>
            {selected.http_status != null ? <><dt>HTTP</dt><dd>{selected.http_status}</dd></> : null}
            {selected.error ? <><dt>{t("cluster.col.error")}</dt><dd className="cluster-error-text">{selected.error}</dd></> : null}
            {nodeMap[selected.node_id]?.arch ? <><dt>{t("cluster.col.arch")}</dt><dd>{nodeMap[selected.node_id].arch}</dd></> : null}
            {selected.trace?.length ? (
              <>
                <dt>{t("cluster.jobTraceSummary")}</dt>
                <dd><JobTraceSummary trace={selected.trace} /></dd>
                <dt>{t("cluster.jobExpertHops")}</dt>
                <dd><JobExpertHopTable trace={selected.trace} t={t} /></dd>
                <dt>{t("cluster.jobTrace")}</dt>
                <dd><JobTraceGantt job={selected} /></dd>
              </>
            ) : null}
          </dl>
        ) : (
          <p className="cluster-empty">{t("cluster.selectJob")}</p>
        )}
      </aside>
    </div>
  )
}

function ExecutorsPanel({ nodes, placement, selectedNode, selectedJob, onSelectNode, baseUrl, apiKey, connected, t }: {
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  selectedNode: ClusterNode | null
  selectedJob: ClusterJob | null
  onSelectNode: (id: string) => void
  baseUrl: string
  apiKey: string
  connected: boolean
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const [heatmapOpen, setHeatmapOpen] = useState(false)
  const owned = selectedNode ? expertsForNode(placement, selectedNode.node_id) : []
  const remotes = selectedNode ? remoteExpertsForNode(placement, selectedNode.node_id).slice(0, 16) : []
  useEffect(() => { setHeatmapOpen(false) }, [selectedNode?.node_id])
  return (
    <div className="cluster-executors-layout">
      <div className="cluster-spark-split">
        <div className="cluster-executor-list">
          {nodes.map((node) => (
            <button key={node.node_id} type="button" className={cn("cluster-executor-row", selectedNode?.node_id === node.node_id && "selected", node.status)} onClick={() => onSelectNode(node.node_id)}>
              <div className="cluster-executor-row-top">
                <strong>{shortId(node.node_id)}</strong>
                <Badge className={statusClass(node.status)}>{node.status}</Badge>
              </div>
              <div className="cluster-executor-row-meta">{node.endpoint}</div>
              <div className="cluster-executor-row-stats">
                <span><Activity className="size-3" /> {node.inflight}</span>
                <span><HardDrive className="size-3" /> {node.tiers ? `${node.tiers.ram} RAM` : "—"}</span>
                <span><Layers className="size-3" /> {expertsForNode(placement, node.node_id).length} hot</span>
              </div>
            </button>
          ))}
        </div>
        <div className="cluster-spark-detail wide">
          {!selectedNode ? (
            <p className="cluster-empty">{t("cluster.selectExecutor")}</p>
          ) : (
            <>
              <header className="cluster-detail-head">
                <div>
                  <h3>{shortId(selectedNode.node_id)}</h3>
                  <p>{selectedNode.endpoint}</p>
                </div>
                {selectedJob?.node_id === selectedNode.node_id ? (
                  <Badge className="badge-speed"><Zap className="size-3" /> {t("cluster.servingJob")} {shortId(selectedJob.job_id)}</Badge>
                ) : null}
              </header>

              <div className="cluster-detail-grid">
                <div><span>{t("cluster.col.arch")}</span><strong>{selectedNode.arch || "—"}</strong></div>
                <div><span>{t("cluster.inflight")}</span><strong>{selectedNode.inflight}</strong></div>
                <div><span>{t("cluster.uptime")}</span><strong>{Math.round(selectedNode.uptime_sec)}s</strong></div>
                <div><span>{t("cluster.heartbeat")}</span><strong>{selectedNode.last_heartbeat_age_sec.toFixed(1)}s</strong></div>
              </div>

              {selectedNode.hwinfo ? (
                <div className="cluster-detail-hw">
                  <span><Cpu className="size-3" /> {selectedNode.hwinfo.cpu || t("cluster.unknown")}</span>
                  <span><MemoryStick className="size-3" /> {selectedNode.hwinfo.ram_avail_gb.toFixed(0)}/{selectedNode.hwinfo.ram_total_gb.toFixed(0)} GB</span>
                  {selectedNode.tiers ? <span><HardDrive className="size-3" /> {selectedNode.tiers.ram} RAM · {selectedNode.tiers.disk} disk</span> : null}
                </div>
              ) : null}

              <section className="cluster-subsection">
                <h4>{t("cluster.ownedExperts")} ({owned.length})</h4>
                <div className="cluster-expert-chips">
                  {owned.length ? owned.slice(0, 48).map((e) => (
                    <span key={e.key} className={cn("cluster-expert-chip", e.tier >= 2 ? "vram" : e.tier >= 1 ? "ram" : "disk")}>{e.key}</span>
                  )) : <span className="cluster-empty-inline">{t("cluster.noOwnedExperts")}</span>}
                </div>
              </section>

              {remotes.length ? (
                <section className="cluster-subsection">
                  <h4>{t("cluster.remoteTargets")}</h4>
                  <div className="cluster-expert-chips">
                    {remotes.map((e) => (
                      <span key={e.key} className="cluster-expert-chip remote">{e.key} → {shortId(e.nodeId)}</span>
                    ))}
                  </div>
                </section>
              ) : null}

              {selectedNode.emap?.rows ? (
                <section className="cluster-subsection">
                  <button type="button" className="cluster-heatmap-launch" onClick={() => setHeatmapOpen(true)}>
                    <div>
                      <h4>{t("cluster.expertHeatmap")}</h4>
                      <p>{t("cluster.heatmap.openHint")}</p>
                    </div>
                    <ChevronRight className="size-4" />
                  </button>
                  <ExpertHeatmapModal
                    open={heatmapOpen}
                    onClose={() => setHeatmapOpen(false)}
                    node={selectedNode}
                    nodes={nodes}
                    placement={placement}
                    baseUrl={baseUrl}
                    apiKey={apiKey}
                    connected={connected}
                    t={t}
                  />
                </section>
              ) : null}
            </>
          )}
        </div>
      </div>
      <ClusterHeatmapSection nodes={nodes} placement={placement} selectedNode={selectedNode} t={t} />
    </div>
  )
}

function PlacementPanel({ nodes, placement, selectedNodeId, onSelectNode, t }: {
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  selectedNodeId: string | null
  onSelectNode: (id: string) => void
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const byNode = nodes.map((node) => ({
    node,
    experts: expertsForNode(placement, node.node_id),
    inbound: Object.entries(placement?.experts || {}).filter(([, nid]) => nid === node.node_id).length,
  }))
  return (
    <div className="cluster-spark-body">
      <ClusterHeatmapSection nodes={nodes} placement={placement} t={t} />
      <div className="cluster-placement-grid">
        {byNode.map(({ node, experts, inbound }) => (
          <section key={node.node_id} className={cn("cluster-placement-card", selectedNodeId === node.node_id && "selected")} onClick={() => onSelectNode(node.node_id)}>
            <header>
              <strong>{shortId(node.node_id)}</strong>
              <Badge className={statusClass(node.status)}>{node.status}</Badge>
            </header>
            <p>{node.host}:{node.http_port}</p>
            <div className="cluster-placement-stats">
              <span>{inbound} {t("cluster.assignedExperts")}</span>
              <span>{experts.filter((e) => e.tier >= 1).length} {t("cluster.residentExperts")}</span>
              {layerBlocksForNode(placement, node.node_id).length ? (
                <span>{layerBlocksForNode(placement, node.node_id).length} {t("cluster.layerBlocks")}</span>
              ) : null}
            </div>
            <div className="cluster-layer-blocks">
              {layerBlocksForNode(placement, node.node_id).map((b) => (
                <span key={`${b.start}-${b.end}`} className="cluster-layer-block">L{b.start}–{b.end}</span>
              ))}
            </div>
            <div className="cluster-expert-chips compact">
              {experts.slice(0, 24).map((e) => (
                <span key={e.key} className={cn("cluster-expert-chip", e.tier >= 2 ? "vram" : e.tier >= 1 ? "ram" : "disk")}>{e.key}</span>
              ))}
              {experts.length > 24 ? <span className="cluster-more">+{experts.length - 24}</span> : null}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}

function RpcPanel({ nodes, placement, t }: {
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const ids = nodes.map((n) => n.node_id)
  const matrix = placement?.rpc_matrix_us || {}
  const hist = placement?.rpc_histogram
  const pairs = ids.flatMap((src) => ids.filter((dst) => src !== dst).map((dst) => ({
    src, dst, us: matrix[src]?.[dst] ?? null,
  }))).filter((p) => p.us != null) as Array<{ src: string; dst: string; us: number }>
  const maxUs = pairs.length ? Math.max(...pairs.map((p) => p.us)) : 1
  const maxBucket = hist?.buckets?.length ? Math.max(...hist.buckets.map((b) => b.count), 1) : 1
  return (
    <div className="cluster-spark-body">
      {hist && hist.count > 0 ? (
        <section className="cluster-panel">
          <header className="cluster-panel-head">
            <h3><Timer className="size-4" /> {t("cluster.rpcHistogram")}</h3>
          </header>
          <p className="cluster-hist-meta">
            {hist.count} samples · p50 {Math.round(hist.p50_us)}µs · p95 {Math.round(hist.p95_us)}µs
          </p>
          <div className="cluster-rpc-bars">
            {hist.buckets.map((b) => (
              <div key={b.max_us} className="cluster-rpc-row">
                <code>≤{b.max_us}µs</code>
                <div className="cluster-rpc-bar-wrap">
                  <div className="cluster-rpc-bar hist" style={{ width: `${Math.max(4, (b.count / maxBucket) * 100)}%` }} />
                </div>
                <span>{b.count}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
      <section className="cluster-panel">
        <header className="cluster-panel-head"><h3><Network className="size-4" /> {t("cluster.rpcMatrix")}</h3></header>
        {pairs.length ? (
          <div className="cluster-rpc-bars">
            {pairs.sort((a, b) => a.us - b.us).map((p) => (
              <div key={`${p.src}-${p.dst}`} className="cluster-rpc-row">
                <code>{shortId(p.src)} → {shortId(p.dst)}</code>
                <div className="cluster-rpc-bar-wrap"><div className="cluster-rpc-bar" style={{ width: `${Math.max(4, (p.us / maxUs) * 100)}%` }} /></div>
                <span>{Math.round(p.us)} µs</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="cluster-empty">{t("cluster.noRpcData")}</p>
        )}
      </section>
      {ids.length > 1 ? (
        <section className="cluster-panel">
          <header className="cluster-panel-head"><h3>{t("cluster.rpcTable")}</h3></header>
          <div className="cluster-rpc-table-wrap">
            <table className="cluster-table compact">
              <thead><tr><th>from \ to</th>{ids.map((id) => <th key={id}>{shortId(id)}</th>)}</tr></thead>
              <tbody>
                {ids.map((src) => (
                  <tr key={src}>
                    <th>{shortId(src)}</th>
                    {ids.map((dst) => (
                      <td key={dst}>{src === dst ? "—" : matrix[src]?.[dst] != null ? `${Math.round(matrix[src][dst])}` : "·"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  )
}

function JobTraceSummary({ trace }: { trace: ClusterJob["trace"] }) {
  const events = trace || []
  const local = events.filter((e) => e.kind === "local" || e.local).length
  const remote = events.filter((e) => e.kind === "remote").length
  const fallback = events.filter((e) => e.kind === "fallback").length
  const layers = new Set(events.map((e) => e.layer)).size
  return (
    <div className="cluster-trace-summary">
      <span>{local} local</span>
      <span>{remote} remote</span>
      <span>{fallback} fallback</span>
      <span>{layers} layers</span>
    </div>
  )
}

function JobExpertHopTable({ trace, t }: { trace: ClusterJob["trace"]; t: (key: string) => string }) {
  const events = trace || []
  if (!events.length) return null
  return (
    <div className="cluster-trace-table-wrap">
      <table className="cluster-trace-table">
        <thead>
          <tr>
            <th>{t("cluster.col.layer")}</th>
            <th>{t("cluster.col.expert")}</th>
            <th>{t("cluster.col.kind")}</th>
            <th>{t("cluster.col.executor")}</th>
            <th>{t("cluster.col.rpcUs")}</th>
          </tr>
        </thead>
        <tbody>
          {events.map((ev, i) => (
            <tr key={`${ev.ts}-${i}`}>
              <td>L{ev.layer}</td>
              <td>E{ev.expert}</td>
              <td><code>{ev.kind}</code></td>
              <td><code>{ev.local ? "local" : shortId(ev.node_id)}</code></td>
              <td>{ev.rpc_us != null ? `${Math.round(ev.rpc_us)}µs` : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function JobTraceGantt({ job }: { job: ClusterJob }) {
  const trace = job.trace || []
  if (!trace.length) return null
  const t0 = job.started_at
  const t1 = job.ended_at || trace[trace.length - 1]?.ts || t0 + job.duration_sec
  const span = Math.max(0.001, t1 - t0)
  return (
    <div className="cluster-gantt">
      <div className="cluster-gantt-primary">
        <span>{shortId(job.node_id)} · primary</span>
      </div>
      {trace.map((ev, i) => (
        <div
          key={`${ev.ts}-${i}`}
          className={cn("cluster-gantt-seg", ev.kind === "remote" ? "remote" : ev.kind === "fallback" ? "fallback" : "local")}
          style={{ marginLeft: `${Math.max(0, Math.min(92, ((ev.ts - t0) / span) * 100))}%` }}
          title={`${ev.kind} L${ev.layer} E${ev.expert}${ev.rpc_us ? ` · ${Math.round(ev.rpc_us)}µs` : ""}`}
        >
          <code>{ev.kind} L{ev.layer}:E{ev.expert}</code>
        </div>
      ))}
    </div>
  )
}
