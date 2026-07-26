import { useEffect, useState } from "react"
import { Cpu, HardDrive, MemoryStick, Network, Server } from "lucide-react"

import { getClusterNodes, type ClusterNode, type ClusterNodesResponse } from "@/lib/api"
import { Brain } from "./Brain"
import { Badge } from "@/components/ui/badge"
import { useLocale } from "./i18n"
import { cn } from "@/lib/utils"

function statusClass(status: string) {
  if (status === "healthy") return "badge-live"
  if (status === "stale") return "badge-speed"
  return ""
}

export function Cluster({ baseUrl, apiKey, connected }: { baseUrl: string; apiKey: string; connected: boolean }) {
  const { t } = useLocale()
  const [data, setData] = useState<ClusterNodesResponse | null>(null)
  const [error, setError] = useState("")
  const [isMaster, setIsMaster] = useState(false)

  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      try {
        const next = await getClusterNodes(baseUrl, apiKey)
        if (disposed) return
        setData(next)
        setIsMaster(true)
        setError("")
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

  if (!connected) {
    return <div className="empty-state"><p>{t("cluster.connectFirst")}</p></div>
  }

  if (!isMaster) {
    return <div className="empty-state"><p>{error || t("cluster.notMaster")}</p></div>
  }

  const nodes = data?.nodes || []

  return (
    <div className="cluster-view">
      <div className="cluster-summary">
        <Badge><Server className="size-3" /> {t("cluster.nodes", { n: nodes.length })}</Badge>
        <Badge className="badge-live"><Network className="size-3" /> {t("cluster.healthy", { n: data?.healthy ?? 0 })}</Badge>
      </div>
      <div className="cluster-grid">
        {nodes.map((node) => <NodeCard key={node.node_id} node={node} t={t} />)}
      </div>
      <div className="cluster-heatmaps">
        {nodes.filter((n) => n.emap?.rows).map((node) => (
          <section key={node.node_id} className="cluster-node-heatmap">
            <h3>{node.node_id.slice(0, 8)} · {node.endpoint}</h3>
            <Brain
              baseUrl={baseUrl}
              apiKey={apiKey}
              connected={connected}
              staticData={{
                rows: node.emap!.rows,
                cols: node.emap!.cols,
                map: node.emap!.map,
                hits: node.hits || "",
                seq: node.hits_seq || 0,
              }}
            />
          </section>
        ))}
      </div>
    </div>
  )
}

function NodeCard({ node, t }: { node: ClusterNode; t: (key: string, vars?: Record<string, string | number>) => string }) {
  const hw = node.hwinfo
  const tiers = node.tiers
  return (
    <article className={cn("cluster-node-card", node.status)}>
      <header>
        <strong>{node.node_id.slice(0, 8)}</strong>
        <Badge className={statusClass(node.status)}>{node.status}</Badge>
      </header>
      <div className="cluster-node-meta">{node.endpoint}</div>
      <div className="cluster-node-stats">
        <span><Cpu className="size-3" /> {hw?.cpu || t("cluster.unknown")}</span>
        <span><MemoryStick className="size-3" /> {hw ? `${hw.ram_avail_gb.toFixed(0)}/${hw.ram_total_gb.toFixed(0)} GB` : "—"}</span>
        <span><HardDrive className="size-3" /> {tiers ? `${tiers.ram} RAM · ${tiers.disk} disk` : "—"}</span>
      </div>
      <footer>
        {t("cluster.inflight", { n: node.inflight })}
        {" · "}
        {t("cluster.uptime", { s: Math.round(node.uptime_sec) })}
        {" · "}
        {t("cluster.heartbeat", { s: node.last_heartbeat_age_sec.toFixed(1) })}
      </footer>
    </article>
  )
}
