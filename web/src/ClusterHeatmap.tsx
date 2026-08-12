import { useEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Layers, X } from "lucide-react"

import type { ClusterNode, ClusterPlacementResponse } from "@/lib/api"
import { Brain } from "./Brain"
import {
  clusterOwnerGrid,
  clusterResidentGrid,
  decodeEmap,
  hexToRgb,
  type HeatmapMode,
  layerDominanceStats,
  maxEmapDims,
  nodeColor,
  NODE_PALETTE,
  overallLayerCoherence,
  TIER_RGB,
} from "./clusterHeatmap"
import { cn } from "@/lib/utils"

const TIER_KEYS = ["tier.disk", "tier.ram", "tier.vram"] as const
const TIER_LABEL_COLORS = ["#8b9aa3", "#5a9bd8", "#4ed6a5"] as const

interface AtlasEntry { affinity: Record<string, number>; entropy: number; top: string; label: string }

function depthRoleKey(row: number, rows: number, isMtp: boolean): string {
  if (isMtp) return "brain.mtp"
  const f = row / Math.max(rows - 1, 1)
  if (f < 0.2) return "brain.early"
  if (f < 0.45) return "brain.lowerMiddle"
  if (f < 0.7) return "brain.upperMiddle"
  if (f < 0.9) return "brain.late"
  return "brain.final"
}

function realLayerIndex(row: number, rows: number) {
  const isMtp = row === rows - 1
  return { isMtp, layer: isMtp ? 78 : row + 3 }
}

function ExpertCellTip({ tip, rows, atlas, t }: {
  tip: { x: number; y: number; layer: number; expert: number; tier: number; heat: number } | null
  rows: number
  atlas: Record<string, AtlasEntry> | null
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  if (!tip) return null
  const { isMtp, layer } = realLayerIndex(tip.layer, rows)
  const entry = atlas?.[`${layer}:${tip.expert}`]
  return createPortal(
    <div
      className="brain-tip cluster-expert-tip"
      style={{ left: Math.min(tip.x + 14, window.innerWidth - 260), top: Math.min(tip.y + 14, window.innerHeight - 170) }}
    >
      <div className="brain-tip-title"><Layers className="size-3" /> Layer {layer}{isMtp ? " (MTP)" : ""} · Expert {tip.expert}</div>
      <div>Tier: <strong style={{ color: TIER_LABEL_COLORS[tip.tier] ?? TIER_LABEL_COLORS[0] }}>{t(TIER_KEYS[tip.tier] ?? TIER_KEYS[0])}</strong></div>
      <div>Heat: <strong>{tip.heat === 0 ? t("brain.neverRouted") : t("brain.selections", { heat: tip.heat })}</strong></div>
      {entry ? (
        <>
          <div className={entry.label.startsWith("specialist") ? "brain-tip-spec" : undefined}>
            {entry.label.startsWith("specialist") ? t("brain.specialist", { top: entry.top }) : t("brain.generalist")}
            <small> (entropy {entry.entropy})</small>
          </div>
          <div className="brain-tip-aff">
            {Object.entries(entry.affinity).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([c, p]) => `${c} ${Math.round(p * 100)}%`).join(" · ")}
          </div>
        </>
      ) : (
        <div className="brain-tip-role">{t(depthRoleKey(tip.layer, rows, isMtp))}</div>
      )}
    </div>,
    document.body,
  )
}

function shortId(id: string) {
  return id.slice(0, 8)
}

function tierCellColor(tier: number, heat: number) {
  const [R, G, B] = TIER_RGB[tier] ?? TIER_RGB[0]
  const lum = 0.35 + 0.65 * Math.min(heat / 24, 1)
  return `rgb(${(R * lum) | 0},${(G * lum) | 0},${(B * lum) | 0})`
}

function ownerCellColor(nodeId: string, nodeIds: string[], unassigned = "#1a2228") {
  if (!nodeId) return unassigned
  const [r, g, b] = hexToRgb(nodeColor(nodeId, nodeIds))
  return `rgb(${r},${g},${b})`
}

type HeatmapPanelProps = {
  mode: HeatmapMode
  node?: ClusterNode | null
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  compact?: boolean
  minHeight?: number
  large?: boolean
  t: (key: string, vars?: Record<string, string | number>) => string
}

/** Viewport height for the shared scrollable heatmap container (content may be larger). */
const DEFAULT_HEATMAP_H = 420
const LARGE_HEATMAP_H = 480
const MODAL_HEATMAP_H = 560

export function HeatmapModeToggle({ mode, onChange, t, modes }: {
  mode: HeatmapMode
  onChange: (m: HeatmapMode) => void
  t: (key: string) => string
  modes?: HeatmapMode[]
}) {
  const all: { id: HeatmapMode; label: string }[] = [
    { id: "resident", label: t("cluster.heatmap.resident") },
    { id: "byLayer", label: t("cluster.heatmap.byLayer") },
    { id: "clusterOwner", label: t("cluster.heatmap.clusterOwner") },
    { id: "clusterResident", label: t("cluster.heatmap.clusterResident") },
    { id: "layerDominance", label: t("cluster.heatmap.layerDominance") },
  ]
  const shown = modes ? all.filter((m) => modes.includes(m.id)) : all
  return (
    <div className="cluster-heatmap-modes">
      {shown.map((m) => (
        <button key={m.id} type="button" className={cn(mode === m.id && "active")} onClick={() => onChange(m.id)}>
          {m.label}
        </button>
      ))}
    </div>
  )
}

export function ExpertHeatmapPanel({ mode, node, nodes, placement, compact, minHeight, large, t }: HeatmapPanelProps) {
  const height = minHeight ?? (large ? LARGE_HEATMAP_H : DEFAULT_HEATMAP_H)
  const nodeIds = useMemo(() => nodes.map((n) => n.node_id), [nodes])
  const dims = useMemo(() => {
    if (mode === "resident" || mode === "byLayer") {
      return node?.emap ? { rows: node.emap.rows, cols: node.emap.cols } : { rows: 0, cols: 0 }
    }
    return maxEmapDims(nodes)
  }, [mode, node, nodes])

  if (!dims.rows || !dims.cols) {
    return <p className="cluster-empty-inline">{t("cluster.heatmap.noData")}</p>
  }

  if (mode === "byLayer" && node?.emap) {
    return <LayerStripHeatmap node={node} minHeight={height} large={large} t={t} />
  }

  if (mode === "layerDominance") {
    const stats = layerDominanceStats(placement, nodeIds)
    const coherence = overallLayerCoherence(stats)
    return (
      <div className="cluster-layer-dominance">
        <p className="cluster-heatmap-coherence">
          {t("cluster.heatmap.coherenceScore", {
            score: Math.round(coherence.score * 100),
            multi: coherence.multiNodeLayers,
            total: coherence.totalLayers,
          })}
        </p>
        <div className={cn("cluster-heatmap-scroll", large && "large")} style={{ height }}>
          <div className="cluster-layer-dominance-grid">
            {stats.map((row) => (
              <div key={row.layer} className="cluster-layer-dominance-row" title={`L${row.layer}: ${Object.keys(row.nodeCounts).length} nodes`}>
                <code className="cluster-layer-label">L{row.layer}</code>
                <div className="cluster-layer-dominance-bar">
                  {Object.entries(row.nodeCounts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([nid, count]) => (
                      <span
                        key={nid}
                        style={{
                          flex: count,
                          background: nodeColor(nid, nodeIds),
                        }}
                        title={`${shortId(nid)}: ${count} (${Math.round((count / row.assigned) * 100)}%)`}
                      />
                    ))}
                </div>
                <span className="cluster-layer-dominance-meta">
                  {row.dominant ? shortId(row.dominant) : "—"} · {Math.round(row.share * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <CanvasHeatmap
      mode={mode}
      node={node}
      nodes={nodes}
      placement={placement}
      rows={dims.rows}
      cols={dims.cols}
      nodeIds={nodeIds}
      compact={compact}
      minHeight={height}
      large={large}
      t={t}
    />
  )
}

function CanvasHeatmap({ mode, node, nodes, placement, rows, cols, nodeIds, compact, minHeight = DEFAULT_HEATMAP_H, large, t }: {
  mode: HeatmapMode
  node?: ClusterNode | null
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  rows: number
  cols: number
  nodeIds: string[]
  compact?: boolean
  minHeight?: number
  large?: boolean
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [tip, setTip] = useState<{ x: number; y: number; layer: number; expert: number; detail: string } | null>(null)
  const [wrapSize, setWrapSize] = useState({ w: 960, h: minHeight })

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setWrapSize({ w: Math.max(120, el.clientWidth - 24), h: Math.max(120, el.clientHeight - 24) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [minHeight])

  // Readable cells; overflow scrolls inside the fixed-height container.
  const cell = useMemo(() => {
    const minCell = large ? 8 : 6
    const maxCell = large ? 18 : 14
    return Math.max(minCell, Math.min(maxCell, Math.floor(wrapSize.w / cols)))
  }, [large, wrapSize.w, cols])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const gap = cell >= 5 ? 1 : 0
    canvas.width = cols * (cell + gap)
    canvas.height = rows * (cell + gap)
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const resident = mode === "resident" && node?.emap
      ? decodeEmap(node.emap.map, rows, cols)
      : mode === "clusterResident"
        ? clusterResidentGrid(nodes)?.cells
        : null
    const owners = mode === "clusterOwner" ? clusterOwnerGrid(placement, rows, cols) : null

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c
        let fill: string
        if (resident) {
          fill = tierCellColor(resident[i].tier, resident[i].heat)
        } else if (owners) {
          fill = ownerCellColor(owners[i], nodeIds)
        } else {
          fill = "#1a2228"
        }
        ctx.fillStyle = fill
        ctx.fillRect(c * (cell + gap), r * (cell + gap), cell, cell)
      }
    }
  }, [mode, node, nodes, placement, rows, cols, cell, nodeIds])

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const scaleX = e.currentTarget.width / rect.width
    const scaleY = e.currentTarget.height / rect.height
    const gap = cell >= 5 ? 1 : 0
    const col = Math.floor(((e.clientX - rect.left) * scaleX) / (cell + gap))
    const row = Math.floor(((e.clientY - rect.top) * scaleY) / (cell + gap))
    if (row < 0 || row >= rows || col < 0 || col >= cols) { setTip(null); return }
    const i = row * cols + col
    let detail = ""
    if (mode === "resident" && node?.emap) {
      const byte = parseInt(node.emap.map.substr(i * 2, 2), 16) || 0
      detail = `tier ${byte >> 6} · heat ${byte & 63}`
    } else if (mode === "clusterOwner") {
      const owners = clusterOwnerGrid(placement, rows, cols)
      detail = owners[i] ? `owner ${shortId(owners[i])}` : t("cluster.heatmap.unassigned")
    } else if (mode === "clusterResident") {
      const grid = clusterResidentGrid(nodes)
      const c = grid?.cells[i]
      detail = c ? `best tier ${c.tier} · heat ${c.heat}` : ""
    }
    setTip({ x: e.clientX, y: e.clientY, layer: row, expert: col, detail })
  }

  return (
    <div className={cn("cluster-heatmap-scroll", compact && "compact", large && "large")} ref={wrapRef} style={{ height: minHeight }}>
      {!compact ? (
        <div className="cluster-heatmap-legend">
          {mode === "clusterOwner" ? nodeIds.map((nid, i) => (
            <span key={nid}><i style={{ background: NODE_PALETTE[i % NODE_PALETTE.length] }} /> {shortId(nid)}</span>
          )) : (
            <>
              <span><i style={{ background: "#4ed6a5" }} /> {t("tier.vram")}</span>
              <span><i style={{ background: "#5a9bd8" }} /> {t("tier.ram")}</span>
              <span><i style={{ background: "#3a4750" }} /> {t("tier.disk")}</span>
            </>
          )}
        </div>
      ) : null}
      <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setTip(null)} />
      {tip ? (
        <div className="cluster-heatmap-tip" style={{ left: Math.min(tip.x + 12, window.innerWidth - 200), top: tip.y + 12 }}>
          <strong>L{tip.layer} · E{tip.expert}</strong>
          <span>{tip.detail}</span>
        </div>
      ) : null}
    </div>
  )
}

function LayerStripHeatmap({ node, minHeight, large, t }: { node: ClusterNode; minHeight: number; large?: boolean; t: (key: string, vars?: Record<string, string | number>) => string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [wrapSize, setWrapSize] = useState({ w: 900, h: minHeight })
  const [tip, setTip] = useState<{ x: number; y: number; layer: number; expert: number; tier: number; heat: number } | null>(null)
  const [atlas, setAtlas] = useState<Record<string, AtlasEntry> | null>(null)

  useEffect(() => {
    fetch("/experts.json").then((r) => (r.ok ? r.json() : null)).then((d) => { if (d?.experts) setAtlas(d.experts) }).catch(() => {})
  }, [])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWrapSize({ w: Math.max(120, el.clientWidth - 24), h: minHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [minHeight])

  const { rows, cols, map } = node.emap!
  const cells = decodeEmap(map, rows, cols)
  const cellSize = useMemo(
    () => Math.max(large ? 4 : 2, Math.min(large ? 12 : 8, Math.floor(wrapSize.w / cols))),
    [wrapSize.w, cols, large],
  )
  const cellGap = cellSize >= 4 ? 1 : 0

  const showCell = (layer: number, expert: number, c: { tier: number; heat: number }) => (e: React.MouseEvent<HTMLSpanElement>) => {
    setTip({ x: e.clientX, y: e.clientY, layer, expert, tier: c.tier, heat: c.heat })
  }

  return (
    <div ref={wrapRef} className={cn("cluster-heatmap-scroll", "cluster-layer-strips", large && "large")} style={{ height: minHeight }}>
      <div className="cluster-heatmap-legend">
        <span><i style={{ background: "#4ed6a5" }} /> {t("tier.vram")}</span>
        <span><i style={{ background: "#5a9bd8" }} /> {t("tier.ram")}</span>
        <span><i style={{ background: "#3a4750" }} /> {t("tier.disk")}</span>
      </div>
      {Array.from({ length: rows }, (_, layer) => {
        const slice = cells.slice(layer * cols, (layer + 1) * cols)
        const hot = slice.filter((c) => c.tier >= 1 || c.heat > 0).length
        return (
          <div key={layer} className="cluster-layer-strip">
            <code className="cluster-layer-label">L{layer}</code>
            <div
              className="cluster-layer-strip-cells"
              style={{ gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: cellGap }}
            >
              {slice.map((c, expert) => (
                <span
                  key={expert}
                  className={cn("cluster-layer-cell", c.tier >= 2 ? "vram" : c.tier >= 1 ? "ram" : c.heat > 0 ? "warm" : "cold")}
                  style={{ width: cellSize, height: cellSize, background: tierCellColor(c.tier, c.heat) }}
                  onMouseEnter={showCell(layer, expert, c)}
                  onMouseMove={showCell(layer, expert, c)}
                  onMouseLeave={() => setTip(null)}
                />
              ))}
            </div>
            <span className="cluster-layer-strip-meta">{hot}/{cols} {t("cluster.heatmap.hotShort")}</span>
          </div>
        )
      })}
      <ExpertCellTip tip={tip} rows={rows} atlas={atlas} t={t} />
    </div>
  )
}

export function ExpertHeatmapModal({ open, onClose, node, nodes, placement, baseUrl, apiKey, connected, t, modes = ["resident", "byLayer"] }: {
  open: boolean
  onClose: () => void
  node: ClusterNode
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  baseUrl: string
  apiKey: string
  connected: boolean
  t: (key: string, vars?: Record<string, string | number>) => string
  modes?: HeatmapMode[]
}) {
  const [mode, setMode] = useState<HeatmapMode>("byLayer")

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    document.addEventListener("keydown", onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      document.removeEventListener("keydown", onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  if (!open || !node.emap?.rows) return null

  return createPortal(
    <div className="cluster-modal-overlay" onClick={onClose}>
      <div
        className="cluster-modal cluster-heatmap-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="expert-heatmap-modal-title"
      >
        <header className="cluster-modal-head">
          <div>
            <h2 id="expert-heatmap-modal-title">{t("cluster.heatmap.modalTitle", { node: shortId(node.node_id) })}</h2>
            <p>{node.endpoint}</p>
          </div>
          <button type="button" className="cluster-modal-close" onClick={onClose} aria-label={t("cluster.heatmap.close")}>
            <X className="size-4" />
          </button>
        </header>
        <div className="cluster-heatmap-modal-toolbar">
          <HeatmapModeToggle mode={mode} onChange={setMode} t={t} modes={modes} />
        </div>
        <div className="cluster-heatmap-modal-body">
          {mode === "resident" ? (
            <div className="cluster-brain-large">
              <Brain
                baseUrl={baseUrl}
                apiKey={apiKey}
                connected={connected}
                staticData={{
                  rows: node.emap.rows,
                  cols: node.emap.cols,
                  map: node.emap.map,
                  hits: node.hits || "",
                  seq: node.hits_seq || 0,
                }}
              />
            </div>
          ) : (
            <ExpertHeatmapPanel mode={mode} node={node} nodes={nodes} placement={placement} large minHeight={MODAL_HEATMAP_H} t={t} />
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}

export function ClusterHeatmapSection({ nodes, placement, selectedNode, t }: {
  nodes: ClusterNode[]
  placement: ClusterPlacementResponse | null
  selectedNode?: ClusterNode | null
  t: (key: string, vars?: Record<string, string | number>) => string
}) {
  const [mode, setMode] = useState<HeatmapMode>("clusterOwner")
  const nodeIds = nodes.map((n) => n.node_id)
  const coherence = overallLayerCoherence(layerDominanceStats(placement, nodeIds))
  const clusterModes: HeatmapMode[] = ["clusterOwner", "clusterResident", "layerDominance"]
  const activeNode = clusterModes.includes(mode) ? null : (selectedNode || nodes[0] || null)

  return (
    <section className="cluster-panel cluster-heatmap-panel">
      <header className="cluster-panel-head">
        <div>
          <h3>{t("cluster.heatmap.clusterTitle")}</h3>
          <p className="cluster-heatmap-note">{t("cluster.heatmap.placementNote")}</p>
        </div>
      </header>
      <div className="cluster-heatmap-body">
        <HeatmapModeToggle mode={mode} onChange={setMode} t={t} />
        {coherence.totalLayers > 0 ? (
          <p className={cn("cluster-heatmap-coherence", coherence.multiNodeLayers > 0 && "warn")}>
            {t("cluster.heatmap.coherenceScore", {
              score: Math.round((placement?.layer_coherence?.score ?? coherence.score) * 100),
              multi: placement?.layer_coherence?.multi_node_layers ?? coherence.multiNodeLayers,
              total: placement?.layer_coherence?.total_layers ?? coherence.totalLayers,
            })}
            {(placement?.layer_coherence?.multi_node_layers ?? coherence.multiNodeLayers) > 0
              ? ` · ${t("cluster.heatmap.coherenceHint")}` : ""}
            {placement?.layer_coherent ? ` · ${t("cluster.heatmap.layerCoherentOn")}` : ""}
          </p>
        ) : null}
        <ExpertHeatmapPanel mode={mode} node={activeNode} nodes={nodes} placement={placement} large minHeight={LARGE_HEATMAP_H} t={t} />
      </div>
    </section>
  )
}
