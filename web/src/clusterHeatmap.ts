import type { ClusterNode, ClusterPlacementResponse } from "@/lib/api"

export type HeatmapMode = "resident" | "byLayer" | "clusterOwner" | "clusterResident" | "layerDominance"

export const TIER_RGB: [number, number, number][] = [[58, 71, 80], [90, 155, 216], [78, 214, 165]]
export const NODE_PALETTE = ["#4ed6a5", "#5a9bd8", "#e6a04a", "#c678dd", "#ff6b6b", "#56b6c2", "#d19a66", "#98c379"]

export function decodeEmap(map: string, rows: number, cols: number) {
  const cells: { tier: number; heat: number }[] = []
  for (let i = 0; i < rows * cols; i++) {
    const byte = parseInt(map.substr(i * 2, 2), 16) || 0
    cells.push({ tier: byte >> 6, heat: byte & 63 })
  }
  return cells
}

export function nodeColor(nodeId: string, nodeIds: string[]) {
  const i = Math.max(0, nodeIds.indexOf(nodeId))
  return NODE_PALETTE[i % NODE_PALETTE.length]
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "")
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

export function maxEmapDims(nodes: ClusterNode[]) {
  return nodes.reduce(
    (acc, n) => ({
      rows: Math.max(acc.rows, n.emap?.rows || 0),
      cols: Math.max(acc.cols, n.emap?.cols || 0),
    }),
    { rows: 0, cols: 0 },
  )
}

/** Best resident tier/heat for each cell across all executors. */
export function clusterResidentGrid(nodes: ClusterNode[]) {
  const { rows, cols } = maxEmapDims(nodes)
  if (!rows || !cols) return null
  const out = Array.from({ length: rows * cols }, () => ({ tier: 0, heat: 0 }))
  for (const node of nodes) {
    if (!node.emap?.map) continue
    const cells = decodeEmap(node.emap.map, node.emap.rows, node.emap.cols)
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i]
      if (c.tier > out[i].tier || (c.tier === out[i].tier && c.heat > out[i].heat)) out[i] = c
    }
  }
  return { rows, cols, cells: out }
}

/** Scheduler owner per layer:expert (unassigned = ""). */
export function clusterOwnerGrid(placement: ClusterPlacementResponse | null, rows: number, cols: number) {
  const owners = Array.from({ length: rows * cols }, () => "")
  if (!placement?.experts) return owners
  for (const [key, nid] of Object.entries(placement.experts)) {
    const [ls, es] = key.split(":")
    const layer = parseInt(ls, 10)
    const expert = parseInt(es, 10)
    if (Number.isNaN(layer) || Number.isNaN(expert) || layer >= rows || expert >= cols) continue
    owners[layer * cols + expert] = nid
  }
  return owners
}

export type LayerDominanceRow = {
  layer: number
  dominant: string
  share: number
  assigned: number
  nodeCounts: Record<string, number>
}

/** How coherently hot experts within each layer sit on one executor (placement plan). */
export function layerDominanceStats(placement: ClusterPlacementResponse | null, nodeIds: string[]) {
  if (!placement?.experts) return []
  const byLayer: Record<number, Record<string, number>> = {}
  for (const [key, nid] of Object.entries(placement.experts)) {
    const layer = parseInt(key.split(":")[0], 10)
    if (Number.isNaN(layer)) continue
    byLayer[layer] = byLayer[layer] || {}
    byLayer[layer][nid] = (byLayer[layer][nid] || 0) + 1
  }
  return Object.entries(byLayer)
    .map(([ls, counts]) => {
      const layer = parseInt(ls, 10)
      const assigned = Object.values(counts).reduce((a, b) => a + b, 0)
      const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || ""
      const share = assigned ? (counts[dominant] || 0) / assigned : 0
      return { layer, dominant, share, assigned, nodeCounts: counts }
    })
    .sort((a, b) => a.layer - b.layer)
}

export function overallLayerCoherence(stats: LayerDominanceRow[]) {
  if (!stats.length) return { score: 0, multiNodeLayers: 0, totalLayers: 0 }
  const weighted = stats.reduce((s, r) => s + r.share * r.assigned, 0)
  const total = stats.reduce((s, r) => s + r.assigned, 0)
  const multiNodeLayers = stats.filter((r) => Object.keys(r.nodeCounts).length > 1).length
  return { score: total ? weighted / total : 0, multiNodeLayers, totalLayers: stats.length }
}
