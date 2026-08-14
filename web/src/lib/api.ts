export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
}

interface OpenAIError {
  error?: { message?: string }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

export interface HealthResponse {
  status: string
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
  cluster?: boolean
}

export interface ClusterNode {
  node_id: string
  endpoint: string
  host: string
  http_port: number
  model_id: string
  model_path: string
  arch?: string
  engine_id?: number
  status: string
  inflight: number
  uptime_sec: number
  last_heartbeat_age_sec: number
  hwinfo?: HwinfoHealth
  tiers?: TiersHealth
  emap?: { rows: number; cols: number; map: string }
  hits?: string
  hits_seq?: number
  usage?: Array<{ layer: number; expert: number; count: number }>
  costs?: Array<{ layer: number; expert: number; tier: number; load_us: number; exec_us: number }>
  profile?: ProfileTurn[]
  expert_port?: number
}

export interface ClusterNodesResponse {
  nodes: ClusterNode[]
  healthy: number
  total: number
  cohort?: { arch: string; model_id: string; engine_id: number | null }
  merged_usage?: Array<{ layer: number; expert: number; count: number }>
}

export interface ClusterJobTrace {
  ts: number
  job_id?: string
  kind: string
  node_id: string
  layer: number
  expert: number
  local?: boolean
  rpc_us?: number
}

export interface ClusterJob {
  job_id: string
  node_id: string
  node_endpoint: string
  path: string
  status: "running" | "completed" | "failed"
  started_at: number
  ended_at: number | null
  duration_sec: number
  http_status: number | null
  error: string | null
  trace?: ClusterJobTrace[]
}

export interface ClusterJobsResponse {
  active: ClusterJob[]
  history: ClusterJob[]
  active_count: number
  completed_count: number
  failed_count: number
}

export interface ClusterLayerBlock {
  start: number
  end: number
  max_tier?: number
}

export interface ClusterNodeTierPref {
  max_tier: number
  vram_slow?: boolean
  median_exec?: Record<string, number>
}

export interface ClusterRpcHistogram {
  count: number
  buckets: Array<{ max_us: number; count: number }>
  over_max?: number
  p50_us: number
  p95_us: number
  window_sec: number
}

export interface ClusterLayerCoherence {
  score: number
  multi_node_layers: number
  total_layers: number
  layers?: Array<{ layer: number; dominant: string; share: number; assigned: number; nodes: number }>
}

export interface ClusterPlacementResponse {
  experts: Record<string, string>
  expert_tiers: Record<string, number>
  rpc_matrix_us: Record<string, Record<string, number>>
  computed_at: number
  usage_top: Array<{ layer: number; expert: number; count: number }>
  blocks?: Record<string, ClusterLayerBlock[]>
  planned_blocks?: ClusterLayerBlock[]
  layer_caps?: Record<string, number>
  node_tier_prefs?: Record<string, ClusterNodeTierPref>
  layer_blocks?: boolean
  layer_coherence?: ClusterLayerCoherence
  layer_coherent?: boolean
  reassignments?: number
  rpc_histogram?: ClusterRpcHistogram
}

export interface ClusterOverviewResponse {
  nodes: ClusterNodesResponse
  placement: ClusterPlacementResponse
  jobs: ClusterJobsResponse
  updated_at: number
}

export interface ClusterBenchHopMix {
  local: number
  remote: number
  fallback: number
}

export interface ClusterBenchScoreboard {
  p50_cold_sec?: number | null
  p50_warm_sec?: number | null
  p95_cold_sec?: number | null
  p95_warm_sec?: number | null
  epa?: number | null
  rps_at_w?: number | null
  hop_mix_pct?: ClusterBenchHopMix
  primary_node_mix_pct?: Record<string, number>
}

export interface ClusterBenchStepResult {
  step: string
  wipe_usage: boolean
  workers: number
  requests: number
  ok: number
  errors: number
  elapsed_sec: number
  latency?: { p50_sec?: number; p95_sec?: number }
  rps?: number
  hop_mix_pct?: ClusterBenchHopMix
}

export interface ClusterBenchResult {
  finished_at: string
  meta: {
    preset: string
    wipe_usage?: boolean
    local_only: boolean
    requests: number
    workers: number
    max_tokens: number
    prompt: string
  }
  steps: ClusterBenchStepResult[]
  scoreboard: ClusterBenchScoreboard
  last_error?: string
  markdown?: string
}

export interface ClusterBenchProgress {
  preset: string
  step: string
  chat_index: number
  chat_total: number
  last_error: string
}

export interface ClusterBenchResponse {
  status: "idle" | "running" | "failed"
  progress: ClusterBenchProgress
  last_result: (ClusterBenchResult & { markdown?: string }) | null
}

export type ClusterBenchStartRequest = {
  preset: "cold_sequential" | "warm_sequential" | "concurrent" | "suite"
  wipe_usage?: boolean
  local_only?: boolean
  requests?: number
  workers?: number
  max_tokens?: number
  prompt?: string
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface ProfileResponse {
  seq: number
  turns: ProfileTurn[]
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string }> }
  return (body.data || []).map((model) => model.id)
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export async function getProfile(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "profile"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ProfileResponse
}

export async function getClusterNodes(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ClusterNodesResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "cluster/nodes"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ClusterNodesResponse
}

export async function getClusterPlacement(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ClusterPlacementResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "cluster/placement"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ClusterPlacementResponse
}

export async function getClusterJobs(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ClusterJobsResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "cluster/jobs"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ClusterJobsResponse
}

export async function getClusterOverview(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ClusterOverviewResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "cluster/overview"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ClusterOverviewResponse
}

export async function getClusterBench(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ClusterBenchResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "cluster/bench"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ClusterBenchResponse
}

export async function startClusterBench(baseUrl: string, apiKey: string, req: ClusterBenchStartRequest, signal?: AbortSignal): Promise<ClusterBenchResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "cluster/bench"), {
    method: "POST",
    headers: headers(apiKey),
    signal,
    body: JSON.stringify(req),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ClusterBenchResponse
}

export function extractSSE(buffer: string) {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const data = frames.flatMap((frame) =>
    frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()),
  )
  return { data, rest }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  cacheSlot?: number
  signal: AbortSignal
  onDelta: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      messages: options.messages.map(({ role, content }) => ({ role, content })),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
    }
    const choice = event.choices?.[0]
    const text = choice?.delta?.content
    if (text) options.onDelta(text)
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) usage = event.usage
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }

  const queueWaitHeader = response.headers.get("x-colibri-queue-wait-ms")
  const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
  return {
    finishReason,
    usage,
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
}
