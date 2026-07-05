"""
Motore di inferenza MoE con EXPERT-STREAMING dal disco.

Idea (quella dell'utente, resa reale):
  - la parte DENSA (embedding, attenzione, router, norme, lm_head) sta in RAM;
  - gli EXPERT stanno su disco in un file safetensors mappato in memoria (mmap)
    e vengono letti SOLO quando un token li attiva;
  - una cache LRU tiene in RAM gli expert "caldi" -> meno letture da disco.

Cosi' un modello che NON entra in RAM gira lo stesso: in RAM ci tieni solo
densa + cache, il resto vive sul disco. Validato qui su OLMoE-1B-7B.

NB: scritto per OLMoE (Llama-style con QK-norm). I punti specifici del modello
(routing, norme) sono isolati cosi' che lo stesso scheletro valga per GLM/DeepSeek.
"""
import os, json, glob, struct, time, mmap, collections
import torch
import torch.nn.functional as F

ST_DTYPE = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class Shards:
    """Indicizza i tensori in piu' file safetensors e li legge via mmap on-demand."""
    def __init__(self, snap_dir):
        self.index = {}          # name -> (shard_path, abs_offset, nbytes, torch_dtype, shape)
        self.mm = {}             # shard_path -> mmap
        for shard in sorted(glob.glob(os.path.join(snap_dir, "model-*.safetensors"))):
            with open(shard, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hlen))
            data_start = 8 + hlen
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                a, b = meta["data_offsets"]
                self.index[name] = (shard, data_start + a, b - a,
                                    ST_DTYPE[meta["dtype"]], tuple(meta["shape"]))
            fd = open(shard, "rb")
            self.mm[shard] = mmap.mmap(fd.fileno(), 0, prot=mmap.PROT_READ)

    def read(self, name):
        """Legge un tensore dal disco (mmap) e ne fa una copia RESIDENTE in RAM."""
        shard, off, nbytes, dt, shape = self.index[name]
        mv = memoryview(self.mm[shard])[off:off + nbytes]
        return torch.frombuffer(mv, dtype=dt).reshape(shape).clone()

    def has(self, name):
        return name in self.index


def quant_dequant(w, bits):
    """Quantizzazione simmetrica per-riga a `bits` bit, poi dequantizza in bf16.
    Simula numericamente cosa darebbe un expert salvato a `bits` bit sul disco."""
    if bits >= 16:
        return w
    qmax = (1 << (bits - 1)) - 1            # int8->127, int4->7, int3->3, int2->1
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp_min(1e-8)
    wq = torch.round(wf / scale).clamp(-qmax - 1, qmax)
    return (wq * scale).to(torch.bfloat16)


class ExpertCache:
    """Cache LRU degli expert. capacity = quanti expert teniamo residenti PER LAYER."""
    def __init__(self, shards, n_layers, capacity, quant_bits=16):
        self.shards = shards
        self.cap = capacity
        self.quant_bits = quant_bits
        self.caches = [collections.OrderedDict() for _ in range(n_layers)]
        self.hits = 0
        self.miss = 0

    def get(self, layer, eid):
        """Ritorna (gate_w, up_w, down_w) dell'expert, da cache o da disco."""
        c = self.caches[layer]
        if eid in c:
            self.hits += 1
            c.move_to_end(eid)
            return c[eid]
        self.miss += 1
        p = f"model.layers.{layer}.mlp.experts.{eid}."
        # tengo gli expert in bf16 (niente .float(): -24% tempo, -50% RAM, piu' fedele al riferimento)
        b = self.quant_bits
        w = (quant_dequant(self.shards.read(p + "gate_proj.weight"), b),
             quant_dequant(self.shards.read(p + "up_proj.weight"), b),
             quant_dequant(self.shards.read(p + "down_proj.weight"), b))
        c[eid] = w
        if len(c) > self.cap:
            c.popitem(last=False)
        return w

    def hitrate(self):
        t = self.hits + self.miss
        return self.hits / t if t else 0.0


def rmsnorm(x, w, eps=1e-5):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * w.float()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class OlmoeStreaming:
    def __init__(self, snap_dir, expert_cap=16, quant_bits=16):
        self.cfg = json.load(open(os.path.join(snap_dir, "config.json")))
        self.shards = Shards(snap_dir)
        c = self.cfg
        self.L = c["num_hidden_layers"]
        self.H = c["num_attention_heads"]
        self.hd = c["hidden_size"] // self.H
        self.topk = c["num_experts_per_tok"]
        self.eps = c.get("rms_norm_eps", 1e-5)
        self.norm_topk = c.get("norm_topk_prob", False)
        theta = c.get("rope_theta", 10000.0)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.cache = ExpertCache(self.shards, self.L, expert_cap, quant_bits)

        # --- parte DENSA: residente in RAM (float32) ---
        t = time.time()
        self.embed = self.shards.read("model.embed_tokens.weight").float()
        self.lm_head = self.shards.read("lm_head.weight").float()
        self.final_norm = self.shards.read("model.norm.weight").float()
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            self.layers.append({
                "in_ln":  self.shards.read(p + "input_layernorm.weight").float(),
                "post_ln":self.shards.read(p + "post_attention_layernorm.weight").float(),
                "q":  self.shards.read(p + "self_attn.q_proj.weight").float(),
                "k":  self.shards.read(p + "self_attn.k_proj.weight").float(),
                "v":  self.shards.read(p + "self_attn.v_proj.weight").float(),
                "o":  self.shards.read(p + "self_attn.o_proj.weight").float(),
                "qn": self.shards.read(p + "self_attn.q_norm.weight").float(),
                "kn": self.shards.read(p + "self_attn.k_norm.weight").float(),
                "gate": self.shards.read(p + "mlp.gate.weight").float(),
            })
        self.dense_load_s = time.time() - t

    def _rope(self, x, pos):
        # x: (heads, seq, hd) ; pos: (seq,)
        freqs = torch.outer(pos.float(), self.inv_freq)          # (seq, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)                  # (seq, hd)
        cos, sin = emb.cos(), emb.sin()
        return x * cos + rotate_half(x) * sin

    def _attn(self, lw, x, pos, layer, kv):
        """Attenzione con KV-cache. x = SOLO i token nuovi (S in prefill, 1 in decode).
        pos = posizioni assolute dei token nuovi. kv = lista per-layer dei (k,v) passati."""
        S = x.shape[0]
        q = rmsnorm(x @ lw["q"].T, lw["qn"], self.eps).view(S, self.H, self.hd).transpose(0, 1)
        k = rmsnorm(x @ lw["k"].T, lw["kn"], self.eps).view(S, self.H, self.hd).transpose(0, 1)
        v = (x @ lw["v"].T).view(S, self.H, self.hd).transpose(0, 1)
        q = self._rope(q, pos); k = self._rope(k, pos)
        if kv is not None and kv[layer] is not None:             # concateno il passato
            pk, pv = kv[layer]
            k = torch.cat([pk, k], dim=1); v = torch.cat([pv, v], dim=1)
        if kv is not None:
            kv[layer] = (k, v)
        Tk = k.shape[1]                                          # lunghezza totale (passato+nuovi)
        scores = (q @ k.transpose(-1, -2)) / (self.hd ** 0.5)    # (H,S,Tk)
        # mask causale: query a posizione assoluta pos[i] vede key j<=pos[i]
        kpos = torch.arange(Tk)
        mask = torch.where(kpos[None, :] > pos[:, None], float("-inf"), 0.0)  # -inf dove vietato
        a = F.softmax(scores + mask, dim=-1)
        out = (a @ v).transpose(0, 1).reshape(S, self.H * self.hd)
        return out @ lw["o"].T

    def _moe(self, layer, lw, x):
        S = x.shape[0]
        logits = x @ lw["gate"].T                                # (S,64)
        probs = F.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, self.topk, dim=-1)            # (S,topk)
        if self.norm_topk:
            w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        # raggruppo per expert: per ogni expert davvero usato, processo i suoi token
        for eid in torch.unique(idx).tolist():
            sel = (idx == eid)                                   # (S,topk) bool
            rows = sel.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            gw, uw, dw = self.cache.get(layer, eid)              # <-- streaming dal disco (bf16)
            xe = x[rows].to(torch.bfloat16)                      # calcolo expert in bf16
            h = (F.silu(xe @ gw.T) * (xe @ uw.T)) @ dw.T
            weight = (w[rows] * sel[rows].float()).sum(-1, keepdim=True)
            out[rows] += weight * h.float()
        return out

    @torch.no_grad()
    def _step(self, ids_new, pos, kv):
        """Un passo del modello sui token nuovi. Ritorna logit dell'ultimo token."""
        x = self.embed[torch.tensor(ids_new)]                   # (S,hidden)
        for i, lw in enumerate(self.layers):
            x = x + self._attn(lw, rmsnorm(x, lw["in_ln"], self.eps), pos, i, kv)
            x = x + self._moe(i, lw, rmsnorm(x, lw["post_ln"], self.eps))
        x = rmsnorm(x, self.final_norm, self.eps)
        return (x @ self.lm_head.T)[-1]

    @torch.no_grad()
    def generate(self, token_ids, n_new, greedy=True):
        kv = [None] * self.L
        ids = list(token_ids)
        # PREFILL: tutti i token del prompt in un colpo, riempie la kv-cache
        logit = self._step(ids, torch.arange(len(ids)), kv)
        for s in range(n_new):
            nxt = int(torch.argmax(logit)) if greedy else int(torch.multinomial(F.softmax(logit, -1), 1))
            ids.append(nxt)
            if s == n_new - 1:
                break
            # DECODE: un solo token nuovo, usa la kv-cache (qui la cache expert torna a funzionare)
            logit = self._step([nxt], torch.tensor([len(ids) - 1]), kv)
        return ids
