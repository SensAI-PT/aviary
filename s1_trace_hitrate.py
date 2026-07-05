"""
Pilastro 2 - La domanda che decide tutto:
quanto e' alto l'hit-rate di una cache di expert su un router MoE VERO?

Facciamo girare OLMoE-1B-7B su testo reale, registriamo per ogni (layer, posizione)
quali 8 expert su 64 vengono attivati, e poi SIMULIAMO diverse politiche di cache
al variare della capacita' K (quanti expert/layer teniamo residenti in RAM).

Output: hit-rate per policy e per K  ->  mappato su token/sec col cost model.
"""
import sys, time, collections
import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
TOPK = 8
N_EXPERTS = 64
N_LAYERS = 16
EXPERT_MB = 12.6   # bf16

# Banda misurata allo Stadio 0 (random read, ext4). Aggiorna se rifai il bench.
DISK_BW_GBS = 7.33

PROMPTS = [
    "The history of the Roman Empire spans over a thousand years, from the founding "
    "of the city to the fall of Constantinople. Its legacy in law, language and "
    "engineering still shapes the modern world.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n"
    "    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "In quantum mechanics, the wave function encodes the probability amplitude of a "
    "particle's state. Measurement collapses this superposition into a definite outcome, "
    "a process that remains philosophically contested.",
    "La politica monetaria della banca centrale influenza i tassi di interesse, "
    "l'inflazione e l'occupazione. Alzare i tassi raffredda la domanda ma rischia "
    "di rallentare la crescita economica.",
]


def collect_trace():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Carico il modello (bf16, CPU)...", flush=True)
    t = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"  caricato in {time.time()-t:.0f}s", flush=True)

    # trace[layer] = lista (in ordine di token) di tuple di 8 expert id
    trace = [[] for _ in range(N_LAYERS)]
    for pi, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(ids, output_router_logits=True, use_cache=False)
        # out.router_logits: tupla di N_LAYERS tensori (n_token, n_experts)
        for li, rl in enumerate(out.router_logits):
            topk = rl.topk(TOPK, dim=-1).indices  # (n_token, 8)
            for row in topk.tolist():
                trace[li].append(tuple(sorted(row)))
        print(f"  prompt {pi}: {ids.shape[1]} token", flush=True)
    return trace


def simulate(trace, K, policy="lru"):
    """Cache per-layer di capacita' K. Ritorna hit-rate globale sugli accessi a expert."""
    hits = total = 0
    for li in range(N_LAYERS):
        if policy == "lru":
            cache = collections.OrderedDict()
        elif policy == "lfu":
            cache = {}            # eid -> freq
            freq = collections.Counter()
        for experts in trace[li]:
            for e in experts:
                total += 1
                if policy == "lru":
                    if e in cache:
                        hits += 1
                        cache.move_to_end(e)
                    else:
                        cache[e] = 1
                        if len(cache) > K:
                            cache.popitem(last=False)
                elif policy == "lfu":
                    freq[e] += 1
                    if e in cache:
                        hits += 1
                    else:
                        if len(cache) >= K:
                            victim = min(cache, key=lambda x: freq[x])
                            del cache[victim]
                        cache[e] = 1
    return hits / total if total else 0.0


def consecutive_reuse(trace):
    """Frazione di expert al token t gia' attivi al token t-1 (stesso layer)."""
    same = tot = 0
    for li in range(N_LAYERS):
        seq = trace[li]
        for t in range(1, len(seq)):
            prev = set(seq[t-1]); cur = set(seq[t])
            same += len(prev & cur); tot += len(cur)
    return same / tot if tot else 0.0


def tok_per_sec(hitrate):
    bytes_cold_gb = N_LAYERS * TOPK * EXPERT_MB / 1024
    eff = bytes_cold_gb * (1 - hitrate)
    return DISK_BW_GBS / eff if eff > 0 else float("inf")


if __name__ == "__main__":
    trace = collect_trace()
    ntok = sum(len(trace[0]) for _ in [0])
    print(f"\nToken totali tracciati: {len(trace[0])}  x {N_LAYERS} layer")
    print(f"Riuso consecutivo (expert in comune t vs t-1): {consecutive_reuse(trace)*100:.1f}%")

    print("\nHit-rate cache per-layer al variare di K (expert residenti su 64):")
    print(f"{'K':>4} {'RAM/GB':>7} {'LRU':>8} {'LFU':>8} {'tok/s@LRU':>10}")
    for K in (8, 12, 16, 24, 32, 48):
        ram = K * N_LAYERS * EXPERT_MB / 1024
        hl = simulate(trace, K, "lru")
        hf = simulate(trace, K, "lfu")
        print(f"{K:>4} {ram:>6.1f}G {hl*100:>7.1f}% {hf*100:>7.1f}% {tok_per_sec(hl):>9.1f}")
    print("\nNota: K=8 e' il minimo teorico (8 attivi/token). K=64 = tutto in RAM (no streaming).")
