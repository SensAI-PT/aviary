"""
RICERCA - Il cardine del metodo: lo SKEW degli expert e' sfruttabile?

Se pochi expert "caldi" coprono gran parte delle attivazioni, allora la strategia
giusta per un modello che NON entra in RAM e':
   - PIN dei caldi (residenti per sempre in RAM, profilati offline)
   - STREAM dei freddi dal disco
invece di una LRU dinamica (che su RAM piccola va in pressione, l'abbiamo visto).

Test onesto: determino il "set caldo" dalla PRIMA meta' dei token, e misuro la
copertura sulla SECONDA meta' (mai vista). Confronto PIN-caldi statico vs LRU a parita' di K.
"""
import json, glob, collections, time
import torch

MODEL = "allenai/OLMoE-1B-7B-0924"
N_EXP, TOPK, N_LAYERS = 64, 8, 16

# testo piu' lungo e vario per statistiche decenti
PROMPTS = [
    "The Roman Empire was one of the largest empires in history. At its height under "
    "Trajan, it covered five million square kilometres and held seventy million people, "
    "about a fifth of the world's population at the time. The empire's longevity and vast "
    "extent ensured a lasting influence on language, religion, architecture, philosophy, law "
    "and forms of government across the territory it once governed. ",
    "Photosynthesis is a biological process used by plants, algae and some bacteria to "
    "convert light energy into chemical energy stored in glucose. It occurs in the chloroplasts, "
    "specifically using the green pigment chlorophyll. The process consumes carbon dioxide and "
    "water and releases oxygen as a by-product, sustaining most life on Earth. ",
    "def fibonacci(n):\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        "
    "result.append(a)\n        a, b = b, a + b\n    return result\n\n"
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n"
    "        self.items.append(x)\n    def pop(self):\n        return self.items.pop()\n",
    "L'economia mondiale nel ventunesimo secolo e' caratterizzata da una crescente "
    "globalizzazione, dall'integrazione dei mercati finanziari e dalla rapida diffusione "
    "delle tecnologie digitali. Le banche centrali giocano un ruolo cruciale nel mantenere "
    "la stabilita' dei prezzi attraverso la politica monetaria. ",
]


def collect():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("carico modello...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).eval()
    trace = [[] for _ in range(N_LAYERS)]
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(ids, output_router_logits=True, use_cache=False)
        for li, rl in enumerate(out.router_logits):
            for row in rl.topk(TOPK, -1).indices.tolist():
                trace[li].append(tuple(row))
        print(f"  +{ids.shape[1]} token", flush=True)
    return trace


def lru_hit(seq, K):
    c = collections.OrderedDict(); hit = tot = 0
    for experts in seq:
        for e in experts:
            tot += 1
            if e in c: hit += 1; c.move_to_end(e)
            else:
                c[e] = 1
                if len(c) > K: c.popitem(last=False)
    return hit / tot


def static_hot_hit(train, test, K):
    """Set caldo = K piu' frequenti nel train; copertura misurata sul test."""
    freq = collections.Counter(e for experts in train for e in experts)
    hot = set(e for e, _ in freq.most_common(K))
    hit = tot = 0
    for experts in test:
        for e in experts:
            tot += 1
            if e in hot: hit += 1
    return hit / tot


if __name__ == "__main__":
    trace = collect()
    ntok = len(trace[0])
    print(f"\nToken totali: {ntok} x {N_LAYERS} layer = {ntok*N_LAYERS*TOPK} accessi expert\n")

    # skew: distribuzione di frequenza (media sui layer), e curva di copertura top-K
    print("COPERTURA del set caldo (statico, profilato su prima meta', testato su seconda):")
    print(f"{'K':>4} {'RAM':>7} {'pin-caldo':>10} {'LRU':>8}  (uniforme=K/64)")
    for K in (8, 12, 16, 24, 32):
        cov_static, cov_lru = [], []
        for li in range(N_LAYERS):
            seq = trace[li]; h = len(seq) // 2
            cov_static.append(static_hot_hit(seq[:h], seq[h:], K))
            cov_lru.append(lru_hit(seq, K))
        cs = sum(cov_static)/N_LAYERS; cl = sum(cov_lru)/N_LAYERS
        ram = K * N_LAYERS * 12.6 / 1024
        print(f"{K:>4} {ram:>5.1f}GB {cs*100:>9.1f}% {cl*100:>7.1f}%   {K/64*100:>4.0f}%")

    # quanto e' skewata la distribuzione? entropia normalizzata e top-8 share
    import math
    shares = []
    for li in range(N_LAYERS):
        freq = collections.Counter(e for ex in trace[li] for e in ex)
        tot = sum(freq.values())
        top8 = sum(c for _, c in freq.most_common(8)) / tot
        shares.append(top8)
    print(f"\nSkew: gli 8 expert piu' caldi (su 64) coprono in media "
          f"{sum(shares)/len(shares)*100:.1f}% delle attivazioni (uniforme sarebbe 12.5%).")
