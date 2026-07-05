"""Lancia il motore streaming, confronta con il riferimento, misura RAM/hit-rate/velocita'."""
import json, time, glob, sys, resource
from engine import OlmoeStreaming

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 16   # expert residenti per layer (su 64)
snap = glob.glob("/home/vincenzo/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/*")[0]
ref = json.load(open("run_ref.json" if False else "ref.json"))

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # KB->GB su Linux

print(f"== Motore streaming, cache = {CAP} expert/layer su 64 ==")
t = time.time()
m = OlmoeStreaming(snap, expert_cap=CAP)
print(f"densa caricata in {m.dense_load_s:.1f}s | RSS dopo load densa: {rss_gb():.2f} GB")

n_new = len(ref["full_ids"]) - len(ref["prompt_ids"])
t = time.time()
out = m.generate(ref["prompt_ids"], n_new, greedy=True)
dt = time.time() - t

# confronto
gen = out[len(ref["prompt_ids"]):]
exp = ref["full_ids"][len(ref["prompt_ids"]):]
match = sum(a == b for a, b in zip(gen, exp))
print(f"\nRiferimento (transformers): {exp}")
print(f"Motore streaming          : {gen}")
print(f"Token coincidenti: {match}/{len(exp)}")
print(f"\nRSS PICCO: {rss_gb():.2f} GB  (modello completo in bf16 = ~13 GB)")
print(f"Hit-rate cache expert: {m.cache.hitrate()*100:.1f}%  (hit={m.cache.hits} miss={m.cache.miss})")
print(f"Velocita': {n_new/dt:.2f} tok/s ({dt:.1f}s per {n_new} token, no kv-cache)")
