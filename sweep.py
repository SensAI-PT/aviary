"""Sweep della cache: per ogni capacita' misura correttezza, hit-rate, RAM cache, velocita'."""
import json, time, glob
from engine import OlmoeStreaming

snap = glob.glob("/home/vincenzo/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/*")[0]
ref = json.load(open("ref.json"))
exp = ref["full_ids"][len(ref["prompt_ids"]):]
n_new = len(exp)
EXPERT_MB_BF16 = 12.6

print(f"{'cap':>4} {'RAMcache':>9} {'match':>6} {'hit%':>6} {'tok/s':>7} {'sec':>6}")
for cap in (16, 32, 48, 64):
    m = OlmoeStreaming(snap, expert_cap=cap)
    t = time.time()
    out = m.generate(ref["prompt_ids"], n_new, greedy=True)
    dt = time.time() - t
    gen = out[len(ref["prompt_ids"]):]
    match = sum(a == b for a, b in zip(gen, exp))
    ram = cap * m.L * EXPERT_MB_BF16 / 1024
    print(f"{cap:>4} {ram:>7.1f}GB {match:>3}/{n_new:<2} {m.cache.hitrate()*100:>5.1f}% {n_new/dt:>7.2f} {dt:>6.1f}")
