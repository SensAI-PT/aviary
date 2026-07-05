"""Profila dove va il tempo: lettura expert dal disco vs attenzione vs moe vs matmul."""
import cProfile, pstats, io, glob, json
from engine import OlmoeStreaming

snap = glob.glob("/home/vincenzo/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/*")[0]
ref = json.load(open("ref.json"))
m = OlmoeStreaming(snap, expert_cap=16)

pr = cProfile.Profile()
pr.enable()
m.generate(ref["prompt_ids"], 8, greedy=True)   # 8 token bastano per il profilo
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(15)
print(s.getvalue())
print(f"Hit-rate: {m.cache.hitrate()*100:.1f}%  hit={m.cache.hits} miss={m.cache.miss}")
