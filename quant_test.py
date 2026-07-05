"""La domanda che conta: a quanti bit l'output degli expert REGGE ancora?
Quantizzo solo gli expert (la parte densa resta bf16) e confronto col riferimento."""
import json, glob
from engine import OlmoeStreaming

snap = glob.glob("/home/vincenzo/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/*")[0]
ref = json.load(open("ref.json"))
exp = ref["full_ids"][len(ref["prompt_ids"]):]
n_new = len(exp)

print(f"{'bit':>4} {'MB/expert':>10} {'match':>7}  testo")
for bits in (16, 8, 4, 3, 2):
    m = OlmoeStreaming(snap, expert_cap=64, quant_bits=bits)  # cap64: isola l'effetto quant
    out = m.generate(ref["prompt_ids"], n_new, greedy=True)
    gen = out[len(ref["prompt_ids"]):]
    match = sum(a == b for a, b in zip(gen, exp))
    mb = 6.29 * bits / 8 / 1.0  # ~6.29M param/expert * bit / 8 -> MB
    # decode testo per vedere se e' ancora sensato
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    txt = tok.decode(gen)
    print(f"{bits:>4} {mb:>9.1f}M {match:>4}/{n_new:<2}  {txt!r}")
