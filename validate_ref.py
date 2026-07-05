"""Genera il riferimento con transformers (greedy) e lo salva. Va lanciato da solo (usa ~13GB)."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"
N = 12
tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
ids = tok(PROMPT, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924",
                                             torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N, do_sample=False)
full = out[0].tolist()
json.dump({"prompt": PROMPT, "prompt_ids": ids[0].tolist(), "full_ids": full,
           "text": tok.decode(full)}, open("ref.json", "w"))
print("RIFERIMENTO salvato:", repr(tok.decode(full)))
print("full_ids:", full)
