#!/usr/bin/env python3
"""Tiny random Hy3 (hy_v3) oracle for Colibri hy3.c validation."""
import json
import torch

try:
    from transformers import HYV3Config, HYV3ForCausalLM
except ImportError as e:
    raise SystemExit("pip install 'transformers>=5.6.0' torch safetensors") from e


def gpt2_bytes_to_unicode():
    """Same mapping as tok.h tk_build_bytemap / GPT-2 ByteLevel."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def write_byte_tokenizer(path, vocab_size=256):
    """Minimal tokenizer.json so coli agent / hy3 SERVE can load the tiny snap.

    Vocab is identity byte→id (0..255) with GPT-2 ByteLevel spellings.
    Empty merges → tok.h rankbpe mode. Enough for cluster smoke tests; not a
    real Hy3 chat tokenizer.
    """
    if vocab_size != 256:
        raise ValueError("hy3_tiny vocab_size must stay 256 to match this stub")
    byte_enc = gpt2_bytes_to_unicode()
    vocab = {byte_enc[i]: i for i in range(256)}
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, separators=(",", ":"))

torch.manual_seed(5678)

cfg = HYV3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,
    moe_intermediate_size=32,
    num_hidden_layers=5,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_experts=8,
    num_experts_per_tok=2,
    num_shared_experts=1,
    route_norm=True,
    router_scaling_factor=2.826,
    moe_router_use_sigmoid=True,
    moe_router_enable_expert_bias=True,
    qk_norm=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = HYV3ForCausalLM(cfg).eval()
with torch.no_grad():
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp.gate, "e_score_correction_bias"):
            mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.1, 0.1, cfg.num_experts))

print("=== Hy3 tiny state_dict ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

out_dir = "hy3_tiny"
model.save_pretrained(out_dir, safe_serialization=True)
json.dump(cfg.to_dict(), open(f"{out_dir}/config.json", "w"))
write_byte_tokenizer(f"{out_dir}/tokenizer.json", vocab_size=cfg.vocab_size)
json.dump(
    {"model_max_length": 4096},
    open(f"{out_dir}/tokenizer_config.json", "w"),
)
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_hy3.json", "w"))
print(f"\nsaved: {out_dir}/ (incl. tokenizer.json) and ref_hy3.json")
