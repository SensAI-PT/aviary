"""Convert Qwen3 MoE HuggingFace checkpoint -> Colibri int4 safetensors container.

Usage:
  python3 tools/convert_qwen3_moe.py --indir qwen3_moe_tiny --outdir qwen3_moe_tiny_i4 --ebits 4
  python3 tools/convert_qwen3_moe.py --repo Qwen/Qwen3-30B-A3B-Instruct-2507 --outdir /path/i4
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import quant_int2, quant_int4, quant_int8, quant_int4_grouped, free_gb, layer_idx


def _quantize(w, bits, group_size):
    if group_size > 0 and bits <= 4:
        return quant_int4_grouped(w, bits, group_size)
    return (quant_int2(w, bits) if bits <= 2 else
            quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))


def dequant(f, name, keys=None):
    import torch
    if keys is None:
        keys = set(f.keys())
    dt = f.get_slice(name).get_dtype()
    if dt not in ("F8_E4M3", "float8_e4m3fn"):
        return f.get_tensor(name).to(torch.float32).numpy()
    w = f.get_tensor(name).to(torch.float32)
    if (name + "_scale_inv") in keys:
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)
        if sc.ndim == 0:
            return (w * sc).numpy()
        o, i = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
        return (w * sc).numpy()
    if (name + "_scale") in keys:
        return (w * f.get_tensor(name + "_scale").to(torch.float32)).numpy()
    raise KeyError(f"FP8 tensor {name} missing weight_scale or weight_scale_inv")


def classify(name, n_layers):
    if name.endswith("_scale_inv") or name.endswith("_scale"):
        return "consumed"
    li = layer_idx(name)
    if li >= n_layers:
        return "skip"
    if name.endswith("e_score_correction_bias") or name.endswith("expert_bias"):
        return "skip"
    if name.endswith("mlp.gate.weight"):
        return "f32"
    if name.endswith("norm.weight") or name == "model.norm.weight":
        return "f32"
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if ".mlp.experts." in name and name.endswith(".weight"):
        return "x"
    if name.endswith(".weight"):
        return "q"
    return "f32"


def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits, group_size=64):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            kind = classify(name, n_layers)
            if kind in ("skip", "consumed"):
                continue
            w = dequant(f, name)
            if kind == "f32":
                out_dict[name] = w.astype(np.float32)
            else:
                bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                if w.ndim != 2:
                    out_dict[name] = w.astype(np.float32)
                    continue
                q, s = _quantize(w, bits, group_size)
                out_dict[name] = q
                out_dict[name + ".qs"] = s


def convert_local(indir, outdir, n_layers, ebits, io_bits, xbits, group_size=64):
    from safetensors.numpy import save_file
    shards = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    os.makedirs(outdir, exist_ok=True)
    for i, sp in enumerate(shards):
        out = {}
        convert_shard(sp, out, n_layers, ebits, io_bits, xbits, group_size)
        save_file(out, os.path.join(outdir, f"out-{i:05d}.safetensors"))
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja", "special_tokens_map.json"]:
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    print(f"converted {len(shards)} shards -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Qwen3 MoE -> Colibri int4 container")
    ap.add_argument("--repo", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=4)
    ap.add_argument("--io-bits", type=int, default=8)
    ap.add_argument("--xbits", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=48)
    ap.add_argument("--group-size", type=int, default=64,
                    help="int4 scale group size (64 default; 0 = per-row scales)")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    a = ap.parse_args()
    if a.xbits is None:
        a.xbits = a.ebits

    if a.indir:
        if not a.outdir:
            sys.exit("--outdir required with --indir")
        cfg_path = os.path.join(a.indir, "config.json")
        if os.path.isfile(cfg_path):
            a.n_layers = json.load(open(cfg_path)).get("num_hidden_layers", a.n_layers)
        convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, a.xbits, a.group_size)
        return

    if not a.outdir:
        sys.exit("--outdir required")

    import convert_fp8_to_int4 as cvt
    cvt.classify = lambda name, n_layers, keep_mtp=False, keep_idx=False: classify(name, n_layers)
    cvt.dequant = dequant
    sys.argv = [
        "convert_qwen3_moe.py",
        "--repo", a.repo,
        "--outdir", a.outdir,
        "--ebits", str(a.ebits),
        "--io-bits", str(a.io_bits),
        "--xbits", str(a.xbits),
        "--n-layers", str(a.n_layers),
        "--group-size", str(a.group_size),
        "--min-free-gb", str(a.min_free_gb),
    ]
    cvt.main()


if __name__ == "__main__":
    main()
