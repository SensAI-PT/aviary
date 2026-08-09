# Qwen3 MoE on Colibri / Aviary

`c/qwen3_moe.c` runs [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
(30.5B total / ~3.3B active) with Colibri's expert-streaming approach. Auto-detected from
`config.json` (`model_type: qwen3_moe`); same `coli chat` / `coli serve` / `coli agent` front end.

## Build

```bash
cd c
make qwen3_moe
./setup.sh   # builds colibri + hy3 + qwen3_moe and runs tiny self-test when fixtures exist
```

## Convert weights to int4

```bash
cd c
./coli convert --repo Qwen/Qwen3-30B-A3B-Instruct-2507 --model /path/to/qwen3_i4
# or from a local BF16/FP8 checkout:
python3 tools/convert_qwen3_moe.py --indir /path/to/model --outdir /path/to/qwen3_i4
```

Publish the converted container (e.g. `Qwen3-30B-A3B-Instruct-2507-colibri-int4`) for cluster nodes.

## Single-node quick start

```bash
COLI_MODEL=/path/to/qwen3_i4 ./coli chat --ram 16
COLI_MODEL=/path/to/qwen3_i4 ./coli plan --ram 16
COLI_MODEL=/path/to/qwen3_i4 ./coli doctor --ram 16
```

## Aviary cluster testing

Each agent needs the same int4 weights on fast local storage:

```bash
# master
./coli master --host 0.0.0.0 --port 9000

# agent(s)
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://MASTER:9000 --host 0.0.0.0 --port 8001
```

See [AVIARY.md](AVIARY.md) for the full cluster protocol.

## Tiny oracle (CI / local smoke, no 30B download)

```bash
cd c
python3 tools/make_qwen3_oracle.py
SNAP=./qwen3_moe_tiny TF=1 ./qwen3_moe 64 16 16    # expect 32/32
python3 tools/convert_qwen3_moe.py --indir qwen3_moe_tiny --outdir qwen3_moe_tiny_i4 --ebits 4
SNAP=./qwen3_moe_tiny_i4 TF=1 ./qwen3_moe 64 4 8   # int4 tolerance (~24/32)
```

## Memory guidance

| Profile | RAM | Notes |
|---|---|---|
| Lean test | 16 GB | Dense stack + small expert cache; cold start reads from disk |
| Comfortable | 32 GB | Larger expert LRU, fewer disk round-trips |

Only ~3.3B parameters are active per token; the full 30B expert pool stays on disk and streams on demand.

## Architecture notes

- GQA attention with per-head Q/K RMSNorm + rotate-half RoPE (same pattern as Hy3)
- MoE router: softmax + top-k + `norm_topk_prob` (Qwen3Moe style, not Hy3 sigmoid router)
- Config fields: `num_experts` or `num_local_experts`, `moe_intermediate_size`, `head_dim`
- Engine argv: `./qwen3_moe <cap> 4 8` (cap, expert ebits=4, dense dbits=8)
