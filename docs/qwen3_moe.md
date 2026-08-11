# Qwen3 MoE on Colibri / Aviary

`c/qwen3_moe.c` runs [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
(30.5B total / ~3.3B active) with Colibri's expert-streaming approach. Auto-detected from
`config.json` (`model_type: qwen3_moe`); same `coli chat` / `coli serve` / `coli agent` front end.

## Pre-converted weights (Hugging Face)

**https://huggingface.co/UnderstandLing/Qwen3_30B_A3B_i4**

Colibri/Aviary int4 container (grouped gs=64). **Not** GGUF / AWQ / vLLM — only loads in this engine.

```bash
# once per machine (needs huggingface_hub / hf CLI)
pip install -U "huggingface_hub[cli]"
hf download UnderstandLing/Qwen3_30B_A3B_i4 --local-dir /path/to/qwen3_i4
```

Put the directory on fast local storage (NVMe/ext4), not a network mount or `/mnt/c`.
Every Aviary agent needs its own copy (or the same path on each node).

## Build

```bash
cd c
make qwen3_moe
make qwen3_moe METAL=1   # Apple Silicon: opt-in Metal MoE (+ dense GEMM; attention CPU)
./setup.sh   # builds colibri + hy3 + qwen3_moe (METAL=1 on Darwin) and runs tiny self-test when fixtures exist
```

Metal usage (see [metal.md](metal.md)):

```bash
COLI_METAL=1 COLI_NO_OMP_TUNE=1 DIRECT=1 PIPE=1 \
  COLI_MODEL=/path/to/qwen3_i4 ./coli chat --ram 16
```

The published Qwen int4 container is grouped (`fmt=4`, gs=64); Metal MoE handles
that format. Dense GEMM also runs on Metal when `S` is large enough. GQA attention
stays on the CPU. With `AVIARY_CLUSTER=1`, Metal still runs for local experts;
placed remotes RPC to peers (who may also use Metal on `EXEC_EXPERT`).
## Convert weights yourself (optional)

Only needed if you prefer regenerating from the upstream BF16/FP8 checkpoint:

```bash
cd c
python coli convert --repo Qwen/Qwen3-30B-A3B-Instruct-2507 --model /path/to/qwen3_i4
# or from a local checkout:
python tools/convert_qwen3_moe.py --indir /path/to/model --outdir /path/to/qwen3_i4
```

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

# agent — lean RAM budget (expert LRU sized from RAM_GB)
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://MASTER:9000 \
  --host 0.0.0.0 --port 8001 --ram 10

# agent — with GPU expert tier (requires: make qwen3_moe CUDA=1)
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://MASTER:9000 \
  --host 0.0.0.0 --port 8001 --ram 10 --gpu 0 --vram 8
```

| Flag / env | Effect |
|---|---|
| `--ram N` / `RAM_GB` | Expert-cache budget (GB). Without it the engine uses ~88% of free RAM and may raise `cap` aggressively. |
| `--cap N` | Initial LRU slots/layer (default 8 for Qwen). `RAM_GB` can still raise/lower this. |
| `--gpu 0` / `--vram N` | CUDA hot-expert tier (`COLI_CUDA=1`, `CUDA_EXPERT_GB=N`). Needs a CUDA build of `qwen3_moe`. |
| `COLI_METAL=1` | Apple Silicon Metal MoE (needs `make qwen3_moe METAL=1`). See [metal.md](metal.md). `--gpu` is CUDA-only. |
| `--gpu none` | Force CPU-only. |

Expect `[RAM_GB=10.0] …` (no `auto`) on stderr when `--ram` is set.

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
