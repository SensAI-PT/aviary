# colibrì 🐦

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a consumer machine with ~25 GB of RAM — in pure C, with zero dependencies, by streaming experts from disk.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ pronto in 32s · residente 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## The idea

A 744B Mixture-of-Experts model activates only ~40B parameters per token — and only ~11 GB of those change from token to token (the routed experts). So:

- the **dense part** (attention, shared experts, embeddings — ~17B params) stays **resident in RAM at int4** (~9.9 GB);
- the **21,504 routed experts** (75 MoE layers × 256 experts + the MTP head, ~19 MB each at int4) live **on disk** (~370 GB) and are **streamed on demand**, with a per-layer LRU cache, an optional pinned hot-store, and the OS page cache as a free L2.

The engine is a single C file (`c/glm.c`, ~1,300 lines) plus small headers. No BLAS, no Python at runtime, no GPU.

## What's implemented

- **Faithful GLM-5.2 (`glm_moe_dsa`) forward** — validated token-exact against a `transformers` oracle (teacher-forcing 32/32, greedy 20/20 on a tiny-random model with the real architecture).
- **MLA attention** (q/kv-LoRA, interleaved partial RoPE) with **compressed KV-cache**: 576 floats/token instead of 32,768 (57× smaller — GLM-5.2 has 64 heads and no GQA).
- **DeepSeek-V3-style sigmoid router** (noaux_tc, routed_scaling_factor), shared expert, first-3-dense layers.
- **Native MTP speculative decoding** — GLM-5.2's own multi-token-prediction head (layer 78) drafts tokens that the main model verifies in one batched forward. Measured **2.00 tokens/forward (100% acceptance)** on structured text. Lossless: output identical to greedy.
- **Quantization kernels**: int8 / packed int4 / packed int2, per-row scales, AVX2, dequant-on-use. Packing validated bit-identical to the int8 container.
- **Batch-union MoE**: in prefill (and MTP verification), each unique expert of the batch is read once and applied to every position that routes to it.
- **Byte-level BPE tokenizer in C** (GPT-2-style with Unicode-property regex, 320k merges).
- **RAM safety**: the expert cache is auto-sized from `MemAvailable` at startup — an honest peak projection (working set, KV, MTP row, reconstruction buffers) so the kernel OOM-killer never fires.
- **Offline FP8→int4 converter** (`c/convert_fp8_to_int4.py`): downloads one shard at a time (~5 GB), dequants (128×128 block scales), requantizes to the engine's container, deletes the shard — the 756 GB FP8 checkpoint never needs to exist on disk at once. Resumable.

## Honest numbers (WSL2, 12 cores, 25 GB RAM, NVMe via VHDX)

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (VHDX random) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation | 2.0 tok/forward measured |

This is not fast. It is a 744B frontier-class model **answering correctly on a machine that costs less than one H100 fan**. Warm cache, pinned hot experts and MTP push the useful-response latency down considerably; the physics of the disk does the rest.

## Quick start

```bash
cd c
./setup.sh                      # checks gcc/OpenMP, builds, self-tests

# convert the model (resumable, needs ~400 GB free on a real ext4/NVMe path):
./coli convert                  # from zai-org/GLM-5.2-FP8

# chat (RAM budget and expert cache size itself automatically):
COLI_MODEL=/path/to/glm52_i4 ./coli chat
```

Useful knobs (env or flags): `--topp 0.7` adaptive expert top-p (30–40% less disk), `--ngen N` max tokens, `STATS=f`/`PIN=f PIN_GB=g` record expert usage then pin the hottest in RAM, `THINK=1` enable GLM-5.2's reasoning block, `DRAFT=n` MTP draft depth, `TF=1` teacher-forcing validation.

## Repo layout

```
c/glm.c          the engine (GLM-5.2 forward, streaming MoE, MTP, serve mode)
c/st.h           safetensors reader: pread + fadvise, no mmap (RSS stays flat)
c/tok.h          byte-level BPE tokenizer in C
c/coli           CLI: chat / run / bench / convert / info
c/convert_fp8_to_int4.py   disk-safe FP8 → int4 converter
c/make_glm_oracle.py       tiny-random oracle generator for validation
c/olmoe.c        stage-A engine (OLMoE), first validation target
*.py             research scripts (cost model, trace analysis, py engine)
```

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand flowers a day. This engine keeps a 744-billion-parameter giant alive on hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
