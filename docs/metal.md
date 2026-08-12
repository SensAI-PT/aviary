# Metal backend (Apple Silicon, experimental)

On Apple Silicon the decode profile is matmul-bound, and unified memory removes
the PCIe copy tax that keeps CUDA's streaming experts on the CPU — so colibrì
has an opt-in Metal backend that runs the **routed-expert SwiGLU (batched,
zero-copy from the RAM slabs)**, and on GLM also the **fused decode attention**
(full MLA layer in one command buffer, S≤4) and **prefill's large GEMMs** on the
GPU.

**Engines:** `colibri` (GLM) gets MoE + fused MLA attention + GEMM. `hy3` and
`qwen3_moe` get MoE (fmt 1/2/4) + dense GEMM; **GQA attention stays on the CPU**.
`inkling` uses the same MoE path. Decode is token-exact vs the CPU path for MoE
fmt 1/2/4 (grouped int4 uses per-group scales, same as dense `mm_gemv`).

Prefill's large GEMMs run on the GPU in a
different accumulation order, so on **near-tie logits** they can occasionally pick a
different top token than the CPU — a floating-point ordering difference, not a kernel
bug (`make metal-test` passes the GEMM at ~3e-6 against a 1e-4 tolerance; see
[#622](https://github.com/JustVugg/colibri/issues/622)). It is invisible in normal use
but can surface in teacher-forced oracle comparisons on pathological 4-bit toy
containers. Set `COLI_METAL_GEMM_MIN=100000` to keep every GEMM on the CPU for
bit-exact prefill (`DEBUG_LOGITS=1` on a `TF=1` run dumps the top-5 logits and the
top1–top2 margin at each mismatch, so you can see how close the tie was).

`COLI_METAL_PREFILL=1` extends the fused attention to **prefill** (S>4) on GLM only: the whole
attention — projections, scores, softmax, value, output — runs on the GPU in one
command buffer instead of the CPU. On a 544-token prompt this cuts prefill attention
~4x (35.9 s → 9.0 s). It is **off by default**: like the prefill GEMM above, the GPU
accumulates in a different order and can pick a different top token on near-tie logits
(same [#622](https://github.com/JustVugg/colibri/issues/622) family), so a greedy stream
is not guaranteed bit-identical to the CPU — on natural prompts it stays consistent, on
pathological repetitive prompts an early token can flip. Turn it on when prefill latency
matters more than exact CPU parity; prompts past the single-dispatch thread cap fall
back to the CPU automatically.

**Platform:** macOS / Apple Silicon only. `make … METAL=1` is refused on Linux
(including WSL) with `METAL=1 is supported only on macOS` — use `CUDA=1` there
instead. No Xcode app is required (the Metal shader compiles at runtime);
`xcode-select --install` is enough for the clang toolchain.

**OpenMP:** Apple clang does not ship `omp.h`. Metal builds `#include <omp.h>`
(for `omp_in_parallel` guards around GPU dispatch), so without Homebrew
libomp the compile fails with `omp.h file not found`. Install once:

```bash
brew install libomp
```

A plain CPU build can limp along single-threaded without libomp; `METAL=1`
cannot.

```bash
cd c
make colibri METAL=1          # macOS + libomp; shader compiles at runtime
make hy3 METAL=1              # MoE + GEMM; GQA attention on CPU
make qwen3_moe METAL=1        # same as hy3
                              # any macOS SDK builds; the COLI_METAL_RESSET residency-set
                              # path needs the macOS 15 SDK and is compiled out below it
make metal-test           # standalone kernel/attention correctness vs CPU reference
COLI_METAL=1 COLI_MODEL=/path/glm52_i4 ./coli chat --ram 96
COLI_METAL=1 COLI_NO_OMP_TUNE=1 DIRECT=1 PIPE=1 \
  COLI_MODEL=/path/hy3_i4 ./coli chat --ram 56
COLI_METAL=1 COLI_NO_OMP_TUNE=1 DIRECT=1 PIPE=1 \
  COLI_MODEL=/path/qwen3_i4 ./coli chat --ram 16
```

Measured on an M4 Max (128 GB, warm cache, MTP on): CPU 0.30 → Metal
**0.42 tok/s (~1.4×)** (best config adds `DIRECT=1`; ~3× vs this machine's
first cold run). An M5 Max with a 46.9 GB learned pin reached **2.06 tok/s**
([#103](https://github.com/JustVugg/colibri/issues/103); see also the
[M5 Max performance report](METAL-M5MAX-PERF-REPORT.md)).

Key design points: Metal's ~5 ms submit latency makes per-matmul dispatch a
loss — everything is batched into few command buffers per layer, and the
resident experts' GPU work is submitted *before* the missed experts' disk reads
so I/O and compute overlap. `COLI_METAL_GEMM_MIN` tunes the prefill GEMM row
threshold (default 16). Streaming, cache, MTP, DSA and the persistence formats
are unchanged; every GPU path falls back to the CPU per-block on any fault.
Numerics are dequant→f32-MAC (same as the CUDA tier); greedy outputs are
byte-identical to the CPU engine.

**Aviary cluster:** with `AVIARY_CLUSTER=1`, hy3/qwen still use Metal for
experts executed locally. Placement remotes go through `EXEC_EXPERT` RPC first;
peers can run that single-expert forward on Metal too. Empty placement (or
RPC miss) falls back to local Metal/CPU — Metal is no longer disabled wholesale
when the cluster flag is on.
