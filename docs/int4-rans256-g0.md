# `int4-rans256-g0` — lossless entropy-coded container tier for per-row int4 experts

Status: format specification + offline tools, PR 1 of a 3-PR ladder. This PR
adds the codec (`c/rans.h`), the repack/verify tools
(`c/tools/repack_rans.py`, `c/tools/rans_verify.py`) and this document —
**no engine code paths change**. PR 2 (loader + CPU decode on expert load)
and PR 3 (Metal batched decode) build on it. A registry row is *proposed*
here, not claimed: the format's identity is the NAME string
`int4-rans256-g0`; the numeric `fmt` ordinal is left for the maintainer to
assign whenever a registry (FORMATS.md) lands.

## What it is, in one paragraph

`int4-rans256-g0` losslessly entropy-codes the packed int4 weight bytes of
routed-expert projections (`gate_proj`/`up_proj`/`down_proj`) with a
**single static rANS table shared per shard**, laid out as **256 independent
round-robin-interleaved streams per tensor** so wide decoders (SIMD lanes,
GPU threadgroups) get coalesced output. On a full-corpus census of a per-row
int4 GLM-5.2 container the achieved ratio is ~0.76 (≈24% fewer bytes to
store and stream), with byte-exact reconstruction — this is a codec, not a
quantizer: decode reproduces the original packed bytes exactly, always.

Name breakdown: `int4` — source alphabet is 4-bit quantization codes
(nominally 16, effectively 15 symbols: the symmetric quantizer never emits
the -8 code point, and the table handles that automatically via a zero
frequency); `rans256` — rANS entropy coding, 256-way interleave; `g0` —
shared static table, generation 0.

## Container shape

Output is **ordinary safetensors shards**. Each entropy-coded weight tensor
keeps its **original, unmodified logical name** (so any name-indexed loader
finds it with zero special-casing), stored as dtype `U8`, shape
`[record_len]` — an opaque byte blob whose contents are the chunk record
below. The `.qs` per-row scale sidecars are **unchanged raw F32** (they are
~0.1% of the byte stream on per-row geometry; coding them is not worth a
second codec path in v1).

### Chunk record (one record = one weight tensor's bytes)

All integers little-endian. `N` = 256 streams.

```
offset 0:  n_symbols     u64   -- nibble count for this tensor
offset 8:  packed_bytes  u64   -- original packed-byte count (= ceil(n_symbols/2)),
                                  stored so a reader can size its output buffer
offset 16: stream_offsets[N+1]  u32 x (N+1)
                                -- offsets[i] = byte offset, relative to the start
                                   of `payload`, where stream i begins;
                                   offsets[N] = total payload length
then:      zero-pad to the next 16-byte boundary (derived, never stored)
payload:   N independent rANS byte streams, concatenated in stream order:
           stream i is payload[offsets[i] : offsets[i+1]]
then:      zero-pad to the next 16-byte boundary (derived, never stored)
```

The two padding runs are computed identically by writer and reader
(`round16()`), never stored — a stored length could disagree with the
derived value. Validators refuse nonzero padding bytes. Alignment is
**record-relative**, not file-absolute: safetensors does not 16-align
tensor offsets, so a consumer that wants aligned u32/u64 access must read or
map the tensor's bytes into an aligned allocation (any `malloc`/GPU-buffer
allocation qualifies).

### Interleaving

Nibbles are unpacked low-nibble-first (`nib[2k] = byte[k] & 0xF`,
`nib[2k+1] = byte[k] >> 4`). Nibble at logical index `j` belongs to stream
`j % N` and is the `(j / N)`-th symbol of that stream. Every stream is an
ordinary, self-contained, single-state rANS stream over the same shared
table — no cross-stream coupling. Round-robin (not block) assignment is the
property that makes wide decode output-coalesced: lane `l` of a group based
at stream `b` emits logical position `r*N + b + l` at round `r`, so a
G-lane group writes G contiguous nibbles = G/2 whole packed bytes.

### Per-stream codec

ryg_rans-style construction (Fabian Giesen's public-domain reference
design): 32-bit state, byte renormalization, `L = 2^23`,
`scale_bits = 14` (`M = 16384`).

- encode walks the input backwards, writes bytes backwards, flushes the
  final 4-byte state; decode reads forward:
  `x = 4 bytes big-endian`, then per symbol:
  `slot = x & (M-1)`; `s = slot_to_symbol[slot]`;
  `x = freq[s]*(x >> scale_bits) + slot - start[s]`;
  `while (x < L) { if bytes remain: x = (x<<8)|next byte; else break; }`.
- The `else break` is load-bearing: a stream's final symbols legally decode
  with `x < L`; a decoder that reads past the stream end produces wrong
  tail symbols.
- Verifiable invariants of any genuine encoder output (validators use them
  to refuse corruption **without ground-truth bytes**): the initial 4-byte
  state lies in `[L, 256L)`; decode consumes the stream exactly; the final
  state is exactly `L`.

### Shared table, in shard metadata

The table ships **once per shard** in
`__metadata__["colibri.int4-rans256-g0.table"]`, a JSON string:

```json
{
  "table_id": "g0",
  "n_streams": 256,
  "scale_bits": 14,
  "M": 16384,
  "freq": [16 uint32, summing to M],
  "start": [16 uint32, the exclusive prefix sum of freq],
  "slot_to_symbol_b64": "<base64 of M x uint16, little-endian>",
  "table_crc32": "<hex crc32 of the raw slot_to_symbol bytes>"
}
```

Decode is always shard-local: every record in a shard decodes against that
shard's embedded copy, no cross-shard or external-file dependency. A
multi-shard writer must stamp the byte-identical table into every shard it
emits (the repack tool builds the table once, from the pooled histogram of
every selected tensor, and reuses the same JSON string). `table_crc32`
lets tooling confirm two shards used the identical table without diffing
the base64 blob. `table_id` is provenance metadata (a retuned future table
would ship as `g1`); readers never resolve it externally.

## THE STAMP IS MANDATORY (the load-bearing difference from fmt=7)

`__metadata__["colibri.fmt"]` maps each entropy-coded **weight** tensor
name (never `.qs`) to the string `int4-rans256-g0`, JSON-encoded with
sorted keys.

For the existing formats the engine can infer identity from byte arithmetic
(`weight bytes == formula(O, I)`), and a stamp is at most a cross-check.
**That inference is structurally impossible here**: entropy-coded size is
data-dependent — there is no `expected_bytes(O, I)` to compare against. The
stamp is therefore the **only** signal that a `U8` tensor is entropy-coded
at all. Consequences any consumer must respect:

- an *unstamped* `U8` tensor must never be presumed entropy-coded by any
  size heuristic;
- a future engine read path needs a stamp-gated dispatch **ahead of** the
  byte-arithmetic format inference: stamped `int4-rans256-g0` → rANS decode
  path; anything else → today's inference, completely unchanged. Getting
  that order wrong would misread a compressed blob as a fixed-ratio format
  and corrupt every weight with no error;
- validators refuse a shard whose stamps and table are not both present and
  coherent (`tools/rans_verify.py` is the reference validator, with a named
  refusal per corruption class).

## Scope (v1)

Per-row ("gs=0") int4 expert containers only. Per-group (g64-class)
containers are out of scope: their better-normalized codes measure a much
weaker ratio (~0.89 vs ~0.76) and their scale arrays are a two-orders-larger
share of the stream — a g64 round would need scale coding to be worth
doing, which is its own design. The repack tool *refuses* (by name) a
tensor whose weight-bytes/scale-rows ratio is the g64 signature rather than
silently coding it under this format's identity.

## Reference implementation map

- `c/rans.h` — dependency-free C99, header-only: scalar codec, record
  writer/reader with named refusals, batched decode arms (scalar, branch-free
  scalar, NEON, AVX-512 F+BW). Vector arms are byte-identical to scalar by
  contract and carry a compile-time ISA gate, a first-use round-trip
  selftest (a failing arm disables itself loudly) and env kill-switches
  (`RANS_NEON=0`, `RANS_AVX512=0`; `RANS_PATH=<arm>` forces one and refuses
  if unavailable — never a silent downgrade).
- `make rans` — builds `tools/librans_c.*`, the ctypes bridge the Python
  tools prefer; without it they fall back to a pure-Python codec that emits
  identical bytes, slowly.
- `c/tools/repack_rans.py` — writer. Deterministic: same input ⇒
  byte-identical shards. Verifies every record byte-exact before writing.
- `c/tools/rans_verify.py` — validator (TRUST-VERIFY-REFUSE; the doc of
  refusal classes is its module docstring).
- `c/tests/test_rans.c`, `c/tests/test_rans_repack.py` — property suite,
  arm-identity sweep, framing/refusal battery, e2e round trip.

A third-party consumer needs only this document plus the scalar decode
description above to read records: parse the shard header, find the stamp
and table in `__metadata__`, locate a tensor's record by name, and decode
its 256 streams with the table (in any order or in parallel).

## Proposed registry row

| ordinal | name | weight bytes | scale layout | status |
|---|---|---|---|---|
| *(maintainer-assigned)* | `int4-rans256-g0` | data-dependent (chunk record above; **stamp mandatory**) | per-row `F32` `.qs`, raw (unchanged from int4-row) | proposed, offline tools only |
