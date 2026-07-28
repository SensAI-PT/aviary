#!/usr/bin/env python3
"""int4-rans256-g0 repack tool: per-row-int4 expert shards -> an
entropy-coded container (lossless, byte-exact round trip guaranteed and
verified before anything is written).

WHAT IT DOES
  1. Scans every input shard's header (header-only reads, cheap) and builds a
     corpus-wide {tensor -> shard} index — routed-expert tensors and their
     `.qs` sidecars can live in different shards.
  2. Selects routed-expert projection weights
     (model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight, dtype U8)
     whose `.qs` sidecar is per-row F32 — the per-row ("gs=0") int4 geometry
     this format's v1 targets. GEOMETRY GUARD (enforced, not assumed): for a
     per-row container weight_bytes/scale_rows == I/2, a per-group (g64)
     container measures exactly 32 — any tensor at ratio 32, at a
     non-integral ratio, or below MIN_ROW_RATIO is REFUSED by name rather
     than silently entropy-coded under the wrong format identity.
  3. Builds ONE global static rANS table from the pooled nibble histogram of
     every selected tensor (pass 1), then encodes each tensor as a 256-stream
     round-robin-interleaved chunk record (pass 2). The byte-identical table
     is stamped into EVERY output shard (the encoder invariant: readers are
     shard-local, so each shard must carry the same copy).
  4. Verifies EVERY record byte-exact (decode -> repack -> compare against
     the original weight bytes) before its shard is written. Not sampled.
  5. Writes ordinary safetensors shards: each weight tensor keeps its
     ORIGINAL logical name (its bytes are the chunk record), `.qs` sidecars
     pass through raw and untouched, and __metadata__ carries:
       colibri.fmt                      = JSON {weight_name: "int4-rans256-g0"}
       colibri.int4-rans256-g0.table    = JSON shared-table blob
     The colibri.fmt stamp is MANDATORY for this format: entropy-coded sizes
     are data-dependent, so no byte-arithmetic inference exists — the stamp
     is the only signal a U8 tensor is entropy-coded at all (see
     docs/int4-rans256-g0.md, "the stamp is load-bearing").

DETERMINISM: two runs over the same input produce byte-identical output —
sorted tensor order, sorted JSON keys, no timestamps, a codec whose output
is a pure function of (bytes, table). The e2e test diffs two runs.

Build `make rans` first for the C codec (tools/librans_c.*); without it this
tool falls back to a pure-Python codec that emits the same bytes orders of
magnitude slower (fine for the test fixtures, not for a real container).

Usage:
  python3 tools/repack_rans.py --indir <int4_dir> --outdir <out> --dry-run
  python3 tools/repack_rans.py --indir <int4_dir> --outdir <out> [--layers 20,55]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rans_format as rf  # noqa: E402

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

# Per-row geometry guard (see module docstring). MIN_ROW_RATIO = I/2 for the
# smallest hidden dim this tool accepts; the load-bearing exclusion is 32,
# the exact ratio EVERY per-group-64 container measures regardless of shape.
G64_RATIO = 32
MIN_ROW_RATIO = 64


def build_index(indir):
    paths = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    index = {}
    for p in paths:
        header, data_start = rf.read_header(p)
        for name in header:
            if name == "__metadata__":
                continue
            index[name] = (p, header, data_start)
    return index, paths


def select_targets(index, layers):
    """Returns sorted list of selected weight names; refuses (loud, named)
    any expert tensor whose geometry is not per-row int4."""
    selected = []
    for name in sorted(index):
        m = EXPERT_RE.match(name)
        if not m:
            continue
        if layers is not None and int(m.group(1)) not in layers:
            continue
        _, header, _ = index[name]
        info = header[name]
        if info["dtype"] != "U8":
            continue                     # not a packed-int4 weight (e.g. int8 MTP)
        qs = name + ".qs"
        if qs not in index:
            raise rf.RansRefusal(
                "E_GEOMETRY_NO_SCALES",
                f"{name}: no {qs} sidecar anywhere in the corpus — refusing to "
                f"entropy-code a tensor whose format cannot be confirmed")
        qpath, qheader, _ = index[qs]
        qinfo = qheader[qs]
        if qinfo["dtype"] != "F32":
            raise rf.RansRefusal(
                "E_GEOMETRY_NOT_PER_ROW",
                f"{name}: {qs} dtype {qinfo['dtype']} != F32")
        wb = info["data_offsets"][1] - info["data_offsets"][0]
        sb = qinfo["data_offsets"][1] - qinfo["data_offsets"][0]
        if sb % 4 or sb == 0:
            raise rf.RansRefusal("E_GEOMETRY_NOT_PER_ROW",
                                 f"{name}: {qs} is {sb} bytes (not F32-shaped)")
        rows = sb // 4
        if wb % rows:
            raise rf.RansRefusal(
                "E_GEOMETRY_NOT_PER_ROW",
                f"{name}: weight bytes {wb} not divisible by scale rows {rows}")
        ratio = wb // rows
        if ratio == G64_RATIO:
            raise rf.RansRefusal(
                "E_GEOMETRY_G64",
                f"{name}: weight/scale ratio {ratio} is the per-group-64 "
                f"signature — g64 containers are out of this format's v1 scope "
                f"(materially weaker entropy payoff; needs its own round)")
        if ratio < MIN_ROW_RATIO:
            raise rf.RansRefusal(
                "E_GEOMETRY_NOT_PER_ROW",
                f"{name}: weight/scale ratio {ratio} < {MIN_ROW_RATIO} — not a "
                f"per-row int4 shape this tool recognizes")
        selected.append(name)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True,
                    help="directory of per-row-int4 *.safetensors shards")
    ap.add_argument("--outdir", required=True,
                    help="destination for the entropy-coded container shards")
    ap.add_argument("--layers",
                    help="comma-separated layer subset (default: every layer)")
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory only: scan headers, print selection, write nothing")
    a = ap.parse_args()

    layers = None
    if a.layers:
        layers = {int(x) for x in a.layers.split(",")}

    index, shard_paths = build_index(a.indir)
    if not shard_paths:
        print(f"ERROR: no *.safetensors files in {a.indir}", file=sys.stderr)
        sys.exit(1)
    try:
        selected = select_targets(index, layers)
    except rf.RansRefusal as exc:
        print(f"REFUSED {exc}", file=sys.stderr)
        sys.exit(1)
    if not selected:
        print(f"ERROR: no per-row int4 routed-expert tensors selected under "
              f"{a.indir} — nothing to repack; refusing to emit an empty "
              f"container", file=sys.stderr)
        sys.exit(1)

    by_shard = {}
    total_w = 0
    for name in selected:
        p, header, _ = index[name]
        by_shard.setdefault(p, []).append(name)
        total_w += header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
    print(f"[select] {len(selected)} weight tensor(s), {total_w / 1e9:.3f} GB "
          f"packed bytes, across {len(by_shard)} input shard(s)"
          + (f", layers={sorted(layers)}" if layers else ""))
    if a.dry_run:
        for p in sorted(by_shard):
            print(f"[dry-run] {os.path.basename(p)}: {len(by_shard[p])} tensor(s)")
        print(f"[dry-run] codec: {'C (librans_c)' if rf.LIB else 'pure Python'}")
        return

    if rf.LIB is None:
        print("[warn] tools/librans_c.* not built (make rans): using the "
              "pure-Python codec — identical bytes, orders of magnitude slower",
              file=sys.stderr)

    # pass 1: pooled histogram -> the ONE global table
    hist = np.zeros(16, dtype=np.int64)
    for name in selected:
        p, header, data_start = index[name]
        raw, _ = rf.read_tensor_bytes(p, header, data_start, name)
        hist += np.bincount(rf.unpack_nibbles(raw), minlength=16)
    freq = rf.quantize_freq(hist)
    start, slot = rf.build_table(freq)
    table_json = json.dumps(rf.table_blob(freq, start, slot), sort_keys=True)
    table = rf.parse_table_blob(table_json)     # writer trusts nothing, not even itself
    print(f"[table] pooled histogram {hist.tolist()}")
    print(f"[table] freq {freq.tolist()} (M={rf.M}, scale_bits={rf.SCALE_BITS})")

    # pass 2: encode + verify + write, one output shard per input shard
    os.makedirs(a.outdir, exist_ok=True)
    out_n = 0
    total_record = 0
    for p in sorted(by_shard):
        tensors = []
        fmt_map = {}
        for name in by_shard[p]:
            _, header, data_start = index[name]
            raw, info = rf.read_tensor_bytes(p, header, data_start, name)
            nib = rf.unpack_nibbles(raw)
            record = rf.build_record(nib, freq, start, slot)
            # verify BEFORE writing: full checked decode, byte-exact repack
            back = rf.decode_record(record, table)
            if back != raw:
                print(f"FATAL: {name}: decode(encode(x)) != x — refusing to "
                      f"write a container that does not round-trip", file=sys.stderr)
                sys.exit(1)
            tensors.append((name, "U8", [len(record)], record))
            fmt_map[name] = rf.FORMAT_NAME
            total_record += len(record)
            qs = name + ".qs"
            qp, qheader, qdata_start = index[qs]
            qraw, qinfo = rf.read_tensor_bytes(qp, qheader, qdata_start, qs)
            tensors.append((qs, qinfo["dtype"], qinfo["shape"], qraw))
        metadata = {
            rf.METADATA_KEY: json.dumps(fmt_map, sort_keys=True),
            rf.TABLE_KEY: table_json,
        }
        out_path = os.path.join(a.outdir, f"out-rans-{out_n:05d}.safetensors")
        rf.write_shard(out_path, tensors, metadata)
        print(f"[write] {out_path}: {len(tensors)} tensor(s) "
              f"({os.path.getsize(out_path) / 1e9:.4f} GB) "
              f"from {os.path.basename(p)}")
        out_n += 1

    ratio = total_record / total_w
    print(f"[ratio] record/original: {ratio:.4f} ({(1 - ratio) * 100:.2f}% "
          f"reduction incl. framing), {total_w / 1e6:.1f} MB -> "
          f"{total_record / 1e6:.1f} MB")
    print(f"[done] {out_n} shard(s), every record round-trip-verified byte-exact")


if __name__ == "__main__":
    main()
