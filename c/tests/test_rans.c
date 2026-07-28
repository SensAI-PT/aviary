/* rans.h codec + int4-rans256-g0 record oracle.
 *
 * A codec has no tolerance band: every check in this file is byte identity.
 * Covered:
 *   1. stream round-trip property suite — random and adversarial nibble
 *      streams (all-zero, all-max, single-symbol, 15-symbol boundary, empty,
 *      length edges) across several table shapes;
 *   2. checked-decode invariants: a genuine encoder output always passes
 *      the state-range / full-consumption / final-state checks, and seeded
 *      corruptions are refused with a named error;
 *   3. record framing: encode -> parse -> decode round trip at the format's
 *      real N=256 plus small-N shapes, and the full named-refusal battery
 *      (truncation, count mismatch, offset malformations, short streams,
 *      nonzero padding, trailing garbage);
 *   4. batched-arm identity: scalar, scalar_bf and every compiled vector arm
 *      (NEON here, AVX-512 on x86 builds) must produce byte-identical packed
 *      output on the whole suite — plus the kill-switch and forced-path
 *      refusal semantics (a forced unavailable arm is an error, never a
 *      silent downgrade).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../rans.h"

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } } while (0)

/* xorshift rng, deterministic across platforms */
static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

/* ---- table construction helpers ---------------------------------------- */

typedef struct {
    uint32_t scale_bits;
    uint32_t freq[16], start[16];
    uint16_t *slot;
    rans_table t;
} test_table;

/* Build a valid table for the given frequency spec (must sum to 1<<sb). */
static void tt_build(test_table *tt, uint32_t sb, const uint32_t *freq) {
    uint32_t M = 1u << sb;
    tt->scale_bits = sb;
    memcpy(tt->freq, freq, sizeof(tt->freq));
    uint64_t acc = 0;
    for (int s = 0; s < 16; s++) { tt->start[s] = (uint32_t)acc; acc += freq[s]; }
    CHECK(acc == M, "table spec must sum to M");
    tt->slot = (uint16_t *)malloc(sizeof(uint16_t) * M);
    for (int s = 0; s < 16; s++)
        for (uint32_t i = 0; i < freq[s]; i++) tt->slot[tt->start[s] + i] = (uint16_t)s;
    rans_err e = rans_table_init(&tt->t, sb, tt->freq, tt->start, tt->slot);
    CHECK(e == RANS_OK, "rans_table_init: %s", rans_err_name(e));
}

/* Build a table from the data's own histogram (largest-remainder-lite: floor
 * shares + deficit to the most frequent present symbol). Any valid table
 * round-trips; closeness to optimal is not what these tests measure. */
static void tt_from_hist(test_table *tt, uint32_t sb, const uint8_t *data, size_t n) {
    uint32_t M = 1u << sb;
    uint64_t hist[16] = {0};
    for (size_t i = 0; i < n; i++) hist[data[i]]++;
    uint32_t freq[16] = {0};
    uint32_t used = 0;
    int top = -1;
    for (int s = 0; s < 16; s++) {
        if (!hist[s]) continue;
        freq[s] = (uint32_t)(hist[s] * M / (n ? n : 1));
        if (freq[s] == 0) freq[s] = 1;
        used += freq[s];
        if (top < 0 || hist[s] > hist[top]) top = s;
    }
    if (top < 0) { top = 0; freq[0] = 0; }
    if (used < M) freq[top] += M - used;
    else if (used > M) {
        uint32_t over = used - M;
        CHECK(freq[top] > over, "histogram quantization underflow");
        freq[top] -= over;
    }
    tt_build(tt, sb, freq);
}

static void tt_free(test_table *tt) {
    rans_table_free(&tt->t);
    free(tt->slot);
}

static void pack_ref(const uint8_t *nib, uint64_t n, uint8_t *out) {
    memset(out, 0, (size_t)((n + 1) / 2));
    for (uint64_t i = 0; i < n; i++)
        out[i >> 1] |= (uint8_t)(nib[i] << ((i & 1) * 4));
}

/* ---- 1. stream round-trip property suite -------------------------------- */

static void stream_roundtrip(const uint8_t *data, size_t n, test_table *tt,
                             const char *label) {
    size_t cap = n + n / 2 + 4096;
    uint8_t *buf = (uint8_t *)malloc(cap);
    uint8_t *dec = (uint8_t *)malloc(n ? n : 1);
    size_t off = rans_encode_stream(data, n, tt->t.freq, tt->t.start,
                                    tt->t.scale_bits, buf, cap);
    CHECK(off != (size_t)-1, "%s: encode overflow", label);
    if (off != (size_t)-1) {
        rans_err e = rans_decode_stream_checked(buf + off, cap - off, n,
                                                tt->t.freq, tt->t.start,
                                                tt->slot, tt->t.scale_bits, dec);
        CHECK(e == RANS_OK, "%s: checked decode: %s", label, rans_err_name(e));
        CHECK(n == 0 || memcmp(data, dec, n) == 0, "%s: round-trip mismatch", label);
        /* the plain decoder must agree with the checked one */
        memset(dec, 0xEE, n ? n : 1);
        int rc = rans_decode_stream(buf + off, cap - off, n, tt->t.freq,
                                    tt->t.start, tt->slot, tt->t.scale_bits, dec);
        CHECK(rc == 0, "%s: plain decode rc=%d", label, rc);
        CHECK(n == 0 || memcmp(data, dec, n) == 0, "%s: plain round-trip", label);
    }
    free(buf); free(dec);
}

static void test_stream_property_suite(void) {
    const size_t sizes[] = {0, 1, 2, 3, 5, 16, 255, 256, 257, 1000, 4096, 100003};
    /* random 16-symbol data, uniform table */
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        size_t n = sizes[si];
        uint8_t *d = (uint8_t *)malloc(n ? n : 1);
        for (size_t i = 0; i < n; i++) d[i] = (uint8_t)(rnd() & 15);
        uint32_t uni[16];
        for (int s = 0; s < 16; s++) uni[s] = (1u << 14) / 16;
        test_table tt;
        tt_build(&tt, 14, uni);
        char label[64];
        snprintf(label, sizeof(label), "uniform n=%zu", n);
        stream_roundtrip(d, n, &tt, label);
        tt_free(&tt);
        /* same data, histogram-matched table */
        if (n) {
            tt_from_hist(&tt, 14, d, n);
            snprintf(label, sizeof(label), "hist n=%zu", n);
            stream_roundtrip(d, n, &tt, label);
            tt_free(&tt);
        }
        free(d);
    }
    /* adversarial patterns at a few sizes */
    const size_t an = 8192;
    uint8_t *d = (uint8_t *)malloc(an);
    test_table tt;

    memset(d, 0, an);                                  /* all-zero */
    tt_from_hist(&tt, 14, d, an);
    stream_roundtrip(d, an, &tt, "all-zero");
    tt_free(&tt);

    memset(d, 15, an);                                 /* all-max */
    tt_from_hist(&tt, 14, d, an);
    stream_roundtrip(d, an, &tt, "all-max");
    tt_free(&tt);

    for (size_t i = 0; i < an; i++) d[i] = 7;          /* single-symbol */
    uint32_t single[16] = {0};
    single[7] = 1u << 14;
    tt_build(&tt, 14, single);
    stream_roundtrip(d, an, &tt, "single-symbol freq=M");
    tt_free(&tt);

    /* 15-symbol boundary: value 0 never appears (the real corpus's shape) */
    for (size_t i = 0; i < an; i++) d[i] = (uint8_t)(1 + (rnd() % 15));
    tt_from_hist(&tt, 14, d, an);
    CHECK(tt.t.freq[0] == 0, "15-symbol table keeps freq[0]==0");
    stream_roundtrip(d, an, &tt, "15-symbol");
    tt_free(&tt);

    /* highly skewed: symbol 8 dominates, extremes rare (real-ish histogram) */
    for (size_t i = 0; i < an; i++) {
        uint32_t r = rnd() % 100;
        d[i] = (uint8_t)(r < 60 ? 8 : r < 80 ? 7 : r < 90 ? 9 : 1 + (rnd() % 15));
    }
    tt_from_hist(&tt, 14, d, an);
    stream_roundtrip(d, an, &tt, "skewed");
    tt_free(&tt);

    /* low-resolution table (scale_bits=6) on 16-symbol data */
    for (size_t i = 0; i < an; i++) d[i] = (uint8_t)(rnd() & 15);
    tt_from_hist(&tt, 6, d, an);
    stream_roundtrip(d, an, &tt, "scale_bits=6");
    tt_free(&tt);

    free(d);
}

/* ---- 2. checked-decode refusal semantics -------------------------------- */

static void test_stream_checked_refusals(void) {
    const size_t n = 4096;
    uint8_t *d = (uint8_t *)malloc(n);
    for (size_t i = 0; i < n; i++) d[i] = (uint8_t)(1 + (rnd() % 15));
    test_table tt;
    tt_from_hist(&tt, 14, d, n);
    size_t cap = n + n / 2 + 4096;
    uint8_t *buf = (uint8_t *)malloc(cap);
    uint8_t *dec = (uint8_t *)malloc(n);
    size_t off = rans_encode_stream(d, n, tt.t.freq, tt.t.start, tt.t.scale_bits,
                                    buf, cap);
    CHECK(off != (size_t)-1, "encode");
    size_t len = cap - off;
    uint8_t *stream = buf + off;

    /* truncated to under 4 bytes -> underrun */
    rans_err e = rans_decode_stream_checked(stream, 3, n, tt.t.freq, tt.t.start,
                                            tt.slot, tt.t.scale_bits, dec);
    CHECK(e == RANS_E_STREAM_UNDERRUN, "3-byte stream: got %s", rans_err_name(e));

    /* zeroed initial state -> state range */
    uint8_t *cp = (uint8_t *)malloc(len);
    memcpy(cp, stream, len);
    cp[0] = cp[1] = cp[2] = cp[3] = 0;
    e = rans_decode_stream_checked(cp, len, n, tt.t.freq, tt.t.start,
                                   tt.slot, tt.t.scale_bits, dec);
    CHECK(e == RANS_E_STREAM_STATE_RANGE, "zero state: got %s", rans_err_name(e));

    /* trailing garbage byte -> leftover */
    uint8_t *cp2 = (uint8_t *)malloc(len + 1);
    memcpy(cp2, stream, len);
    cp2[len] = 0x5A;
    e = rans_decode_stream_checked(cp2, len + 1, n, tt.t.freq, tt.t.start,
                                   tt.slot, tt.t.scale_bits, dec);
    CHECK(e == RANS_E_STREAM_LEFTOVER || e == RANS_E_STREAM_FINAL_STATE,
          "trailing byte: got %s", rans_err_name(e));

    /* single-bit corruption sweep: every flip must be refused (state-range /
     * leftover / final-state) — a corrupted stream that still PASSES the
     * checked decode would be a verifier hole, so enumerate, don't sample */
    int caught = 0, missed = 0;
    for (size_t bit = 0; bit < len * 8; bit += 7) {   /* stride keeps runtime sane */
        memcpy(cp, stream, len);
        cp[bit / 8] ^= (uint8_t)(1u << (bit % 8));
        e = rans_decode_stream_checked(cp, len, n, tt.t.freq, tt.t.start,
                                       tt.slot, tt.t.scale_bits, dec);
        if (e != RANS_OK) caught++;
        else if (memcmp(dec, d, n) != 0) missed++;    /* passed checks, wrong bytes */
        else CHECK(0, "bit flip at %zu decoded to the original bytes", bit);
    }
    CHECK(missed == 0, "%d corruptions passed the checked decode silently "
          "(%d refused)", missed, caught);

    free(cp); free(cp2); free(buf); free(dec); free(d);
    tt_free(&tt);
}

/* ---- 3. record framing: round trip + refusal battery --------------------- */

static uint8_t *make_record(const uint8_t *nib, uint64_t n, test_table *tt,
                            uint32_t N, uint64_t *out_len) {
    uint64_t cap = rans_record_bound(n, N) + RANS_SLACK;
    uint8_t *buf = (uint8_t *)calloc(cap, 1);
    rans_err e = rans_record_encode(nib, n, &tt->t, N, buf, cap, out_len);
    CHECK(e == RANS_OK, "record encode: %s", rans_err_name(e));
    return buf;
}

static void record_roundtrip(const uint8_t *nib, uint64_t n, test_table *tt,
                             uint32_t N, const char *label) {
    uint64_t rec_len = 0;
    uint8_t *buf = make_record(nib, n, tt, N, &rec_len);
    rans_record rec;
    rans_err e = rans_record_parse(buf, rec_len, N, &rec);
    CHECK(e == RANS_OK, "%s: parse: %s", label, rans_err_name(e));
    if (e == RANS_OK) {
        CHECK(rec.n_symbols == n, "%s: n_symbols", label);
        CHECK(rec.packed_bytes == (n + 1) / 2, "%s: packed_bytes", label);
        uint64_t np = rec.packed_bytes;
        uint8_t *ref = (uint8_t *)malloc(np);
        uint8_t *got = (uint8_t *)malloc(np);
        pack_ref(nib, n, ref);
        memset(got, 0xEE, np);
        e = rans_record_decode_packed(&rec, &tt->t, N, RANS_PATH_SCALAR, got);
        CHECK(e == RANS_OK && memcmp(ref, got, np) == 0,
              "%s: scalar packed decode", label);
        /* determinism: re-encoding the decoded stream reproduces the record */
        uint8_t *nib2 = (uint8_t *)malloc(n);
        for (uint64_t i = 0; i < n; i++)
            nib2[i] = (uint8_t)((got[i >> 1] >> ((i & 1) * 4)) & 15);
        uint64_t len2 = 0;
        uint8_t *buf2 = make_record(nib2, n, tt, N, &len2);
        CHECK(len2 == rec_len && memcmp(buf, buf2, rec_len) == 0,
              "%s: re-encode determinism", label);
        free(buf2); free(nib2); free(ref); free(got);
    }
    free(buf);
}

static void test_record_roundtrip(void) {
    /* the format's real stream count, on sizes below/at/above N and a
     * realistically-sized block; odd n exercises the generic path */
    const uint64_t sizes[] = {1, 2, 100, 255, 256, 257, 512, 8192, 12289, 65536};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        uint64_t n = sizes[si];
        uint8_t *d = (uint8_t *)malloc(n);
        for (uint64_t i = 0; i < n; i++) d[i] = (uint8_t)(1 + (rnd() % 15));
        test_table tt;
        tt_from_hist(&tt, 14, d, n);
        char label[64];
        snprintf(label, sizeof(label), "record N=256 n=%llu", (unsigned long long)n);
        record_roundtrip(d, n, &tt, RANS_NSTREAMS, label);
        tt_free(&tt);
        free(d);
    }
    /* small stream counts (kernel widths and non-multiples) */
    const uint32_t Ns[] = {8, 16, 24, 32};
    for (size_t ni = 0; ni < sizeof(Ns) / sizeof(Ns[0]); ni++) {
        uint64_t n = 4096 + 2 * ni;
        uint8_t *d = (uint8_t *)malloc(n);
        for (uint64_t i = 0; i < n; i++) d[i] = (uint8_t)(rnd() & 15);
        test_table tt;
        tt_from_hist(&tt, 14, d, n);
        char label[64];
        snprintf(label, sizeof(label), "record N=%u", Ns[ni]);
        record_roundtrip(d, n, &tt, Ns[ni], label);
        tt_free(&tt);
        free(d);
    }
}

static void test_record_refusals(void) {
    const uint32_t N = RANS_NSTREAMS;
    const uint64_t n = 8192;
    uint8_t *d = (uint8_t *)malloc(n);
    for (uint64_t i = 0; i < n; i++) d[i] = (uint8_t)(1 + (rnd() % 15));
    test_table tt;
    tt_from_hist(&tt, 14, d, n);
    uint64_t rec_len = 0;
    uint8_t *buf = make_record(d, n, &tt, N, &rec_len);
    rans_record rec;
    uint64_t head_raw = 16 + ((uint64_t)N + 1) * 4;   /* before padding */
    uint64_t head = rans_record_header_bytes(N);
    uint8_t *cp = (uint8_t *)calloc(rec_len + 16 + RANS_SLACK, 1);
    rans_err e;

#define EXPECT(err_want, what) do { \
        e = rans_record_parse(cp, cp_len, N, &rec); \
        CHECK(e == (err_want), "%s: want %s got %s", what, \
              rans_err_name(err_want), rans_err_name(e)); \
    } while (0)

    uint64_t cp_len;

    /* truncated inside the header */
    memcpy(cp, buf, rec_len); cp_len = head - 1;
    EXPECT(RANS_E_TRUNCATED, "short header");

    /* truncated inside the payload (cut to a 16-byte boundary so the length
     * check sees a plausible-but-short record) */
    memcpy(cp, buf, rec_len); cp_len = head + 16;
    EXPECT(RANS_E_TRUNCATED, "short payload");

    /* n_symbols = 0 */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    memset(cp, 0, 16);
    EXPECT(RANS_E_EMPTY, "empty");

    /* packed_bytes disagreeing with n_symbols */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    cp[8] ^= 1;
    EXPECT(RANS_E_COUNT_MISMATCH, "count mismatch");

    /* offsets[0] != 0 */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    cp[16] = 1;
    EXPECT(RANS_E_OFFSET_FIRST, "offset[0]");

    /* non-monotonic offsets */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    { uint32_t v; memcpy(&v, cp + 16 + 8 * 4, 4); v += 1u << 20;
      memcpy(cp + 16 + 8 * 4, &v, 4); }
    EXPECT(RANS_E_OFFSETS_MONOTONIC, "monotonic");

    /* a stream shorter than its flushed state: collapse stream 3 to 0 bytes */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    { uint32_t v; memcpy(&v, cp + 16 + 3 * 4, 4);
      memcpy(cp + 16 + 4 * 4, &v, 4); }
    EXPECT(RANS_E_STREAM_SHORT, "stream short");

    /* final offset larger than the blob -> truncated */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    { uint32_t v; memcpy(&v, cp + 16 + (uint64_t)N * 4, 4); v += 4096;
      memcpy(cp + 16 + (uint64_t)N * 4, &v, 4); }
    EXPECT(RANS_E_TRUNCATED, "offsets out of bounds");

    /* trailing garbage past the derived record end */
    memcpy(cp, buf, rec_len); cp_len = rec_len + 16;
    EXPECT(RANS_E_LENGTH_MISMATCH, "trailing garbage");

    /* nonzero header padding */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    CHECK(head > head_raw, "N=256 framing leaves header pad bytes");
    cp[head_raw] = 0x77;
    EXPECT(RANS_E_PAD_NONZERO, "header pad");

    /* nonzero payload padding (only if this record has any) */
    memcpy(cp, buf, rec_len); cp_len = rec_len;
    { uint32_t plen; memcpy(&plen, cp + 16 + (uint64_t)N * 4, 4);
      if (rec_len > head + plen) {
          cp[head + plen] = 0x77;
          EXPECT(RANS_E_PAD_NONZERO, "payload pad");
      } }

#undef EXPECT

    /* table validation refusals */
    {
        rans_table t2;
        uint32_t bad_freq[16];
        memcpy(bad_freq, tt.freq, sizeof(bad_freq));
        bad_freq[15] += 1;   /* sum != M (last symbol: start[] stays coherent,
                                so the sum check is what fires) */
        e = rans_table_init(&t2, tt.scale_bits, bad_freq, tt.start, tt.slot);
        CHECK(e == RANS_E_TABLE_FREQ_SUM, "freq sum: %s", rans_err_name(e));

        uint32_t bad_start[16];
        memcpy(bad_start, tt.start, sizeof(bad_start));
        bad_start[5] += 1;                             /* not the prefix sum */
        e = rans_table_init(&t2, tt.scale_bits, tt.freq, bad_start, tt.slot);
        CHECK(e == RANS_E_TABLE_START, "start prefix: %s", rans_err_name(e));

        uint32_t M = 1u << tt.scale_bits;
        uint16_t *bad_slot = (uint16_t *)malloc(sizeof(uint16_t) * M);
        memcpy(bad_slot, tt.slot, sizeof(uint16_t) * M);
        bad_slot[M / 2] = 16;                          /* symbol out of range */
        e = rans_table_init(&t2, tt.scale_bits, tt.freq, tt.start, bad_slot);
        CHECK(e == RANS_E_TABLE_SLOT, "slot range: %s", rans_err_name(e));
        memcpy(bad_slot, tt.slot, sizeof(uint16_t) * M);
        bad_slot[0] = tt.slot[M - 1];                  /* slot outside its band */
        e = rans_table_init(&t2, tt.scale_bits, tt.freq, tt.start, bad_slot);
        CHECK(e == RANS_E_TABLE_SLOT, "slot band: %s", rans_err_name(e));
        free(bad_slot);

        e = rans_table_init(&t2, 0, tt.freq, tt.start, tt.slot);
        CHECK(e == RANS_E_TABLE_SCALE, "scale 0: %s", rans_err_name(e));
    }

    free(cp); free(buf); free(d);
    tt_free(&tt);
}

/* ---- 4. batched-arm byte identity --------------------------------------- */

static void arms_identity(const uint8_t *nib, uint64_t n, test_table *tt,
                          uint32_t N, const char *label) {
    uint64_t rec_len = 0;
    uint8_t *buf = make_record(nib, n, tt, N, &rec_len);
    rans_record rec;
    rans_err e = rans_record_parse(buf, rec_len, N, &rec);
    CHECK(e == RANS_OK, "%s: parse: %s", label, rans_err_name(e));
    uint64_t np = (n + 1) / 2;
    uint8_t *ref = (uint8_t *)malloc(np);
    uint8_t *got = (uint8_t *)malloc(np);
    pack_ref(nib, n, ref);

    const rans_path arms[] = {RANS_PATH_SCALAR, RANS_PATH_SCALAR_BF,
                              RANS_PATH_NEON, RANS_PATH_AVX512};
    for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
        rans_path p = arms[a];
        memset(got, 0xEE, np);
        e = rans_record_decode_packed(&rec, &tt->t, N, p, got);
        if (e == RANS_E_PATH_UNAVAILABLE) {
            /* arm not compiled into this build: refusal (never a silent
             * scalar run) is exactly the required behavior */
            int compiled = 1;
#ifndef __ARM_NEON
            if (p == RANS_PATH_NEON) compiled = 0;
#endif
#if !(defined(__AVX512F__) && defined(__AVX512BW__))
            if (p == RANS_PATH_AVX512) compiled = 0;
#endif
            CHECK(!compiled, "%s: %s refused despite being compiled",
                  label, rans_path_name(p));
            continue;
        }
        CHECK(e == RANS_OK, "%s: %s decode: %s", label, rans_path_name(p),
              rans_err_name(e));
        CHECK(memcmp(ref, got, np) == 0,
              "%s: %s output differs from packed reference", label,
              rans_path_name(p));
    }
    free(ref); free(got); free(buf);
}

static void test_arm_identity_suite(void) {
    /* size sweep at the real N: even sizes drive the group kernels + tails,
     * odd sizes the generic path; per-stream lengths straddle the kernels'
     * full-round/tail boundaries */
    const uint64_t sizes[] = {256, 512, 4096, 4098, 8192, 12288, 12289,
                              65536, 262144, 262146};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        uint64_t n = sizes[si];
        uint8_t *d = (uint8_t *)malloc(n);
        for (uint64_t i = 0; i < n; i++) d[i] = (uint8_t)(1 + (rnd() % 15));
        test_table tt;
        tt_from_hist(&tt, 14, d, n);
        char label[64];
        snprintf(label, sizeof(label), "arms n=%llu", (unsigned long long)n);
        arms_identity(d, n, &tt, RANS_NSTREAMS, label);
        tt_free(&tt);
        free(d);
    }
    /* adversarial content through the arms too */
    const uint64_t an = 32768;
    uint8_t *d = (uint8_t *)malloc(an);
    test_table tt;
    memset(d, 0, an);
    tt_from_hist(&tt, 14, d, an);
    arms_identity(d, an, &tt, RANS_NSTREAMS, "arms all-zero");
    tt_free(&tt);
    memset(d, 15, an);
    tt_from_hist(&tt, 14, d, an);
    arms_identity(d, an, &tt, RANS_NSTREAMS, "arms all-max");
    tt_free(&tt);
    for (uint64_t i = 0; i < an; i++) d[i] = 9;
    uint32_t single[16] = {0};
    single[9] = 1u << 14;
    tt_build(&tt, 14, single);
    arms_identity(d, an, &tt, RANS_NSTREAMS, "arms single-symbol");
    tt_free(&tt);

    /* rare-symbol table: low-frequency outliers force the decoder through
     * post-update states far below RANS_L, where the renorm has to refill 2+
     * bytes at once — comfortable tables never visit the [2^14, 2^15) band,
     * so a renorm-threshold off-by-one in a vector arm is invisible without
     * this case. Power-of-two freqs alone do NOT reach it either (the shift
     * algebra keeps floor(log2 x) on a rigid lattice, measured directly);
     * the f=3 group is what actually populates the band (~3% of steps).
     * Found by the deliberate-sabotage exercise; keep forever. */
    uint32_t rare[16] = {0};
    for (int s = 1; s <= 7; s++) rare[s] = 1;
    for (int s = 8; s <= 14; s++) rare[s] = 3;
    rare[15] = (1u << 14) - 7 - 21;
    tt_build(&tt, 14, rare);
    for (uint64_t i = 0; i < an; i++)
        d[i] = (uint8_t)((rnd() % 3 == 0) ? 1 + (rnd() % 14) : 15);
    arms_identity(d, an, &tt, RANS_NSTREAMS, "arms rare-symbol");
    stream_roundtrip(d, an, &tt, "rare-symbol stream");
    tt_free(&tt);
    free(d);
}

static void test_path_envelope(void) {
#ifdef _WIN32
    _putenv_s("RANS_PATH", ""); _putenv_s("RANS_NEON", ""); _putenv_s("RANS_AVX512", "");
#else
    unsetenv("RANS_PATH"); unsetenv("RANS_NEON"); unsetenv("RANS_AVX512");
#endif
    /* default selection is never invalid, and its name is real */
    rans_path best = rans_path_select();
    CHECK(best != RANS_PATH_INVALID, "default path select");
    CHECK(strcmp(rans_path_name(best), "invalid") != 0, "best path name");

    /* kill-switches force the vector arms out of best-path selection */
#ifdef _WIN32
#define SET_ENV(k, v) _putenv_s(k, v)
#define UNSET_ENV(k) _putenv_s(k, "")
#else
#define SET_ENV(k, v) setenv(k, v, 1)
#define UNSET_ENV(k) unsetenv(k)
#endif
    SET_ENV("RANS_NEON", "0");
    SET_ENV("RANS_AVX512", "0");
    rans_path p = rans_path_select();
    CHECK(p == RANS_PATH_SCALAR_BF, "kill-switches leave scalar_bf, got %s",
          rans_path_name(p));

    /* forcing a killed arm must refuse, not downgrade */
    SET_ENV("RANS_PATH", "neon");
    p = rans_path_select();
    CHECK(p == RANS_PATH_INVALID, "forced killed neon: got %s", rans_path_name(p));
    SET_ENV("RANS_PATH", "avx512");
    p = rans_path_select();
    CHECK(p == RANS_PATH_INVALID, "forced killed avx512: got %s", rans_path_name(p));

    /* unknown forced name refuses */
    SET_ENV("RANS_PATH", "warp-drive");
    p = rans_path_select();
    CHECK(p == RANS_PATH_INVALID, "unknown forced path");

    /* forcing scalar always works */
    SET_ENV("RANS_PATH", "scalar");
    p = rans_path_select();
    CHECK(p == RANS_PATH_SCALAR, "forced scalar");

    UNSET_ENV("RANS_PATH");
    UNSET_ENV("RANS_NEON");
    UNSET_ENV("RANS_AVX512");

    /* with the switches cleared, any compiled+selftested vector arm is
     * preferred over the scalar twins */
#ifdef __ARM_NEON
    p = rans_path_select();
    CHECK(p == RANS_PATH_NEON || p == RANS_PATH_AVX512,
          "vector arm preferred on this build, got %s", rans_path_name(p));
#endif
#undef SET_ENV
#undef UNSET_ENV
}

int main(void) {
    test_stream_property_suite();
    test_stream_checked_refusals();
    test_record_roundtrip();
    test_record_refusals();
    test_arm_identity_suite();
    test_path_envelope();
    if (failures) {
        printf("test_rans: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_rans: ok\n");
    return 0;
}
