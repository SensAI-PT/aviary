/* fmt=7 (native FP8-e4m3 passthrough) loader-seam tests.
 *
 * fmt=7, PUBLIC ordinal assigned by the maintainer on #524: this format was
 * minted fmt=6 during original development of this branch, before dev's own
 * #465 (E8/IQ3) claimed that ordinal upstream and merged it into dev as a
 * REAL fmt=6 (see quant.h's E8 constants and e8_ helper functions, and
 * qt_resolve_fmt's ns==4-tag early check); re-tagged fmt=100 (PRIVATE ORDINAL
 * BLOCK, see colibri.c's QT struct comment) from that point forward -- there
 * was never a build in this branch's history where this format was reachable
 * as fmt=6 -- and graduated to fmt=7 at merge.
 *
 * Part A: qt_resolve_fmt disambiguation suite -- THE DESIGN LANDMINE. fmt=7
 * weight bytes are byte-identical to fmt=1 (int8): both are O*I raw bytes.
 * The two are told apart ONLY by the scale array's byte count (per-row O*4 for
 * fmt=1, per-128x128-block ceil(O/128)*ceil(I/128)*4 for fmt=7, THIS build's
 * implemented f32 scale encoding). For some shapes those two counts coincide
 * exactly -- INVERSION (maintainer review, #528): qt_resolve_fmt used to
 * REFUSE (exit(1)) this ambiguous case; it now resolves to fmt=1 (the
 * incumbent, already-on-disk, decodable format) instead, because the
 * collision is not hypothetical (GLM-5.2's own self_attn.o_proj.weight hits
 * it, see qt_resolve_fmt's own "REVIEW FINDING"/"INVERSION" comment) and the
 * writer side (repack_fp8_passthrough.py's _check_geometry) now refuses to
 * ever EMIT an fmt=7 container at this same shape, so an unstamped ambiguous
 * tensor reaching this function is never a genuine fmt=7 candidate. The
 * former refusal-testing convention (fork()+waitpid(), mirroring
 * tests/test_st_pread.c's exit(1)-path idiom) is kept for the OTHER
 * refusing cases below (Part A2's fmt=6 collision, Part A3's UE8M0
 * recognized-not-implemented refusal, and the generic garbage-byte-count
 * refusal) -- only the is_row&&is_blk collision in this Part flipped from
 * expect_refuse to expect_fmt(...,1,...).
 *
 * Part A2: fmt=6 (E8/IQ3, upstream #465) vs fmt=7 collision at [O<=128 or
 * O in a 128-block-count range, I=98] -- SECOND DESIGN LANDMINE. Unchanged by
 * the #528 inversion above (a different collision predicate, still refused).
 *
 * Part A3: fmt=7's scale ENCODING is a declared property, not a hardcoded
 * constant -- f32 (Part A/A2 above) is what this build implements. A UE8M0
 * (1 byte/block) encoding is a REAL, distinct byte signature (the DeepSeek-V4
 * checkpoint format for this identical weight geometry) this build recognizes
 * and refuses BY NAME rather than misreading. Unchanged by the #528
 * inversion (a stamp confirms the WEIGHT format, never a decoder this build
 * doesn't have -- see qt_resolve_fmt's own comment).
 *
 * Part B: qt_from_disk loader-seam -- writes a real single-shard .safetensors
 * file containing an fmt=7 tensor (U8 weight + per-block F32 .qs) next to an
 * int8 control tensor of a DIFFERENT, non-colliding shape, loads both through
 * qt_from_disk, and checks the byte-count/.qs-size inference picks fmt=7 vs
 * fmt=1 correctly and the loaded weights dequantize identically to a reference.
 * Mirrors tests/test_int3_load.c's structure for fmt=5.
 *
 * Part C: qt_bytes()/qt_scale_bytes() byte-accounting for fmt=7, plus
 * qt_wire_split() -- the shared weight/scale byte-range split qt_wire_mmap
 * and qt_unwire_mmap both now call (maintainer review, #528: qt_scale_bytes()
 * existed correctly but neither call site actually used it, a defect only
 * -Wno-unused-function's suppression let compile clean). check_wire_split()
 * exercises qt_wire_split() itself directly; test_wire_site_regression()
 * (FIX ROUND, validator finding) additionally exercises the real
 * qt_wire_mmap()/qt_unwire_mmap() call sites through a mem_wire()/munlock()
 * observer seam (defined right below the #include below) -- a mutation that
 * reverts ONLY those two call sites back to the old scale_b=(int64_t)t->O*4
 * hardcode, leaving qt_wire_split() itself untouched, is invisible to
 * check_wire_split() but fails test_wire_site_regression() (proven by
 * actually running that exact mutation -- see the report).
 *
 * This build has NO container metadata stamp (see qt_resolve_fmt's own header
 * comment) -- every ambiguous/unimplemented-encoding shape below EXCEPT the
 * is_row&&is_blk collision (now resolved to fmt=1, see Part A above) refuses
 * unconditionally; stamp-resolves-ambiguity behavior for the OTHER
 * collisions is a separate, follow-up PR (registry + metadata stamp), not
 * exercised here. */
/* WIRE-SITE REGRESSION SEAM (FIX ROUND, validator finding). First attempt
 * (superseded, kept as a note): renaming mem_wire() itself via macro does
 * NOT work as an observer seam -- mem_wire is a real, internally-defined
 * static function, so the SAME rename that frees up the name "mem_wire"
 * for a shadow ALSO renames every CALL SITE (qt_wire_mmap's) to the new
 * name, meaning qt_wire_mmap ends up calling the renamed-but-still-real
 * function directly, bypassing any shadow defined under the old name
 * entirely (confirmed by inspecting the preprocessed output -- caught
 * before it could hide a broken test). The seam that actually works is one
 * level lower: mlock()/munlock() themselves are EXTERNAL POSIX library
 * functions with no body anywhere in this translation unit (only a
 * declaration, via <sys/mman.h>, plus mem_wire's/qt_unwire_mmap's own call
 * sites) -- renaming them redirects those call sites to a name THIS FILE
 * provides its own (self-contained) definition for, with no real
 * implementation being shadowed out of existence. This observes the EXACT
 * (addr,len) qt_wire_mmap (via mem_wire) and qt_unwire_mmap actually pass
 * down to the platform lock/unlock call -- not a reimplementation of what
 * they SHOULD pass, and immune to a future mem_wire refactor since the
 * seam sits at the syscall boundary, not the wrapper. #ifndef _WIN32:
 * mlock/munlock are only called on this `#if defined(__APPLE__) ||
 * defined(__linux__) || defined(__FreeBSD__)` arm; Windows uses
 * compat_mlock/compat_munlock instead (untouched here, matching this
 * file's existing POSIX-only test-seam convention -- the fork/pipe/waitpid
 * refusal tests below skip analogously on Windows). */
#ifndef _WIN32
#define mlock test_mlock_seam
#define munlock test_munlock_seam
#endif
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#ifndef _WIN32
#undef mlock
#undef munlock
#endif

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

/* Shadow definitions for the seam above -- must come after the #include so
 * mem_wire's (unmodified, real) call to mlock() and qt_unwire_mmap's
 * (unmodified, real) call to munlock() have already been renamed to these
 * names by the #define above. Neither mlock() nor munlock() has a body
 * anywhere in this translation unit (both are declared only, via
 * <sys/mman.h>) -- these are the ONLY definitions the renamed call sites
 * can resolve to, both self-contained: returning 0 (success) without
 * actually locking/unlocking anything is fine for this test, since the
 * buffer under test is never really mlocked in the first place (mirrors
 * check_fp8_bytes'/check_wire_split's own stated reasoning that the real
 * RLIMIT_MEMLOCK-gated syscall's success is environment-dependent and not
 * what any of these tests need to prove). Non-static: both must match the
 * extern linkage of the (renamed) declarations <sys/mman.h> already left in
 * this translation unit -- a static definition here would conflict with
 * that non-static declaration ("static declaration follows non-static
 * declaration"). g_seam_wire_* observes mem_wire's (hence qt_wire_mmap's)
 * calls; g_seam_unwire_* observes qt_unwire_mmap's direct calls. */
#ifndef _WIN32
static void *g_seam_wire_addr[4]; static size_t g_seam_wire_len[4]; static int g_seam_wire_n;
int test_mlock_seam(const void *addr, size_t len){
    if(g_seam_wire_n < 4){ g_seam_wire_addr[g_seam_wire_n]=(void*)addr; g_seam_wire_len[g_seam_wire_n]=len; }
    g_seam_wire_n++;
    return 0;
}
static void *g_seam_unwire_addr[4]; static size_t g_seam_unwire_len[4]; static int g_seam_unwire_n;
int test_munlock_seam(const void *addr, size_t len){
    if(g_seam_unwire_n < 4){ g_seam_unwire_addr[g_seam_unwire_n]=(void*)addr; g_seam_unwire_len[g_seam_unwire_n]=len; }
    g_seam_unwire_n++;
    return 0;
}
#endif

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint64_t rng = 0xFEEDFACE0DDBA11ull;
static float rndf(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000; }
static uint8_t rndbyte_nonan(void){
    for(;;){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        uint8_t b = (uint8_t)(rng & 0xFF);
        if(b != 0x7F && b != 0xFF) return b; }
}

/* ---- Part A: qt_resolve_fmt disambiguation (in-process for non-refusing
 * cases, fork+waitpid for refusing ones -- qt_resolve_fmt exit(1)s in place,
 * it does not return an error code). ---- */

static int expect_fmt(int O, int I, int64_t nb, int64_t ns, int expect_fmt_val, const char *tag){
    int gs=0;
    int fmt = qt_resolve_fmt(tag, O, I, nb, ns, &gs);
    if(fmt != expect_fmt_val){
        printf("FAIL %s: got fmt=%d, expected fmt=%d (O=%d I=%d nb=%lld ns=%lld)\n",
               tag, fmt, expect_fmt_val, O, I, (long long)nb, (long long)ns);
        return 0;
    }
    return 1;
}

static int expect_refuse(int O, int I, int64_t nb, int64_t ns, const char *tag){
#ifndef _WIN32
    int pipefd[2]; if(pipe(pipefd)!=0) return 0;
    pid_t pid = fork();
    if(pid < 0) return 0;
    if(pid == 0){
        dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
        int gs=0;
        qt_resolve_fmt(tag, O, I, nb, ns, &gs);   /* must exit(1) inside; must NOT return */
        _exit(42);                                  /* reaching here is the bug */
    }
    close(pipefd[1]);
    char err[1024]={0}; ssize_t n=read(pipefd[0],err,sizeof(err)-1); (void)n;
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
    if(!ok){
        printf("FAIL %s: expected exit(1) refusal, got status=%d, stderr=%.200s\n", tag, status, err);
        return 0;
    }
    if(!strstr(err,"refus")){
        printf("FAIL %s: exited(1) but message lacked a refusal explanation: %.200s\n", tag, err);
        return 0;
    }
    return 1;
#else
    /* fork/pipe/waitpid are POSIX -- mirrors tests/test_st_pread.c's own
     * Windows arm: skip the exit(1) subprocess check, print an explicit
     * (never silent) skip line per case, count it as passing rather than
     * failing. Every non-refusing case in this file (expect_fmt and its
     * callers) still runs and asserts for real on Windows. */
    printf("skipped on Windows (no fork): %s\n", tag);
    (void)O; (void)I; (void)nb; (void)ns;
    return 1;
#endif
}

static void test_disambiguation(void){
    /* --- non-degenerate golden paths (unambiguous either way) --- */
    CHECK(expect_fmt(4096,4096,(int64_t)4096*4096,4096*4,1,"plain int8 4096x4096"));
    /* spec's own worked example: [2048,6144] expert -> block scale [16,48] */
    CHECK(expect_fmt(2048,6144,(int64_t)2048*6144,16LL*48*4,7,"fp8 2048x6144 (spec example)"));
    CHECK(expect_fmt(384,6144,(int64_t)384*6144,3LL*48*4,7,"fp8 384x6144 non-square block grid"));

    /* --- REGRESSION (maintainer review, #528): GLM-5.2's own
     * self_attn.o_proj.weight, [D,H*v_head]=[6144,16384]. nblkO=ceil(6144/128)=48,
     * nblkI=ceil(16384/128)=128, product=6144==O -- this shape hits the
     * is_row&&is_blk collision below on every GLM-5.2 checkpoint, and is a
     * REAL, pre-existing, valid int8-row tensor, not a hypothetical. Confirmed
     * against this repo's own v1 container header (o_proj weight U8
     * 100,663,296 B == 6144*16384; .qs scale blob 24,576 B == 6144*4 ==
     * 48*128*4, both at once) -- see the CENSUS SCAN
     * (tools/fp8_collision_census.py) for the full enumerated family. An
     * earlier revision of qt_resolve_fmt refused this shape unconditionally
     * (exit(1) at load time on an ordinary model); the INVERSION resolves it
     * to fmt=1 instead -- this is that non-refusal, asserted directly. */
    CHECK(expect_fmt(6144,16384,(int64_t)6144*16384,6144LL*4,1,
        "GLM-5.2 o_proj shape [6144,16384]: valid int8-row, ambiguous-by-byte-count, resolves fmt=1 (NOT a refusal)"));

    /* --- degenerate shapes: O<=128 makes nblkO==1, so ns_blk==nblkI*4 can
     * equal ns_row==O*4 whenever nblkI==O. Exhaustive-in-spirit sweep of the
     * boundary the design landmine describes. INVERSION (maintainer review,
     * #528): every one of these used to be an expect_refuse (exit(1)) -- the
     * general family this collision predicate describes (any int8 tensor
     * where O==ceil(O/128)*ceil(I/128)) includes real, valid, pre-existing
     * tensors (see the o_proj case just above), so refusing was the wrong
     * call across the board, not just for the three shapes the maintainer's
     * review named explicitly ([128,16384],[256,16384],[384,16384] below) --
     * every case in this sweep hits the exact same is_row&&is_blk branch and
     * is flipped here for the same reason. --- */
    CHECK(expect_fmt(1,1,      1,        4,  1, "degenerate O=1 I=1 (nblkI=1=O) -> fmt=1, was refuse"));
    CHECK(expect_fmt(1,128,    128,      4,  1, "degenerate O=1 I=128 (nblkI=1=O, I at block edge) -> fmt=1, was refuse"));
    CHECK(expect_fmt(2,256,    2LL*256,  8,  1, "degenerate O=2 I=256 (nblkI=2=O) -> fmt=1, was refuse"));
    CHECK(expect_fmt(6,768,    6LL*768,  24, 1, "degenerate O=6 I=768 (nblkI=6=O) -> fmt=1, was refuse"));
    /* the three shapes the maintainer's review named explicitly (his exact expect_refuse
     * calls, O=128/256/384 at I=16384) -- FLIPPED to assert fmt=1. */
    CHECK(expect_fmt(128,16384,128LL*16384, 512, 1, "degenerate O=128 I=16384 (nblkO=1,nblkI=128=O) -> fmt=1, was refuse"));
    /* O>128 degenerate case: nblkO=2, need nblkI=O/2 -- O=256,I=16384 -> nblkI=128, 2*128=256=O */
    CHECK(expect_fmt(256,16384,256LL*16384, 1024, 1, "degenerate O=256 I=16384 (nblkO=2,nblkI=128, product=O) -> fmt=1, was refuse"));
    /* k=3: the ambiguity isn't a one-off O=256 coincidence -- it's structural for ANY
     * O that's a multiple of 128 (nblkO=k), since nblkI==128 (I in (16256,16384]) always
     * makes nblkO*nblkI == k*128 == O. One more multiple (O=384=3*128) confirms the
     * condition generalizes past the k=2 worked example, not just a re-derivation. */
    CHECK(expect_fmt(384,16384,384LL*16384, 1536, 1, "degenerate O=384 I=16384 (nblkO=3,nblkI=128, k=3, product=O) -> fmt=1, was refuse"));

    /* --- boundary-ADJACENT non-degenerate cases: one step past each
     * degenerate case above, both interpretations now legitimately resolve. --- */
    CHECK(expect_fmt(1,129, 129,   4, 1, "adjacent O=1 I=129 as fmt=1 (ns=row)"));
    CHECK(expect_fmt(1,129, 129,   8, 7, "adjacent O=1 I=129 as fmt=7 (ns=block, nblkI=2)"));
    CHECK(expect_fmt(2,257, 2LL*257, 8,  1, "adjacent O=2 I=257 as fmt=1 (ns=row)"));
    CHECK(expect_fmt(2,257, 2LL*257, 12, 7, "adjacent O=2 I=257 as fmt=7 (ns=block, nblkI=3)"));

    /* --- neither interpretation matches: garbage .qs size, must still refuse
     * (the pre-existing generic-mismatch path, exercised through the fmt=7-aware
     * function to confirm the new code didn't disturb it). --- */
    CHECK(expect_refuse(10,10, 100, 999, "garbage ns matches neither row nor block layout"));
}

/* ---- Part A2: fmt=6 (E8/IQ3, upstream #465, merged into dev) vs fmt=7
 * (this branch's fp8-e4m3-b128) collision -- SECOND DESIGN LANDMINE, see the
 * derivation in qt_resolve_fmt's own comment. e8_rowbytes(I) is the constant
 * 98 for every I in (0,256], so dev's fmt=6 tag check (ns==4 &&
 * nb==O*e8_rowbytes(I)) collapses to nb==O*98 -- which coincides with
 * fp8-e4m3-b128's raw weight bytes (O*I) at the ONE value I==98, where a
 * SINGLE-BLOCK (O<=128) fp8 tensor with f32 block scales, OR a FOUR-BLOCK
 * fp8 tensor with ue8m0 (1 byte/block) scales, ALSO carries exactly ns==4,
 * same as the E8 tag. An unstamped [I=98] fp8-e4m3-b128 tensor at either of
 * those shapes is therefore byte-for-byte indistinguishable from a genuine
 * fmt=6 tensor: nb AND ns both coincide, not just ns (contrast the fmt=1/
 * fmt=7 collision in Part A, where only ns ever coincides). O==1 stacks a
 * THIRD candidate: fmt=1's per-row ns (O*4) is also 4 there. This build has
 * no stamp to resolve any of these -- every one refuses. */
static void test_fmt6_fp8_collision(void){
    /* O=64, I=98: nblkO=nblkI=1 (fmt=7, single block, f32 scales) -> ns=4;
     * e8_blocks(98)=1 -> nb=O*98=6272 for BOTH interpretations, and fmt=6's
     * .qs tag is always exactly one f32 -> ns=4 too. Must refuse. */
    int64_t nb64=(int64_t)64*98;
    CHECK(expect_refuse(64,98, nb64, 4, "fmt=6/fmt=7(f32) collision O=64 I=98"));
    /* boundary O=128 variant: still nblkO=1 for fmt=7 (128<=128), same collision. */
    int64_t nb128=(int64_t)128*98;
    CHECK(expect_refuse(128,98, nb128, 4, "fmt=6/fmt=7(f32) collision O=128 I=98"));

    /* O=1, I=98: a THIRD candidate stacks on (fmt=1 plain int8 per-row, ns==O*4==4
     * too) -- a genuine three-way ambiguity. Must refuse. */
    int64_t nb1=(int64_t)1*98;
    CHECK(expect_refuse(1,98, nb1, 4, "fmt=1/fmt=6/fmt=7(f32) THREE-way collision O=1 I=98"));

    /* O in (384,512], I=98: nblkO=4, nblkI=1, product=4 -- a fmt=7 tensor with
     * UE8M0 (1 byte/block) scales also lands at ns==4*1==4 here, the SAME tag
     * fmt=6 uses. Must refuse (and, since this build doesn't implement ue8m0
     * decode at all, would refuse on that basis alone even without the fmt=6
     * collision -- this case exercises the OUTER fmt=6 guard specifically). */
    int64_t nb400=(int64_t)400*98;
    CHECK(expect_refuse(400,98, nb400, 4, "fmt=6/fmt=7(ue8m0, 4-block) collision O=400 I=98"));

    /* regression guard: a GENUINE (non-colliding) fmt=6 fixture -- I!=98, so
     * e8_rowbytes(I)==98 does NOT equal O*I -- must keep resolving to fmt=6.
     * Mirrors test_e8_kernel.c's own O=24,I=512 shape and a small
     * single-super-block shape (I=256, inside the (0,256] range where
     * e8_rowbytes(I)==98 but I!=98, so still non-colliding). */
    CHECK(expect_fmt(24,512, (int64_t)24*e8_rowbytes(512), 4, 6,
        "genuine fmt=6 (non-colliding) O=24 I=512 -> still resolves to fmt=6"));
    CHECK(expect_fmt(64,256, (int64_t)64*e8_rowbytes(256), 4, 6,
        "genuine fmt=6 (non-colliding) O=64 I=256 (I!=98, no collision) -> fmt=6"));
}

/* ---- Part A3: fmt=7's scale ENCODING is a declared property -- UE8M0
 * recognized, refused by name (not implemented in this build). ---- */
static void test_ue8m0_scale_refusal(void){
    /* [2048,6144] (spec example shape, same as Part A's fmt=7/f32 golden
     * path): nblkO=16, nblkI=48, product=768 blocks. A UE8M0 sidecar is
     * exactly 1 byte/block -> ns=768, distinct from BOTH fmt=1's per-row
     * count (O*4=8192) and this build's f32 block-scale count (768*4=3072).
     * Clean, unambiguous UE8M0 signature -- must name-refuse, not silently
     * treat it as a truncated/corrupt f32 array or match it to fmt=1. */
    int64_t nb=(int64_t)2048*6144;
    CHECK(expect_refuse(2048,6144, nb, 768,
        "fp8-e4m3-b128 with ue8m0 scales (spec-shaped, non-degenerate) -> recognized, refused by name"));

    /* O=1, I=400: nblkO=1, nblkI=ceil(400/128)=4, product=4 -- a UE8M0 sidecar
     * here is ns=4*1=4, which ALSO equals fmt=1's per-row count (O*4=4): the
     * same small-O regime that produces the f32-vs-fmt=1 collision in Part A
     * produces a ue8m0-vs-fmt=1 collision too. Must still refuse (combined
     * message), not silently pick fmt=1. */
    int64_t nb1=(int64_t)1*400;
    CHECK(expect_refuse(1,400, nb1, 4,
        "fp8-e4m3-b128 with ue8m0 scales, ALSO colliding with fmt=1 per-row (O=1) -> refused"));
}

/* ---- Part B: qt_from_disk loader-seam (real safetensors file) ---- */

static void deq_fmt7(const QT *t, float *dq){
    int64_t nblkI = fp8_nblk(t->I);
    for(int o=0;o<t->O;o++){
        int64_t blkO = o/FP8_BLOCK; const float *scl = t->s + blkO*nblkI;
        for(int i=0;i<t->I;i++){
            int64_t bi = i/FP8_BLOCK;
            dq[(int64_t)o*t->I+i] = e4m3_decode(t->q8[(int64_t)o*t->I+i]) * scl[bi];
        }
    }
}
static void deq_fmt1(const QT *t, float *dq){
    for(int o=0;o<t->O;o++){ float s=t->s[o];
        for(int i=0;i<t->I;i++) dq[(int64_t)o*t->I+i]=(float)t->q8[(int64_t)o*t->I+i]*s; }
}

#define CDIV(n,d) (((n)+(d)-1)/(d))

static void test_loader_seam(void){
    enum { O7=8, I7=256 };                        /* nblkO=1, nblkI=2 -> 2 block scales total */
    enum { O1=5, I1=64 };                        /* DIFFERENT shape from the fp8 tensor: O1*I1=320
                                                   * bytes, ns=O1*4=20 -- neither collides with the
                                                   * fp8 tensor's own byte counts (kept deliberately
                                                   * distinct so this is a plain, non-degenerate
                                                   * negative control, not another landmine case). */
    enum { NBLK7 = CDIV(O7,128) * CDIV(I7,128) };  /* must be a compile-time constant expression for
                                                     * the static array below -- fp8_nblk() is a real
                                                     * function (runtime, not constexpr), so it can't
                                                     * size a `static` array even with literal inputs. */
    static uint8_t q7[O7*I7]; static float s7[NBLK7];
    for(int i=0;i<O7*I7;i++) q7[i]=rndbyte_nonan();
    for(int i=0;i<(int)(sizeof s7/sizeof *s7);i++) s7[i]=0.01f+0.001f*(float)i;

    static int8_t q1[O1*I1]; static float s1[O1];
    for(int i=0;i<O1*I1;i++) q1[i]=(int8_t)(rndbyte_nonan()-128);
    for(int i=0;i<O1;i++) s1[i]=0.02f+0.001f*(float)i;

    const char *dir="tests/tmp_fp8_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    int64_t nb7=(int64_t)O7*I7, ns7=(int64_t)(sizeof s7);
    int64_t nb1=(int64_t)O1*I1, ns1=(int64_t)O1*4;
    char hdr[1024];
    int hl=snprintf(hdr,sizeof hdr,
        "{\"w7\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
        "\"w7.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"w1\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"w1.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
        (long long)nb7,(long long)nb7,
        (long long)(ns7/4),(long long)nb7,(long long)(nb7+ns7),
        (long long)nb1,(long long)(nb7+ns7),(long long)(nb7+ns7+nb1),
        (long long)O1,(long long)(nb7+ns7+nb1),(long long)(nb7+ns7+nb1+ns1));
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); fails++; return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    fwrite(q7,1,(size_t)nb7,f); fwrite(s7,1,(size_t)ns7,f);
    fwrite(q1,1,(size_t)nb1,f); fwrite(s1,1,(size_t)ns1,f);
    fclose(f);

    static Model gm;                             /* only gm.S is used by qt_from_disk */
    st_init(&gm.S, dir);

    QT t7; memset(&t7,0,sizeof t7);
    qt_from_disk(&gm,"w7",O7,I7,8,0,&t7);
    CHECK(t7.fmt==7);
    CHECK(t7.q8!=NULL && t7.s!=NULL);            /* both weight and scale allocated (qalloc, not falloc) */
    static float dq_load[O7*I7], dq_ref[O7*I7];
    deq_fmt7(&t7,dq_load);
    QT tr7={.fmt=7,.q8=(int8_t*)q7,.s=s7,.O=O7,.I=I7};
    deq_fmt7(&tr7,dq_ref);
    CHECK(memcmp(dq_load,dq_ref,sizeof dq_ref)==0);

    QT t1; memset(&t1,0,sizeof t1);
    qt_from_disk(&gm,"w1",O1,I1,8,0,&t1);
    CHECK(t1.fmt==1);                            /* negative control: plain int8 still resolves as fmt=1 */
    static float dq_load1[O1*I1], dq_ref1[O1*I1];
    deq_fmt1(&t1,dq_load1);
    QT tr1={.fmt=1,.q8=q1,.s=s1,.O=O1,.I=I1};
    deq_fmt1(&tr1,dq_ref1);
    CHECK(memcmp(dq_load1,dq_ref1,sizeof dq_ref1)==0);

    unlink(path); rmdir(dir);
}

/* ---- Part C: qt_bytes()/qt_scale_bytes() byte-accounting for fmt=7 ----
 *
 * qt_bytes() must not fall through to the fmt=2 (packed int4, O*ceil(I/2)+O*4)
 * default -- for a real fp8 tensor that would undercount the resident byte
 * count by roughly half (an AUTOPIN/RAM-budget-feeding hazard). qt_wire_mmap/
 * qt_unwire_mmap must not hardcode scale_b=O*4 (per-row) either -- wrong for
 * fmt=7's per-128x128-block scale array -- hence the dedicated
 * qt_scale_bytes() helper shared by qt_bytes() and both wire functions so
 * there is exactly one place that knows each format's scale geometry. This
 * test exercises the arithmetic directly (no qt_from_disk/disk I/O, no mlock
 * syscall -- qt_wire_mmap's actual mem_wire() call is environment-dependent
 * (RLIMIT_MEMLOCK) and not what changed; the byte-count formula is) across
 * the shapes already used elsewhere in this file plus a block-edge
 * (non-128-multiple) case. */
static void check_fp8_bytes(int O, int I, const char *tag){
    QT t; memset(&t,0,sizeof t); t.fmt=7; t.O=O; t.I=I; t.gs=0;
    int64_t nblkO=fp8_nblk(O), nblkI=fp8_nblk(I), nblk=nblkO*nblkI;
    int64_t want_total = (int64_t)O*I + nblk*4;
    int64_t want_scale = nblk*4;
    int64_t got_total = qt_bytes(&t);
    int64_t got_scale = qt_scale_bytes(&t);
    if(got_total != want_total)
        printf("FAIL %s: qt_bytes=%lld want=%lld\n", tag, (long long)got_total, (long long)want_total);
    CHECK(got_total == want_total);
    if(got_scale != want_scale)
        printf("FAIL %s: qt_scale_bytes=%lld want=%lld\n", tag, (long long)got_scale, (long long)want_scale);
    CHECK(got_scale == want_scale);
    /* weight_b, as qt_wire_mmap/qt_unwire_mmap now compute it, must land on the exact
     * O*I raw-byte weight region -- not short (partial mlock) or long (mlock past the
     * allocation, undefined behavior) by even one byte. */
    CHECK(got_total - got_scale == (int64_t)O*I);
    /* regression guard: the fmt=2 (packed-nibble) formula must NOT be what fmt=7
     * returns -- confirm the value has actually MOVED off it (catches a silent
     * revert of the fmt==7 branch order/placement, not just a formula typo). For
     * O=1,I=1 the two formulas coincide by coincidence (both give 1+4=5), so that
     * shape is skipped for this particular guard -- the other three shapes below are
     * chosen to avoid the coincidence. */
    int64_t old_wrong = (int64_t)O*((I+1)/2) + (int64_t)O*4;
    if(!(O==1 && I==1)) CHECK(got_total != old_wrong);
}

/* qt_wire_mmap()/qt_unwire_mmap() (colibri.c) both compute weight_b/scale_b via
 * qt_wire_split(t,&weight_b,&scale_b) -- this calls that EXACT shared function
 * directly (no mlock syscall: same reasoning as check_fp8_bytes above, mem_wire's
 * actual RLIMIT_MEMLOCK behavior is environment-dependent and not what changed)
 * so a regression at qt_wire_split() itself -- e.g. reverting to the
 * scale_b=(int64_t)t->O*4 hardcode the maintainer's #528 review found dead-coded
 * behind an unused qt_scale_bytes() (compiling clean only because
 * -Wno-unused-function suppressed the warning that should have caught it) --
 * fails here. Covers every format qt_wire_split's callers can see in practice
 * (fmt=1 per-row unaffected; fmt=4/5 grouped-scale formats qt_scale_bytes'
 * own comment names as previously-broken too; fmt=6 (E8/IQ3, FIX ROUND,
 * audit finding: a FIXED 4-byte tag, not O*4 -- see qt_scale_bytes' own
 * comment for why this is reachable, not dead code); fmt=7 nblk>O, the
 * shape this review round is about). */
static void check_wire_split(int fmt, int O, int I, int gs, const char *tag){
    QT t; memset(&t,0,sizeof t); t.fmt=fmt; t.O=O; t.I=I; t.gs=gs;
    int64_t want_scale = qt_scale_bytes(&t);
    int64_t want_weight = qt_bytes(&t) - want_scale;
    int64_t got_weight=-1, got_scale=-1;
    qt_wire_split(&t,&got_weight,&got_scale);
    if(got_scale != want_scale)
        printf("FAIL %s: qt_wire_split scale_b=%lld want=%lld\n", tag, (long long)got_scale, (long long)want_scale);
    CHECK(got_scale == want_scale);
    if(got_weight != want_weight)
        printf("FAIL %s: qt_wire_split weight_b=%lld want=%lld\n", tag, (long long)got_weight, (long long)want_weight);
    CHECK(got_weight == want_weight);
    /* the regression this test exists to catch: a per-row-only scale_b==O*4
     * must NOT be what qt_wire_split returns for a format whose real scale
     * cardinality differs from O (fmt=4/5/6/7 here) -- confirm the value has
     * actually moved off that old constant, not just matched qt_scale_bytes()
     * by coincidence at a degenerate shape. */
    int64_t old_wrong_scale = (int64_t)O*4;
    if((fmt==4 || fmt==5 || fmt==6 || fmt==7) && want_scale != old_wrong_scale)
        CHECK(got_scale != old_wrong_scale);
    /* fmt=6's scale is a FIXED 4 bytes regardless of [O,I] -- assert the
     * literal value directly too, not just "moved off O*4", since a future
     * regression that made it O-dependent in some OTHER wrong way would
     * still pass the generic check above. */
    if(fmt==6) CHECK(want_scale == 4);
}

/* ---- Site-level wire regression (FIX ROUND, validator finding: mutation-
 * proven gap). check_wire_split() above calls qt_wire_split() directly --
 * it would NOT notice a mutation that reverts ONLY qt_wire_mmap's and
 * qt_unwire_mmap's call sites back to the old inline
 * `scale_b=(int64_t)t->O*4` hardcode, leaving qt_wire_split() itself
 * intact (the two sites would simply stop CALLING the now-orphaned-again
 * helper). This test calls the real qt_wire_mmap()/qt_unwire_mmap()
 * functions and asserts, through the mlock()/munlock() observer seam
 * defined above (right after the #include), that the (addr,len) each site
 * ACTUALLY passed down to the platform lock call matches
 * qt_scale_bytes()/qt_bytes() -- i.e. it observes the call sites' own
 * behavior, not a reimplementation of it. Shape chosen so a reverted site
 * is unmistakably wrong, not coincidentally right: O=2, I=16384 ->
 * nblkO=1, nblkI=128, nblk=128, scale_b=512B; the old hardcode would
 * compute O*4=8B instead, a 64x difference. Proven to bite: see the
 * report's mutation output (the exact single-line revert, applied and
 * reverted, with the resulting FAIL line pasted in). POSIX-only (like this
 * file's fork-based refusal tests): the observer seam only intercepts the
 * `#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)`
 * arm both qt_wire_mmap (via mem_wire) and qt_unwire_mmap take -- Windows
 * takes a different call (compat_mlock/compat_munlock) not intercepted
 * here. */
static void test_wire_site_regression(void){
#ifndef _WIN32
    g_seam_wire_n = 0; g_seam_unwire_n = 0;
    enum { O=2, I=16384 };
    enum { NBLKO = CDIV(O,128), NBLKI = CDIV(I,128), NBLK = NBLKO*NBLKI };
    static uint8_t q7[O*I]; static float s7[NBLK];
    for(int i=0;i<O*I;i++) q7[i]=rndbyte_nonan();
    for(int i=0;i<NBLK;i++) s7[i]=0.01f+0.001f*(float)i;

    QT t; memset(&t,0,sizeof t);
    t.fmt=7; t.O=O; t.I=I; t.gs=0; t.q8=(int8_t*)q7; t.s=s7;

    int64_t want_scale = qt_scale_bytes(&t);
    int64_t want_weight = qt_bytes(&t) - want_scale;
    /* sanity: this shape must actually distinguish the fix from the old bug,
     * or the test below would pass either way and prove nothing. */
    CHECK(want_scale != (int64_t)O*4);

    /* qt_wire_mmap doesn't itself gate on g_mmap/mem_should_wire (its
     * caller, pin_wire, does) -- calling it directly always attempts to
     * wire, which is exactly what this test wants. */
    int64_t wired=0; long failed=0;
    qt_wire_mmap(&t, &wired, &failed);
    if(g_seam_wire_n != 2)
        printf("FAIL wire-site regression: qt_wire_mmap called mlock() %d times, expected 2\n", g_seam_wire_n);
    CHECK(g_seam_wire_n == 2);
    if(g_seam_wire_n >= 1 && (int64_t)g_seam_wire_len[0] != want_weight)
        printf("FAIL wire-site regression: qt_wire_mmap's WEIGHT mlock() call got len=%lld want=%lld\n",
               (long long)g_seam_wire_len[0], (long long)want_weight);
    CHECK(g_seam_wire_n>=1 && (int64_t)g_seam_wire_len[0]==want_weight);
    if(g_seam_wire_n >= 2 && (int64_t)g_seam_wire_len[1] != want_scale)
        printf("FAIL wire-site regression: qt_wire_mmap's SCALE mlock() call got len=%lld want=%lld "
               "(this is exactly what a reverted scale_b=O*4 hardcode breaks)\n",
               (long long)g_seam_wire_len[1], (long long)want_scale);
    CHECK(g_seam_wire_n>=2 && (int64_t)g_seam_wire_len[1]==want_scale);

    /* qt_unwire_mmap early-returns unless g_mmap && mem_should_wire() are
     * both true -- force both on for this call, then restore, so this test
     * doesn't change global state for any test that runs after it. */
    int saved_g_mmap = g_mmap, saved_g_mlock = g_mlock;
    g_mmap = 1; g_mlock = 1;
    qt_unwire_mmap(&t);
    g_mmap = saved_g_mmap; g_mlock = saved_g_mlock;

    if(g_seam_unwire_n != 2)
        printf("FAIL wire-site regression: qt_unwire_mmap called munlock() %d times, expected 2\n", g_seam_unwire_n);
    CHECK(g_seam_unwire_n == 2);
    if(g_seam_unwire_n >= 1 && (int64_t)g_seam_unwire_len[0] != want_weight)
        printf("FAIL wire-site regression: qt_unwire_mmap's WEIGHT munlock() call got len=%lld want=%lld\n",
               (long long)g_seam_unwire_len[0], (long long)want_weight);
    CHECK(g_seam_unwire_n>=1 && (int64_t)g_seam_unwire_len[0]==want_weight);
    if(g_seam_unwire_n >= 2 && (int64_t)g_seam_unwire_len[1] != want_scale)
        printf("FAIL wire-site regression: qt_unwire_mmap's SCALE munlock() call got len=%lld want=%lld "
               "(this is exactly what a reverted scale_b=O*4 hardcode breaks)\n",
               (long long)g_seam_unwire_len[1], (long long)want_scale);
    CHECK(g_seam_unwire_n>=2 && (int64_t)g_seam_unwire_len[1]==want_scale);
#else
    printf("skipped on Windows (no mlock/munlock observer seam): wire-site regression\n");
#endif
}

int main(void){
    test_disambiguation();
    test_fmt6_fp8_collision();
    test_ue8m0_scale_refusal();
    test_loader_seam();
    check_fp8_bytes(2048,6144, "qt_bytes fmt=7 gate/up-shaped O=2048 I=6144 (spec example)");
    check_fp8_bytes(6144,2048, "qt_bytes fmt=7 down-shaped O=6144 I=2048");
    check_fp8_bytes(130,200,   "qt_bytes fmt=7 block edges O,I both non-mult-128");
    check_fp8_bytes(1,1,       "qt_bytes fmt=7 degenerate 1x1");
    check_wire_split(1, 4096,4096, 0,  "qt_wire_split fmt=1 plain int8 (per-row scale, unaffected by the fix)");
    check_wire_split(4, 2048,6144, 64, "qt_wire_split fmt=4 grouped int4 (O*ceil(I/gs) scale, not O*4)");
    check_wire_split(5, 2048,6144, 0,  "qt_wire_split fmt=5 int3-g64 (O*ceil(I/64) scale, not O*4)");
    check_wire_split(6, 2048,6144, 0,  "qt_wire_split fmt=6 E8/IQ3 (FIXED 4-byte tag, not O*4=8192B)");
    check_wire_split(6, 1,1,       0,  "qt_wire_split fmt=6 E8/IQ3 degenerate O=1 (O*4 would coincidentally also be 4 -- exercises the literal-4 assert, not just the moved-off-O*4 one)");
    check_wire_split(7, 2,16384, 0,    "qt_wire_split fmt=7 nblk(128) >> O(2): scale=512B, NOT O*4=8B");
    check_wire_split(7, 2048,6144, 0,  "qt_wire_split fmt=7 spec example: scale=3072B, NOT O*4=8192B");
    test_wire_site_regression();
    if(fails){ printf("fp8 loader-seam tests: %d FAILED\n", fails); return 1; }
    printf("fp8 loader-seam tests: ok\n");
    return 0;
}
