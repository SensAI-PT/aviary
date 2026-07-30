/* test_ssd_probe.c -- #379/#386 storage-probe hardening contract.
 *
 * Pins the probe's decision layer, which is pure C by design (colibri.c):
 *   - coli_ssd_cache_parse(): the STRICT .coli_ssd grammar, driven by the
 *     shared vector file tests/fixtures/ssd_cache_vectors.txt so the C reader
 *     and the Python reader (resource_plan.parse_ssd_cache, pinned by
 *     test_resource_plan.py over the SAME vectors) cannot drift apart.
 *   - coli_ssd_cache_format() -> parse round-trip: what the writer emits, the
 *     reader trusts.
 *   - coli_ssd_probe_cold_tiles() + coli_ssd_probe_vetoed(): cold-range
 *     steering and the contamination veto over injected residency vectors --
 *     all-cold, all-warm, mixed, and exactly-at-floor, no mincore, no files.
 * On __APPLE__ it additionally exercises coli_ssd_probe_cached() end-to-end
 * against a scratch dir: v2-with-matching-st_dev is trusted without touching
 * any shard; mismatched st_dev, legacy and garbage caches all fall through to
 * a re-probe; and a freshly written (page-cache-warm) shard is VETOED -- no
 * cache file is written and no bandwidth number escapes.
 * Portable: the pure layer builds and runs on every platform, same as
 * test_cap_precedence.c. */
#define COLI_SSD_PROBE_TEST 1
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int failures = 0;

#define CHECK(cond, ...) do{ if(!(cond)){ fprintf(stderr,"FAIL "); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); failures++; } }while(0)

/* ---- shared grammar vectors ------------------------------------------- */

/* \n \r \t \0 \s \\ unescape; returns payload length (may contain NULs). */
static size_t unescape(const char *in, char *out, size_t outsz){
    size_t n=0;
    for(size_t i=0; in[i] && n+1<outsz; i++){
        char c=in[i];
        if(c=='\\' && in[i+1]){
            i++;
            switch(in[i]){
                case 'n': c='\n'; break;
                case 'r': c='\r'; break;
                case 't': c='\t'; break;
                case '0': c='\0'; break;
                case 's': c=' ';  break;
                case '\\': c='\\'; break;
                default: fprintf(stderr,"FAIL bad escape \\%c in vector file\n",in[i]); failures++; break;
            }
        }
        out[n++]=c;
    }
    return n;
}

static void run_vectors(void){
    FILE *f=fopen("tests/fixtures/ssd_cache_vectors.txt","r");
    if(!f){ perror("tests/fixtures/ssd_cache_vectors.txt"); failures++; return; }
    char line[512];
    int nvec=0;
    while(fgets(line,sizeof(line),f)){
        size_t len=strlen(line);
        if(len && line[len-1]=='\n') line[--len]=0;
        if(!len || line[0]=='#') continue;
        /* fields are tab-separated; the payload is everything after the last
         * expectation field and may itself contain literal spaces */
        char *kind=line;
        char *f1=strchr(line,'\t'); if(f1) *f1++=0;
        char payload[256]; size_t plen=0;
        double want_gbs=0; unsigned long long want_dev=0; int want_kind=0;
        if(!strcmp(kind,"garbage")){
            want_kind=0;
            plen = f1 ? unescape(f1,payload,sizeof(payload)) : 0;
        }else if(!strcmp(kind,"legacy")){
            want_kind=1;
            char *f2=f1?strchr(f1,'\t'):NULL;
            if(!f1||!f2){ CHECK(0,"vector line malformed: %s",kind); continue; }
            *f2++=0;
            want_gbs=strtod(f1,NULL);
            plen=unescape(f2,payload,sizeof(payload));
        }else if(!strcmp(kind,"v2")){
            want_kind=2;
            char *f2=f1?strchr(f1,'\t'):NULL;
            char *f3=f2?strchr(f2+1,'\t'):NULL;
            if(!f1||!f2||!f3){ CHECK(0,"vector line malformed: %s",kind); continue; }
            *f2++=0; *f3++=0;
            want_gbs=strtod(f1,NULL);
            want_dev=strtoull(f2,NULL,10);
            plen=unescape(f3,payload,sizeof(payload));
        }else{
            CHECK(0,"unknown vector kind: %s",kind);
            continue;
        }
        double gbs=0; unsigned long long dev=0;
        int got=coli_ssd_cache_parse(payload,plen,&gbs,&dev);
        CHECK(got==want_kind,"vector %d (%s, %zu bytes): kind %d, want %d",nvec,kind,plen,got,want_kind);
        if(got==want_kind && want_kind>=1)
            CHECK(gbs==want_gbs,"vector %d: gbs %.6f, want %.6f",nvec,gbs,want_gbs);
        if(got==want_kind && want_kind==2)
            CHECK(dev==want_dev,"vector %d: st_dev %llu, want %llu",nvec,dev,want_dev);
        nvec++;
    }
    fclose(f);
    CHECK(nvec>=40,"vector file suspiciously short: %d vectors",nvec);
    fprintf(stderr,"grammar vectors: %d checked\n",nvec);
}

/* ---- writer -> reader round-trip --------------------------------------- */

static void run_roundtrip(void){
    static const double vals[]={0.001,0.5,4.0,8.894,9.742,22.139,999.999};
    static const unsigned long long devs[]={0,1,16777233,18446744073709551615ull};
    for(size_t i=0;i<sizeof(vals)/sizeof(*vals);i++)
        for(size_t j=0;j<sizeof(devs)/sizeof(*devs);j++){
            char buf[COLI_SSD_CACHE_MAX+1];
            int n=coli_ssd_cache_format(buf,sizeof(buf),vals[i],devs[j]);
            CHECK(n>0,"format(%.3f,%llu) failed",vals[i],devs[j]);
            if(n<=0) continue;
            double gbs=0; unsigned long long dev=0;
            int kind=coli_ssd_cache_parse(buf,(size_t)n,&gbs,&dev);
            CHECK(kind==2,"round-trip '%s': kind %d",buf,kind);
            /* the writer rounds to %.3f; the reader must return exactly that */
            CHECK(kind==2 && dev==devs[j] && gbs>0 && gbs<1000
                  && (gbs>vals[i]-0.0005 && gbs<vals[i]+0.0005),
                  "round-trip '%s': gbs %.6f dev %llu (want ~%.3f %llu)",
                  buf,gbs,dev,vals[i],devs[j]);
        }
}

/* ---- steering + veto over injected residency vectors ------------------- */

#define PG 16384u                                   /* Apple Silicon page size */
#define PER (COLI_SSD_PROBE_BLK/PG)                 /* 256 pages per 4MB tile */

static void run_steering(void){
    enum { NT=20 };                                 /* 20 tiles = 80 MB */
    size_t npages=(size_t)NT*PER;
    unsigned char *vec=calloc(npages+PER,1);
    long long tiles[NT+2]; long long cold=0; size_t n;

    /* all-cold: every tile qualifies, offsets are the tile starts, no veto */
    n=coli_ssd_probe_cold_tiles(vec,npages,PG,tiles,&cold);
    CHECK(n==NT,"all-cold: %zu tiles, want %d",n,NT);
    CHECK(cold==(long long)NT*COLI_SSD_PROBE_BLK,"all-cold: cold_bytes %lld",cold);
    CHECK(!coli_ssd_probe_vetoed(cold),"all-cold must not veto");
    for(size_t i=0;i<n;i++)
        CHECK(tiles[i]==(long long)i*COLI_SSD_PROBE_BLK,"all-cold: tile %zu at %lld",i,tiles[i]);

    /* all-warm: nothing qualifies -> veto */
    memset(vec,1,npages);
    n=coli_ssd_probe_cold_tiles(vec,npages,PG,tiles,&cold);
    CHECK(n==0 && cold==0,"all-warm: %zu tiles, cold %lld",n,cold);
    CHECK(coli_ssd_probe_vetoed(cold),"all-warm must veto");

    /* mixed: ONE warm page poisons its whole tile (a 4MB read across it would
     * be part RAM); warm page in tile 0 and tile 5 -> 18 qualifying tiles,
     * none of them at offset 0 or 5*4MB */
    memset(vec,0,npages);
    vec[3]=1; vec[5*PER+PER/2]=1;
    n=coli_ssd_probe_cold_tiles(vec,npages,PG,tiles,&cold);
    CHECK(n==NT-2,"mixed: %zu tiles, want %d",n,NT-2);
    for(size_t i=0;i<n;i++)
        CHECK(tiles[i]!=0 && tiles[i]!=5ll*COLI_SSD_PROBE_BLK,"mixed: poisoned tile %lld selected",tiles[i]);
    CHECK(!coli_ssd_probe_vetoed(cold),"mixed with 72MB cold must not veto");

    /* exactly at the floor: 16 cold tiles = 64 MB passes, 15 = 60 MB vetoes */
    memset(vec,1,npages);
    for(size_t t=0;t<16;t++) memset(vec+t*PER,0,PER);
    n=coli_ssd_probe_cold_tiles(vec,npages,PG,tiles,&cold);
    CHECK(n==16 && cold==COLI_SSD_PROBE_COLD_FLOOR,"floor: %zu tiles, cold %lld",n,cold);
    CHECK(!coli_ssd_probe_vetoed(cold),"exactly-at-floor (64MB) must not veto");
    memset(vec+15*PER,1,PER);
    n=coli_ssd_probe_cold_tiles(vec,npages,PG,tiles,&cold);
    CHECK(n==15,"floor-1: %zu tiles",n);
    CHECK(coli_ssd_probe_vetoed(cold),"one tile below the floor must veto");

    /* partial tail tile (npages not a multiple of PER) is never selected:
     * a short window cannot supply a full 4MB read */
    memset(vec,0,npages+PER/2);
    n=coli_ssd_probe_cold_tiles(vec,npages+PER/2,PG,tiles,&cold);
    CHECK(n==NT,"partial tail: %zu tiles, want %d",n,NT);

    /* 4K pages (Intel macs): 1024 pages per tile, same tiling */
    n=coli_ssd_probe_cold_tiles(vec,3*(COLI_SSD_PROBE_BLK/4096),4096,tiles,&cold);
    CHECK(n==3,"4K pages: %zu tiles, want 3",n);

    free(vec);
}

/* ---- end-to-end cache decisions (darwin: real files, no real model) ----- */
#ifdef __APPLE__
static void write_file(const char *dir, const char *name, const char *data){
    char p[1400]; snprintf(p,sizeof(p),"%s/%s",dir,name);
    FILE *f=fopen(p,"w");
    if(!f){ perror(p); failures++; return; }
    fputs(data,f); fclose(f);
}

static int read_file(const char *dir, const char *name, char *out, size_t outsz){
    char p[1400]; snprintf(p,sizeof(p),"%s/%s",dir,name);
    FILE *f=fopen(p,"r");
    if(!f) return -1;
    size_t n=fread(out,1,outsz-1,f); out[n]=0; fclose(f);
    return (int)n;
}

static void run_cached_decisions(void){
    const char *tmp=getenv("TMPDIR");
    char dirbuf[1200];
    snprintf(dirbuf,sizeof(dirbuf),"%s/coli_ssd_probe_test.XXXXXX",
             (tmp && strlen(tmp)<1100) ? tmp : "/tmp");
    if(!mkdtemp(dirbuf)){ perror("mkdtemp"); failures++; return; }
    struct stat st;
    if(stat(dirbuf,&st)!=0){ failures++; return; }
    unsigned long long dev=(unsigned long long)st.st_dev;
    char cache[128], back[160];

    /* v2 + matching st_dev: trusted as-is -- no shard is even needed */
    snprintf(cache,sizeof(cache),"v2 7.500 %llu\n",dev);
    write_file(dirbuf,".coli_ssd",cache);
    double got=coli_ssd_probe_cached(dirbuf);
    CHECK(got==7.5,"matching v2 cache: got %.3f, want 7.500",got);

    /* v2 from another volume: NOT trusted -> re-probe (no shard here, so the
     * probe cannot run and reports -1); the stale cache file is left in place */
    snprintf(cache,sizeof(cache),"v2 7.500 %llu\n",dev+1);
    write_file(dirbuf,".coli_ssd",cache);
    got=coli_ssd_probe_cached(dirbuf);
    CHECK(got==-1,"foreign-volume v2 cache must re-probe: got %.3f",got);

    /* legacy bare number: NOT trusted -> re-probe (the upgrade-to-v2 write
     * happens only when the re-probe succeeds; grammar + round-trip above pin
     * the upgraded format) */
    write_file(dirbuf,".coli_ssd","9.500\n");
    got=coli_ssd_probe_cached(dirbuf);
    CHECK(got==-1,"legacy cache must re-probe: got %.3f",got);

    /* garbage: NOT trusted -> re-probe */
    write_file(dirbuf,".coli_ssd","inf\n");
    got=coli_ssd_probe_cached(dirbuf);
    CHECK(got==-1,"garbage cache must re-probe: got %.3f",got);

    /* contamination veto, end to end: a freshly WRITTEN 80MB shard is page-
     * cache resident, so the probe must refuse to measure it -- and must NOT
     * write any .coli_ssd */
    {
        char p[1400]; snprintf(p,sizeof(p),"%s/out-00000.safetensors",dirbuf);
        FILE *f=fopen(p,"w");
        if(f){
            char *mb=malloc(1024*1024);
            memset(mb,0xA5,1024*1024);
            for(int i=0;i<80;i++) fwrite(mb,1,1024*1024,f);
            free(mb); fclose(f);
            char cp[1400]; snprintf(cp,sizeof(cp),"%s/.coli_ssd",dirbuf);
            unlink(cp);                          /* start with no cache at all */
            int contaminated=0;
            double raw=coli_ssd_probe_raw(dirbuf,&contaminated);
            if(!contaminated){
                /* the OS evicted a fresh 80MB write already? environment too
                 * unusual to assert on -- report and move on, the pure veto
                 * tests above still hold the line */
                fprintf(stderr,"note: fresh shard not resident (raw=%.3f), skipping veto e2e\n",raw);
            }else{
                CHECK(raw==-1,"vetoed raw probe must report -1: got %.3f",raw);
                got=coli_ssd_probe_cached(dirbuf);
                CHECK(got==-1,"vetoed cached probe must report -1: got %.3f",got);
                CHECK(read_file(dirbuf,".coli_ssd",back,sizeof(back))==-1,
                      "vetoed probe must not write .coli_ssd (found: %s)",back);
            }
            unlink(p);
        }
    }
    /* scrub the scratch dir */
    char cp[1400]; snprintf(cp,sizeof(cp),"%s/.coli_ssd",dirbuf);
    unlink(cp); rmdir(dirbuf);
}
#endif /* __APPLE__ */

int main(void){
    run_vectors();
    run_roundtrip();
    run_steering();
#ifdef __APPLE__
    run_cached_decisions();
#endif
    if(failures){
        fprintf(stderr,"ssd probe tests: %d FAILURE(S)\n",failures);
        return 1;
    }
    puts("ssd probe tests: ok");
    return 0;
}
