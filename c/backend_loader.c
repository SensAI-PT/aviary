/* backend_loader.c — Windows runtime loader for the GPU backend DLL.
 *
 * Why this exists: the engine is built with MinGW-w64 (gcc), but CUDA kernels
 * must be compiled with MSVC + nvcc. We cannot link a CUDA .o into a gcc binary
 * reliably across the MSVC/MinGW ABI, and nvcc requires cl.exe as its host
 * compiler. The clean cross-toolchain split is: build the CUDA backend into a
 * standalone coli_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (glm.exe) never links cudart directly.
 *
 * On Linux this file is not compiled (the Makefile links backend_cuda.o
 * directly). On Windows, when COLI_CUDA is defined, glm.c calls the
 * coli_cuda_* wrappers below, which forward through function pointers resolved
 * from the DLL at first use. If the DLL is absent, every call safely returns
 * the "not initialized" sentinel (0 / no-op) and the engine falls back to CPU.
 *
 * ABI note: ColiCudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, size_t, pointers)
 * agree between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <windows.h>

#include "backend_cuda.h"

/* Which backend DLL this host looks for, and how it labels its own messages.
 * The Makefile defines COLI_HIP_DLL for a HIP_DLL=1 host and leaves it undefined
 * for CUDA_DLL=1; the two builds are mutually exclusive there, so exactly one
 * arm is live. Every filename-coupled site below uses COLI_BACKEND_DLL —
 * including the path bounds check, which must derive from the same constant so
 * it can never drift from the name it is guarding.
 *
 * The EXPORTED symbol prefix stays coli_cuda_ for both vendors: one shared ABI,
 * one shared backend_cuda.cu (see GPU_BACKENDS.md). Only the container filename
 * and the diagnostic label differ. */
#ifdef COLI_HIP_DLL
#define COLI_BACKEND_DLL "coli_hip.dll"
#define COLI_VENDOR_TAG  "[HIP]"
#else
#define COLI_BACKEND_DLL "coli_cuda.dll"
#define COLI_VENDOR_TAG  "[CUDA]"
#endif

/* Function-pointer typedefs matching each exported symbol. */
typedef int            (*fn_init)(const int *devices, int count);
typedef void           (*fn_shutdown)(void);
typedef int            (*fn_device_count)(void);
typedef int            (*fn_device_at)(int index);
typedef int            (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef int            (*fn_device_integrated)(int device);
typedef void           (*fn_stats)(int device, size_t *tensor_count, size_t *tensor_bytes);
typedef void           (*fn_group_stats)(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                         double *h2d_ms, double *kernel_ms, double *d2h_ms);
typedef int            (*fn_expert_mlp)(ColiCudaTensor *gate, ColiCudaTensor *up,
                                        ColiCudaTensor *down, float *y, const float *x, int S);
typedef int            (*fn_expert_group)(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                                          ColiCudaTensor *const *downs, const int *rows, int count,
                                          float *y, const float *x);
typedef int            (*fn_expert_group_issue)(ColiCudaTensor *const *gates,
                                                ColiCudaTensor *const *ups,
                                                ColiCudaTensor *const *downs,
                                                const int *rows, int count, const float *x);
typedef const float *  (*fn_expert_group_take)(int device);
typedef int            (*fn_attention_absorb)(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                              const float *latent, const float *rope, int H, int Q,
                                              int R, int V, int K, int T, float attention_scale);
typedef int            (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                           const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_tensor_upload_g)(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device, int gs);
typedef int            (*fn_e8_set_grid)(const void *grid);
typedef int            (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device, int gs);
typedef void           (*fn_tensor_free)(ColiCudaTensor *tensor);
typedef size_t         (*fn_tensor_bytes)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_device)(const ColiCudaTensor *tensor);

/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */


/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */
typedef int (*fn_attention_absorb_batch)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale);
typedef int (*fn_attention_absorb_batch_dev)(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_absorb_kvdev)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale);
typedef int (*fn_attention_project_batch)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale);
typedef int (*fn_attention_project_ragged)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);
typedef int (*fn_attention_project_batch_dev)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_project_batch_dev_out)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_pipe_add)(int device,float *x_dev,const float *t_dev,size_t n);
typedef void * (*fn_pipe_alloc)(int device,size_t bytes);
typedef int (*fn_pipe_copy2d)(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height);
typedef int (*fn_pipe_download)(int device,const void *src,void *dst,size_t bytes);
typedef void (*fn_pipe_free)(int device,void *p);
typedef int (*fn_pipe_gemm)(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
typedef int (*fn_pipe_peer_copy)(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes);
typedef int (*fn_pipe_rmsnorm)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps);
typedef int (*fn_pipe_rmsnorm_s)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride);
typedef int (*fn_group_resident_issue)(ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,const float *weights,int count,int home_device,const float *x_src_dev,float *partial_slot_dev);
typedef int (*fn_group_resident_take)(int home_device,const int *devices,int n_issued,float *slots_dev,float *acc_dev,int D);
typedef int (*fn_pipe_router)(int device,const float *x_dev,const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,float topp,int norm_topk,float routed_scale,int *idx_host,float *w_host,int *keff_host);
typedef int (*fn_pipe_rope)(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rope_base)(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rows_add)(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D);
typedef float * (*fn_pipe_scratch)(int device,int slot,size_t bytes);
typedef int (*fn_pipe_silu_mul)(int device,float *gate_dev,const float *up_dev,size_t n);
typedef int (*fn_pipe_sync)(int device);
typedef int (*fn_pipe_upload)(int device,void *dst,const void *src,size_t bytes);
typedef int (*fn_shared_mlp_w4a16)(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S);
typedef int (*fn_tensor_update)(ColiCudaTensor *tensor, const void *weights, const float *scales);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_device_integrated device_integrated;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_expert_group_issue expert_group_issue;
    fn_expert_group_take expert_group_take;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_tensor_upload_g tensor_upload_g;
    fn_e8_set_grid     e8_set_grid;
    fn_matmul          matmul;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_ragged attention_project_ragged;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_pipe_add pipe_add;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_group_resident_issue expert_group_resident_issue;
    fn_group_resident_take expert_group_resident_take;
    fn_pipe_router pipe_router;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
} g_cuda;

#ifdef COLI_HIP_DLL
/* ---- COLI_HIP_RUNTIME_DIR: the HIP runtime-directory contract ----
 *
 * A HIP host must be told, explicitly and unambiguously, which ROCm/HIP
 * runtime it is meant to use. This machine class routinely carries more than
 * one amdhip64_7.dll (a system-wide ROCm install plus whatever the user
 * unpacked), and Windows resolves an import by BASE NAME against whatever is
 * already loaded — so "whichever one the search order happens to find" is not
 * a contract, it is a coin flip with a native crash on the losing side.
 *
 * This slice validates the configuration only. It reads the variable, checks
 * that it names a real directory holding a file called amdhip64_7.dll, and
 * then hands over to the existing backend load unchanged. Nothing is loaded,
 * preloaded or pinned here, and no state survives the call: making the load
 * itself deterministic is a separate, self-contained change.
 *
 * Placement matters. The check lives inside coli_cuda_load(), the one-shot
 * loader entry, so it fires exactly when the GPU tier is actually requested.
 * A host that never asks for the GPU never reaches this code and therefore
 * never needs the variable. Under a CUDA_DLL build the whole block is
 * compiled out, so COLI_HIP_RUNTIME_DIR cannot influence CUDA behaviour even
 * when it is set to nonsense.
 *
 * Deliberately not done: no %VAR% expansion (the value is taken literally, so
 * a path containing a percent sign is not silently rewritten), no fallback to
 * the current directory, no PATH search, no process-environment mutation, and
 * no inspection of the runtime file beyond "it exists and is a file". */

#define COLI_HIP_RUNTIME_DIR_VAR L"COLI_HIP_RUNTIME_DIR"
#define COLI_HIP_RUNTIME_FILE    L"amdhip64_7.dll"

static int coli_hip_is_sep(wchar_t c){ return c == L'\\' || c == L'/'; }

/* Absolute in the Windows sense, using only the C runtime — pulling in
 * Shlwapi for PathIsRelativeW would add a link dependency to the host for one
 * predicate. Accepts a drive-absolute path (C:\x, C:/x) and anything rooted at
 * a double separator (\\server\share, \\?\C:\x, //server/share). Everything
 * else is rejected, including the two forms that LOOK rooted but are not:
 * "C:runtime" is relative to the drive's own current directory, and "\runtime"
 * is relative to the current drive. Resolving either against the CWD is
 * precisely the ambiguity this contract exists to remove. */
static int coli_hip_path_is_absolute(const wchar_t *p){
    if(!p || !p[0]) return 0;
    if(coli_hip_is_sep(p[0]) && coli_hip_is_sep(p[1])) return 1;
    if(((p[0] >= L'A' && p[0] <= L'Z') || (p[0] >= L'a' && p[0] <= L'z')) &&
       p[1] == L':' && coli_hip_is_sep(p[2])) return 1;
    return 0;
}

/* stderr is a narrow (byte-oriented) stream here, and mixing wide output into
 * it is undefined, so paths are converted rather than printed with %ls. */
static char *coli_hip_utf8(const wchar_t *w){
    char *s;
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if(n <= 0) return NULL;
    s = (char *)malloc((size_t)n);
    if(!s) return NULL;
    if(WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) <= 0){
        free(s);
        return NULL;
    }
    return s;
}

/* One shape for every rejection: what was wrong, the offending value when
 * there is one, and the same "CPU path remains active" reassurance the
 * backend miss already prints. It never claims a DLL was loaded or tried. */
static void coli_hip_reject(const char *what, const wchar_t *value){
    char *utf8 = value ? coli_hip_utf8(value) : NULL;
    fprintf(stderr, COLI_VENDOR_TAG " %s%s%s; GPU tier disabled "
                    "(CPU path remains active).\n",
            what,
            value ? ": " : "",
            utf8 ? utf8 : (value ? "<unprintable path>" : ""));
    free(utf8);
}

/* Read COLI_HIP_RUNTIME_DIR into a fresh allocation.
 * Returns 1 with *out owned by the caller (possibly the empty string), 0 when
 * the variable is not set at all, and -1 on retrieval or allocation failure
 * with *err carrying the reason. The buffer is sized from the API rather than
 * assumed, so a value longer than MAX_PATH is read in full instead of being
 * truncated into a different path. */
static int coli_hip_env_dup(wchar_t **out, DWORD *err){
    DWORD cap = MAX_PATH;
    int attempt;

    *out = NULL;
    *err = 0;
    for(attempt = 0; attempt < 8; attempt++){
        DWORD n;
        wchar_t *buf = (wchar_t *)malloc((size_t)cap * sizeof(wchar_t));
        if(!buf){ *err = ERROR_NOT_ENOUGH_MEMORY; return -1; }
        /* Cleared first: a set-but-empty value returns 0 with the error code
         * untouched, which is the only way to tell it from "not set". */
        SetLastError(ERROR_SUCCESS);
        n = GetEnvironmentVariableW(COLI_HIP_RUNTIME_DIR_VAR, buf, cap);
        if(n == 0){
            DWORD e = GetLastError();
            if(e == ERROR_ENVVAR_NOT_FOUND){ free(buf); return 0; }
            if(e == ERROR_SUCCESS){ buf[0] = L'\0'; *out = buf; return 1; }
            free(buf);
            *err = e;
            return -1;
        }
        if(n < cap){ *out = buf; return 1; }
        free(buf);
        cap = n;   /* n is now the required size, terminator included */
    }
    /* Only reachable if the value keeps growing between calls. */
    *err = ERROR_MORE_DATA;
    return -1;
}

/* 1 when the configured runtime directory is usable, 0 (with a diagnostic
 * naming the specific problem) otherwise. Validation only — see the block
 * comment above. */
static int coli_hip_runtime_dir_ok(void){
    wchar_t *dir = NULL, *file = NULL;
    DWORD err = 0, attrs;
    size_t len;
    int rc, ok = 0;

    rc = coli_hip_env_dup(&dir, &err);
    if(rc == 0){
        fprintf(stderr, COLI_VENDOR_TAG " COLI_HIP_RUNTIME_DIR is not set; "
                        "set it to the directory containing amdhip64_7.dll. "
                        "GPU tier disabled (CPU path remains active).\n");
        return 0;
    }
    if(rc < 0){
        fprintf(stderr, COLI_VENDOR_TAG " could not read COLI_HIP_RUNTIME_DIR "
                        "(error %lu); GPU tier disabled "
                        "(CPU path remains active).\n", (unsigned long)err);
        return 0;
    }

    len = wcslen(dir);
    if(len == 0){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR is empty", NULL);
        goto done;
    }
    if(!coli_hip_path_is_absolute(dir)){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR must be an absolute path", dir);
        goto done;
    }

    attrs = GetFileAttributesW(dir);
    if(attrs == INVALID_FILE_ATTRIBUTES){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR does not exist", dir);
        goto done;
    }
    if(!(attrs & FILE_ATTRIBUTE_DIRECTORY)){
        coli_hip_reject("COLI_HIP_RUNTIME_DIR is not a directory", dir);
        goto done;
    }

    /* Append the runtime filename under one separator. A value the user ended
     * with a slash must not become a doubled separator: that reads as a UNC
     * root after a drive letter and would fail for a reason unrelated to the
     * real configuration. */
    file = (wchar_t *)malloc((len + 1 + wcslen(COLI_HIP_RUNTIME_FILE) + 1) *
                             sizeof(wchar_t));
    if(!file){
        fprintf(stderr, COLI_VENDOR_TAG " out of memory validating "
                        "COLI_HIP_RUNTIME_DIR; GPU tier disabled "
                        "(CPU path remains active).\n");
        goto done;
    }
    wcscpy(file, dir);
    if(!coli_hip_is_sep(dir[len - 1])) wcscat(file, L"\\");
    wcscat(file, COLI_HIP_RUNTIME_FILE);

    attrs = GetFileAttributesW(file);
    if(attrs == INVALID_FILE_ATTRIBUTES){
        coli_hip_reject("amdhip64_7.dll not found in COLI_HIP_RUNTIME_DIR", file);
        goto done;
    }
    if(attrs & FILE_ATTRIBUTE_DIRECTORY){
        coli_hip_reject("amdhip64_7.dll is a directory, not a file", file);
        goto done;
    }

    ok = 1;

done:
    free(file);
    free(dir);
    return ok;
}
#endif /* COLI_HIP_DLL */

/* Resolve the DLL and all 11 symbols. Returns 1 on success, 0 otherwise.
 * Idempotent: the first call (success or fail) sticks; later calls are no-ops
 * that return the cached result. The engine treats a 0 return as "CUDA
 * unavailable" and falls back to the CPU path without aborting. */
static int coli_cuda_load(void){
    if(g_cuda.loaded) return g_cuda.available;
    g_cuda.loaded = 1;

#ifdef COLI_HIP_DLL
    /* Configuration gate: refuse before touching the filesystem search order,
     * so a misconfigured host fails with a message about its configuration
     * rather than about a DLL it was never going to find. */
    if(!coli_hip_runtime_dir_ok()) return 0;
#endif

    /* Load the backend DLL from the engine's OWN directory, by absolute path —
     * never a bare name. LoadLibraryA(COLI_BACKEND_DLL) searches the current
     * working directory (and, without SafeDllSearchMode, other writable dirs):
     * an attacker who drops a same-named DLL where the user launches glm.exe (or
     * inside a downloaded model directory the user cd's into) would get their
     * DllMain run at load — classic DLL hijacking -> arbitrary code execution.
     * Resolving the path next to glm.exe and loading THAT specific file with
     * LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL and its dependency
     * search to the trusted install directory instead of the CWD. */
    char dllpath[MAX_PATH];
    DWORD mn = GetModuleFileNameA(NULL, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof(COLI_BACKEND_DLL) <= sizeof(dllpath)){
            strcpy(slash + 1, COLI_BACKEND_DLL);
            g_cuda.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!g_cuda.dll){
        /* fallback (GetModuleFileNameA praticamente non fallisce): cerca solo
         * nella dir dell'applicazione e in System32, MAI la CWD. */
        g_cuda.dll = LoadLibraryExA(COLI_BACKEND_DLL, NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if(!g_cuda.dll){
        fprintf(stderr, COLI_VENDOR_TAG " " COLI_BACKEND_DLL " not found; GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name){ \
            fprintf(stderr, COLI_VENDOR_TAG " " COLI_BACKEND_DLL \
                            " missing symbol coli_cuda_" #name "\n"); \
            FreeLibrary(g_cuda.dll); g_cuda.dll=NULL; return 0; }

    /* Optional symbol: a DLL predating it leaves the pointer NULL and only that
     * feature degrades (fmt=6 tensors stay CPU-side), instead of taking the whole
     * GPU backend down over one missing entry point. */
    #define RESOLVE_OPT(name, type) \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop")

    RESOLVE(init,           fn_init)
    RESOLVE(shutdown,       fn_shutdown)
    RESOLVE(device_count,   fn_device_count)
    RESOLVE(device_at,      fn_device_at)
    RESOLVE(mem_info,       fn_mem_info)
    /* Optional: a DLL predating #653 leaves this NULL; the wrapper then reports
     * "not integrated" (0), so the RAM-budget correction simply doesn't apply
     * rather than taking the whole GPU backend down over one missing symbol. */
    RESOLVE_OPT(device_integrated, fn_device_integrated)
    RESOLVE(stats,          fn_stats)
    RESOLVE(group_stats,    fn_group_stats)
    RESOLVE(expert_mlp,     fn_expert_mlp)
    RESOLVE(expert_group,   fn_expert_group)
    RESOLVE(expert_group_issue, fn_expert_group_issue)
    RESOLVE(expert_group_take, fn_expert_group_take)
    RESOLVE(attention_absorb, fn_attention_absorb)
    RESOLVE(tensor_upload,  fn_tensor_upload)
    RESOLVE(tensor_upload_g, fn_tensor_upload_g)
    RESOLVE_OPT(e8_set_grid, fn_e8_set_grid)
    RESOLVE(matmul,         fn_matmul)
    RESOLVE(tensor_free,    fn_tensor_free)
    RESOLVE(tensor_bytes,   fn_tensor_bytes)
    RESOLVE(tensor_device,  fn_tensor_device)

    RESOLVE(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE(attention_project_batch, fn_attention_project_batch)
    RESOLVE(attention_project_ragged, fn_attention_project_ragged)
    RESOLVE(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE(pipe_add, fn_pipe_add)
    RESOLVE(pipe_alloc, fn_pipe_alloc)
    RESOLVE(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE(pipe_download, fn_pipe_download)
    RESOLVE(pipe_free, fn_pipe_free)
    RESOLVE(pipe_gemm, fn_pipe_gemm)
    RESOLVE(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE(expert_group_resident_issue, fn_group_resident_issue)
    RESOLVE(expert_group_resident_take, fn_group_resident_take)
    RESOLVE(pipe_router, fn_pipe_router)
    RESOLVE(pipe_rope, fn_pipe_rope)
    RESOLVE(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE(pipe_scratch, fn_pipe_scratch)
    RESOLVE(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE(pipe_sync, fn_pipe_sync)
    RESOLVE(pipe_upload, fn_pipe_upload)
    RESOLVE(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE(tensor_update, fn_tensor_update)
    #undef RESOLVE

    g_cuda.available = 1;
    return 1;
}

/* ---- Public wrappers: match backend_cuda.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (init returns 0, matmul
 * returns 0 so the caller marks the tensor cuda_failed and uses CPU). ---- */

int coli_cuda_init(const int *devices, int count){
    if(!coli_cuda_load()) return 0;
    return g_cuda.init(devices, count);
}

void coli_cuda_shutdown(void){
    if(g_cuda.available && g_cuda.shutdown) g_cuda.shutdown();
}

int coli_cuda_device_count(void){
    if(!g_cuda.available) return 0;
    return g_cuda.device_count();
}

int coli_cuda_device_at(int index){
    if(!g_cuda.available) return -1;
    return g_cuda.device_at(index);
}

int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_cuda.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_cuda.mem_info(device, free_bytes, total_bytes);
}

int coli_cuda_device_integrated(int device){
    if(!g_cuda.available || !g_cuda.device_integrated) return 0;
    return g_cuda.device_integrated(device);
}

void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_cuda.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_cuda.stats(device, tensor_count, tensor_bytes);
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_mlp(gate, up, down, y, x, S);
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                 ColiCudaTensor *const *ups,
                                 ColiCudaTensor *const *downs,
                                 const int *rows, int count, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group_issue(gates, ups, downs, rows, count, x);
}

const float *coli_cuda_expert_group_take(int device){
    if(!g_cuda.available) return NULL;
    return g_cuda.expert_group_take(device);
}

int coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_cuda.available) return 0;
    return g_cuda.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device, int gs){
    if(!g_cuda.available || !g_cuda.tensor_upload_g){ return 0; }
    return g_cuda.tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs);
}

int coli_cuda_e8_set_grid(const void *grid){
    if(!g_cuda.available || !g_cuda.e8_set_grid) return 0;   /* fmt=6 stays CPU-side */
    return g_cuda.e8_set_grid(grid);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device, gs);
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(g_cuda.available && g_cuda.tensor_free) g_cuda.tensor_free(tensor);
}

size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_bytes(tensor);
}

int coli_cuda_tensor_device(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return -1;
    return g_cuda.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */


/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale){
    if(!coli_cuda_load()) return 0;
    return g_cuda.attention_project_ragged(kv_b,o_proj,out,q,keys,latent,rope,lengths,
        S,H,Q,R,V,K,max_t,attention_scale);
}

int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_add(device, x_dev, t_dev, n);
}

void * coli_cuda_pipe_alloc(int device,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_alloc(device, bytes);
}

int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_download(device, src, dst, bytes);
}

void coli_cuda_pipe_free(int device,void *p){
    if(!g_cuda.available){ return; }
    g_cuda.pipe_free(device, p);
}

int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,const float *weights,int count,int home_device,const float *x_src_dev,float *partial_slot_dev){
    if(!g_cuda.available || !g_cuda.expert_group_resident_issue){ return 0; }
    return g_cuda.expert_group_resident_issue(gates, ups, downs, weights, count, home_device, x_src_dev, partial_slot_dev);
}

int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,float *slots_dev,float *acc_dev,int D){
    if(!g_cuda.available || !g_cuda.expert_group_resident_take){ return 0; }
    return g_cuda.expert_group_resident_take(home_device, devices, n_issued, slots_dev, acc_dev, D);
}

int coli_cuda_pipe_router(int device,const float *x_dev,const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,float topp,int norm_topk,float routed_scale,int *idx_host,float *w_host,int *keff_host){
    if(!g_cuda.available || !g_cuda.pipe_router){ return 0; }
    return g_cuda.pipe_router(device, x_dev, rw_dev, rb_dev, D, E, Ksel, topp, norm_topk, routed_scale, idx_host, w_host, keff_host);
}

int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_scratch(device, slot, bytes);
}

int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_cuda_pipe_sync(int device){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_sync(device);
}

int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload(device, dst, src, bytes);
}

int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!g_cuda.available){ return 0; }
    return g_cuda.tensor_update(tensor, weights, scales);
}

#endif /* _WIN32 */
