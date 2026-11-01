// leakchk: link-time memcheck for xtag (-DXTAG_LEAKCHECK=ON builds xtag_leak). ld --wrap routes every acquire /
// release xtag makes (CRT heap, stdio, Win32 handles + views, CUDA memory / events / streams / modules / graphs /
// contexts, cuBLAS and cuBLASLt objects) through this file, which keeps the live set in a hash table and, after main
// returns, prints whatever is still held, grouped by call site, plus any release of something never acquired.
// Call sites are printed as RVAs: addr2line -f -e build/xtag_leak.exe <ImageBase + rva>.
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define LK_BITS 20
#define LK_N (1u << LK_BITS)
#define LK_M (LK_N - 1)
#define LK_H(k, kd) ((uint32_t)((((uint64_t)(k) ^ ((uint64_t)(kd) << 58)) * 0x9E3779B97F4A7C15ull) >> (64 - LK_BITS)))
#define RA __builtin_return_address(0)

enum { K_HEAP = 1, K_STDIO, K_HANDLE, K_VIEW, K_FIND, K_CUMEM, K_CUHOST, K_CUREG, K_CUEVT, K_CUSTREAM, K_CUMOD,
       K_CUGRAPH, K_CUGEXEC, K_CUCTX, K_CUBLAS, K_LT, K_LTDESC, K_LTPREF, K_LTLAY, K_COUNT };
static const char *lk_kind[K_COUNT] = {"", "heap", "stdio", "handle", "view", "find", "cu mem", "cu host",
                                       "cu hostreg", "cu event", "cu stream", "cu module", "cu graph",
                                       "cu graphexec", "cu ctx", "cublas", "cublasLt", "lt desc", "lt pref",
                                       "lt layout"};

typedef struct {
    uintptr_t key;
    size_t n;
    void *ra;
    const char *api;
    uint32_t kind;
} lk_ent;

typedef struct {
    void *ra;
    const char *api;
    uint32_t kind, cnt;
    size_t n;
} lk_grp;

static lk_ent lk_tab[LK_N];
static SRWLOCK lk_lock = SRWLOCK_INIT;
static uint64_t lk_live, lk_peak, lk_nbad, lk_full, lk_acq;
static lk_ent lk_bad[64];

static void lk_add(uint32_t kind, uintptr_t key, size_t n, const char *api, void *ra)
{
    uint32_t i;

    AcquireSRWLockExclusive(&lk_lock);
    lk_acq++;
    if (lk_live >= LK_N / 2) {
        lk_full++;
        ReleaseSRWLockExclusive(&lk_lock);
        return;
    }
    for (i = LK_H(key, kind); lk_tab[i].kind; i = (i + 1) & LK_M);
    lk_tab[i].key = key, lk_tab[i].n = n, lk_tab[i].ra = ra, lk_tab[i].api = api, lk_tab[i].kind = kind;
    if (++lk_live > lk_peak) lk_peak = lk_live;
    ReleaseSRWLockExclusive(&lk_lock);
}

// linear probing with backward-shift deletion: no tombstones, so per-image churn never degrades the probes
static void lk_del(uint32_t kind, uintptr_t key, const char *api, void *ra)
{
    uint32_t i, j, h;

    AcquireSRWLockExclusive(&lk_lock);
    for (i = LK_H(key, kind); lk_tab[i].kind && (lk_tab[i].kind != kind || lk_tab[i].key != key); i = (i + 1) & LK_M);
    if (!lk_tab[i].kind) {
        if (lk_nbad < 64) lk_bad[lk_nbad].key = key, lk_bad[lk_nbad].ra = ra, lk_bad[lk_nbad].api = api,
                          lk_bad[lk_nbad].kind = kind;
        lk_nbad++;
        ReleaseSRWLockExclusive(&lk_lock);
        return;
    }
    for (j = i;;) {
        j = (j + 1) & LK_M;
        if (!lk_tab[j].kind) break;
        h = LK_H(lk_tab[j].key, lk_tab[j].kind);
        if (((j - h) & LK_M) >= ((j - i) & LK_M)) lk_tab[i] = lk_tab[j], i = j;
    }
    lk_tab[i].kind = 0;
    lk_live--;
    ReleaseSRWLockExclusive(&lk_lock);
}

static void lk_report(void)
{
    static lk_grp g[512];
    uintptr_t base;
    uint32_t i, j, ng;
    size_t bytes;

    base = (uintptr_t)GetModuleHandleA(NULL);
    for (i = ng = 0, bytes = 0; i < LK_N; i++) {
        if (!lk_tab[i].kind) continue;
        for (j = 0; j < ng && (g[j].ra != lk_tab[i].ra || g[j].kind != lk_tab[i].kind || g[j].api != lk_tab[i].api); j++);
        if (j == ng && ng < 512) g[ng].ra = lk_tab[i].ra, g[ng].api = lk_tab[i].api, g[ng].kind = lk_tab[i].kind,
                                 g[ng].cnt = 0, g[ng].n = 0, ng++;
        if (j < ng) g[j].cnt++, g[j].n += lk_tab[i].n;
        bytes += lk_tab[i].n;
    }
    fprintf(stderr, "leakchk: %llu acquired, peak %llu live; at exit %llu live (%zu bytes), %llu bad releases%s\n",
            (unsigned long long)lk_acq, (unsigned long long)lk_peak, (unsigned long long)lk_live, bytes,
            (unsigned long long)lk_nbad, lk_full ? ", TABLE FULL (counts incomplete)" : "");
    for (j = 0; j < ng; j++)
        fprintf(stderr, "  LIVE %-12s %-30s x%-6u %12zu B  rva 0x%llx\n", lk_kind[g[j].kind], g[j].api, g[j].cnt,
                g[j].n, (unsigned long long)((uintptr_t)g[j].ra - base));
    for (j = 0; j < lk_nbad && j < 64; j++)
        fprintf(stderr, "  BAD  %-12s %-30s key 0x%llx  rva 0x%llx\n", lk_kind[lk_bad[j].kind], lk_bad[j].api,
                (unsigned long long)lk_bad[j].key, (unsigned long long)((uintptr_t)lk_bad[j].ra - base));
}

// registered before main: atexit handlers run in reverse, so this one runs after everything xtag registers
__attribute__((constructor)) static void lk_init(void) { atexit(lk_report); }

// ---- CRT heap and stdio
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
char *__real_strdup(const char *);
void __real_free(void *);
FILE *__real_fopen(const char *, const char *);
int __real_fclose(FILE *);

void *__wrap_malloc(size_t n)
{
    void *p;

    p = __real_malloc(n);
    if (p) lk_add(K_HEAP, (uintptr_t)p, n, "malloc", RA);
    return p;
}

void *__wrap_calloc(size_t a, size_t b)
{
    void *p;

    p = __real_calloc(a, b);
    if (p) lk_add(K_HEAP, (uintptr_t)p, a * b, "calloc", RA);
    return p;
}

void *__wrap_realloc(void *p, size_t n)
{
    void *q;

    if (p) lk_del(K_HEAP, (uintptr_t)p, "realloc", RA);
    q = __real_realloc(p, n);
    if (q) lk_add(K_HEAP, (uintptr_t)q, n, "realloc", RA);
    else if (p && n) lk_add(K_HEAP, (uintptr_t)p, 0, "realloc (failed)", RA);
    return q;
}

char *__wrap_strdup(const char *s)
{
    char *p;

    p = __real_strdup(s);
    if (p) lk_add(K_HEAP, (uintptr_t)p, strlen(p) + 1, "strdup", RA);
    return p;
}

void __wrap_free(void *p)
{
    if (p) lk_del(K_HEAP, (uintptr_t)p, "free", RA);
    __real_free(p);
}

FILE *__wrap_fopen(const char *path, const char *mode)
{
    FILE *f;

    f = __real_fopen(path, mode);
    if (f) lk_add(K_STDIO, (uintptr_t)f, 0, "fopen", RA);
    return f;
}

int __wrap_fclose(FILE *f)
{
    if (f) lk_del(K_STDIO, (uintptr_t)f, "fclose", RA);
    return __real_fclose(f);
}

// ---- Win32: these are dllimport calls through __imp_ pointers, so the IAT slot itself is wrapped
#define LK_W32(ret, name, params, args, post)                                                                      \
    typedef ret(WINAPI *lk_t_##name) params;                                                                       \
    extern lk_t_##name __real___imp_##name;                                                                        \
    static ret WINAPI lk_w_##name params                                                                           \
    {                                                                                                              \
        ret r;                                                                                                     \
        r = __real___imp_##name args;                                                                              \
        post;                                                                                                      \
        return r;                                                                                                  \
    }                                                                                                              \
    lk_t_##name __wrap___imp_##name = lk_w_##name;
#define LK_W32REL(name, kind, T)                                                                                   \
    typedef BOOL(WINAPI *lk_t_##name)(T);                                                                          \
    extern lk_t_##name __real___imp_##name;                                                                        \
    static BOOL WINAPI lk_w_##name(T h)                                                                            \
    {                                                                                                              \
        lk_del(kind, (uintptr_t)h, #name, RA);                                                                     \
        return __real___imp_##name(h);                                                                             \
    }                                                                                                              \
    lk_t_##name __wrap___imp_##name = lk_w_##name;

LK_W32(HANDLE, CreateFileA, (LPCSTR a, DWORD b, DWORD c, LPSECURITY_ATTRIBUTES d, DWORD e, DWORD f, HANDLE g),
       (a, b, c, d, e, f, g), if (r != INVALID_HANDLE_VALUE) lk_add(K_HANDLE, (uintptr_t)r, 0, "CreateFileA", RA))
LK_W32(HANDLE, CreateFileMappingA, (HANDLE a, LPSECURITY_ATTRIBUTES b, DWORD c, DWORD d, DWORD e, LPCSTR f),
       (a, b, c, d, e, f), if (r) lk_add(K_HANDLE, (uintptr_t)r, 0, "CreateFileMappingA", RA))
LK_W32(HANDLE, CreateThread, (LPSECURITY_ATTRIBUTES a, SIZE_T b, LPTHREAD_START_ROUTINE c, LPVOID d, DWORD e,
       LPDWORD f), (a, b, c, d, e, f), if (r) lk_add(K_HANDLE, (uintptr_t)r, 0, "CreateThread", RA))
LK_W32(HANDLE, FindFirstFileA, (LPCSTR a, LPWIN32_FIND_DATAA b), (a, b),
       if (r != INVALID_HANDLE_VALUE) lk_add(K_FIND, (uintptr_t)r, 0, "FindFirstFileA", RA))
LK_W32(LPVOID, MapViewOfFile, (HANDLE a, DWORD b, DWORD c, DWORD d, SIZE_T e), (a, b, c, d, e),
       if (r) lk_add(K_VIEW, (uintptr_t)r, e, "MapViewOfFile", RA))
LK_W32REL(CloseHandle, K_HANDLE, HANDLE)
LK_W32REL(FindClose, K_FIND, HANDLE)
LK_W32REL(UnmapViewOfFile, K_VIEW, LPCVOID)

// ---- CUDA driver, cuBLAS, cuBLASLt: every acquire returns status 0 and writes the object through an out-pointer.
// Opaque handles are pointers and enums are ints, so the prototypes need none of the CUDA headers.
#define LK_OUT(kind, name, params, args, key, n)                                                                   \
    int __real_##name params;                                                                                      \
    int __wrap_##name params                                                                                       \
    {                                                                                                              \
        int r;                                                                                                     \
        r = __real_##name args;                                                                                    \
        if (!r) lk_add(kind, (uintptr_t)(key), n, #name, RA);                                                      \
        return r;                                                                                                  \
    }
// null releases are no-ops, like free(NULL); K_CUCTX is keyed by device ordinal, where 0 is a real device
#define LK_REL(kind, name, T)                                                                                      \
    int __real_##name(T);                                                                                          \
    int __wrap_##name(T h)                                                                                         \
    {                                                                                                              \
        if (h || kind == K_CUCTX) lk_del(kind, (uintptr_t)h, #name, RA);                                           \
        return __real_##name(h);                                                                                   \
    }

LK_OUT(K_CUMEM, cuMemAlloc_v2, (uint64_t *p, size_t n), (p, n), *p, n)
LK_REL(K_CUMEM, cuMemFree_v2, uint64_t)
LK_OUT(K_CUHOST, cuMemAllocHost_v2, (void **p, size_t n), (p, n), *p, n)
LK_REL(K_CUHOST, cuMemFreeHost, void *)
LK_OUT(K_CUREG, cuMemHostRegister_v2, (void *p, size_t n, unsigned f), (p, n, f), p, n)
LK_REL(K_CUREG, cuMemHostUnregister, void *)
LK_OUT(K_CUEVT, cuEventCreate, (void **e, unsigned f), (e, f), *e, 0)
LK_REL(K_CUEVT, cuEventDestroy_v2, void *)
LK_OUT(K_CUSTREAM, cuStreamCreate, (void **s, unsigned f), (s, f), *s, 0)
LK_REL(K_CUSTREAM, cuStreamDestroy_v2, void *)
LK_OUT(K_CUMOD, cuModuleLoadData, (void **m, const void *img), (m, img), *m, 0)
LK_REL(K_CUMOD, cuModuleUnload, void *)
LK_OUT(K_CUGRAPH, cuStreamEndCapture, (void *s, void **g), (s, g), *g, 0)
LK_REL(K_CUGRAPH, cuGraphDestroy, void *)
LK_OUT(K_CUGEXEC, cuGraphInstantiateWithFlags, (void **x, void *g, unsigned long long f), (x, g, f), *x, 0)
LK_REL(K_CUGEXEC, cuGraphExecDestroy, void *)
LK_OUT(K_CUCTX, cuDevicePrimaryCtxRetain, (void **c, int d), (c, d), d, 0)
LK_REL(K_CUCTX, cuDevicePrimaryCtxRelease_v2, int)
LK_OUT(K_CUBLAS, cublasCreate_v2, (void **h), (h), *h, 0)
LK_REL(K_CUBLAS, cublasDestroy_v2, void *)
LK_OUT(K_LT, cublasLtCreate, (void **h), (h), *h, 0)
LK_REL(K_LT, cublasLtDestroy, void *)
LK_OUT(K_LTDESC, cublasLtMatmulDescCreate, (void **d, int c, int t), (d, c, t), *d, 0)
LK_REL(K_LTDESC, cublasLtMatmulDescDestroy, void *)
LK_OUT(K_LTPREF, cublasLtMatmulPreferenceCreate, (void **p), (p), *p, 0)
LK_REL(K_LTPREF, cublasLtMatmulPreferenceDestroy, void *)
LK_OUT(K_LTLAY, cublasLtMatrixLayoutCreate, (void **l, int t, uint64_t nr, uint64_t nc, int64_t ld), (l, t, nr, nc, ld),
       *l, 0)
LK_REL(K_LTLAY, cublasLtMatrixLayoutDestroy, void *)
