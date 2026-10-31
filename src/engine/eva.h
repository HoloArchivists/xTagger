#ifndef EVA_H
#define EVA_H
// EVA02-L tagger (wd-eva02-large-tagger-v3) on CUDA. Custom kernels from eva_kernels.cu (cubin embedded at build
// time), fp16 residual stream, fixed batch of EVA_NB model inputs -> one CUDA graph captured once at init. EVA_NBUF
// input/output buffers: the host fills batches ahead while the GPU runs. eva_open returns at once - the CUDA bring-up
// (cold device wake, context, weights, graph) runs on its own thread while the host already prepares batches into
// the buffers, which are page-locked in place once the context exists. eva_ready / eva_poll join it.
//
// Block GEMMs (QKV, out-proj, fc1, fc2): E4M3 x E4M3 on cuBLASLt, fp32 accumulate, per-tensor static scales from
// eva02.bin v3 (smoothing folded in by export_eva.py). Patch embed and head: fp16 cuBLAS.
// -DEVA_CALIB builds the fp16 reference engine (eva02.bin v2) that records per-channel |max| of every block GEMM
// input and writes <weights>.calib on close, which export_eva.py turns into v3.
//
// Input per model input: the patch GEMM operand itself, fp16 [1024 patches][592] (c * 196 + ky * 14 + kx, BGR,
// normalised to [-1, 1], columns 588..591 zero) - procimg writes it straight from the resize.
//
//   eva_open(e, "eva02.bin", device, pool, bytes)  ->  eva_submit(e, s, inputs, n)  ->  eva_wait(e, s)
// Inputs live in the caller's pool (page-locked in place by the bring-up thread) and are DMA'd from there: the host
// writes each model input exactly once, with streaming stores, and never copies it.

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <cuda.h>
#include <cublas_v2.h>
#ifndef EVA_CALIB
#include <cublasLt.h>
#endif

#ifndef EVA_CUBIN
#define EVA_CUBIN "build/eva_kernels.cubin"
#endif
#ifndef EVA_NB
#define EVA_NB 8 // model inputs per launch
#endif
#define EVA_NBUF 8 // host input / output buffers (batches the host may run ahead, also across the bring-up)
#ifdef EVA_CALIB
#define EVA_VER 2
#else
#define EVA_VER 3
#endif

#define EVA_S     448
#define EVA_D     1024
#define EVA_T     1025
#define EVA_NP    1024
#define EVA_HIDP  2736
#define EVA_K0P   592
#define EVA_NL    24
#define EVA_NH    16
#define EVA_NKT   17                                  // 64-query tiles per head
#define EVA_IN    ((size_t)EVA_NP * EVA_K0P)          // halves per model input
#define EVA_M     ((EVA_NB * EVA_T + 15) & ~15)       // token rows, padded to 16
#define EVA_CHUNK (32u << 20)                         // weight streaming chunk
#define EVA_WS    (32u << 20)                         // cuBLAS workspace
#define EVA_WB    (EVA_VER == 3 ? 1 : 2)              // bytes per block-GEMM weight
#define EVA_NG    4                                   // block GEMMs: QKV, O, W12, W2
#define EVA_NS    12                                  // v3 scale table row: wdq x4, adq x4, q, k, v, pad
#define EVA_TP    1088                                // attention keys padded to EVA_NKT * 64
#define EVA_CALN  ((size_t)EVA_NL * (EVA_NG * EVA_HIDP + 3 * EVA_D + EVA_HIDP)) // calibration floats
#define EVA_RT    CUDA_R_16F                          // residual stream type
#define EVA_RB    2

#ifdef EVA_PROF
// Probe (gated, zero footprint otherwise): QPC phase marks, per-op GPU event timing of one direct forward, and the
// host time spent blocked in eva_wait.
static LARGE_INTEGER eva_q0, eva_qf;
static const char *const eva_opname[11] = {"patch+embed", "ln1", "qkv", "stat clear", "flash", "oproj", "ln2", "fc1",
                                           "(mlp ln)", "fc2", "pool+head"};
#define EVA_TSTART() (QueryPerformanceFrequency(&eva_qf), QueryPerformanceCounter(&eva_q0))
#define EVA_TNOW(q) (QueryPerformanceCounter(&(q)), (double)((q).QuadPart - eva_q0.QuadPart) * 1e3 / eva_qf.QuadPart)
#define EVA_TMARK(q, s) fprintf(stderr, "prof %-24s %9.1f ms\n", (s), EVA_TNOW(q))
#define EVA_PT(e, id)                                                                   \
    do {                                                                                \
        if ((e)->pon) {                                                                 \
            cuEventRecord((e)->pev[(e)->pk], (e)->st);                                  \
            (e)->pid[(e)->pk++] = (id);                                                 \
        }                                                                               \
    } while (0)
#else
#define EVA_PT(e, id) ((void)0)
#endif

__asm__(".section .rdata,\"dr\"\n"
        ".balign 64\n"
        ".globl eva_cubin\n"
        "eva_cubin:\n"
        ".incbin \"" EVA_CUBIN "\"\n"
        ".byte 0\n"
        ".text\n");
extern const unsigned char eva_cubin[];

// Weight streaming job (eva_stream): file bytes [256, 256 + bytes) -> device dst, on its own thread and stream.
typedef struct {
    CUcontext ctx;
    const char *path;
    size_t bytes;
    CUdeviceptr dst;
    int status;
} eva_ws;

typedef struct {
    CUdeviceptr ln1w, ln1b, wqkv, wo, ln2w, ln2b, w12, mlnw, mlnb, w2;
    CUdeviceptr rs, cn, dn;                           // v3: h quant multipliers, folded MLP-LN terms
#ifndef EVA_CALIB
    CUdeviceptr sw[EVA_NG], sa[EVA_NG];               // device: weight / activation dequant scales
    cublasLtMatmulDesc_t desc[EVA_NG];
#endif
    float qs[EVA_NG];                                 // host: activation quant multipliers (448 / amax)
    float cq, ck, cv, sinv, vo, a0, a12, a3;          // a0 / a12 / a3: qkv / fc1 / fc2 dequant (weight x act)                           // host: attention E4M3 multipliers, score / output scales
} eva_layer;

typedef struct {
    CUcontext ctx;
    CUdevice dev;
    CUstream st;
    CUmodule mod;
    CUfunction f_embed, f_ln, f_lnh, f_prep, f_flash, f_qkv, f_fc1, f_fc2, f_swiglu, f_pool, f_head, f_amax;
    cublasHandle_t cb;
#ifdef EVA_CALIB
    CUdeviceptr amax;                                 // [EVA_NL][EVA_NG][EVA_HIDP] + [EVA_NL][3 * EVA_D] float bits
#else
    cublasLtHandle_t lt;
    cublasLtMatrixLayout_t la[EVA_NG], lb[EVA_NG], lc[EVA_NG];
    cublasLtMatmulAlgo_t algo[EVA_NG];
#endif
    CUdeviceptr wts, scl, pe, pos, cs, fnw, fnb, hw, hb;
    eva_layer L[EVA_NL];
    CUdeviceptr in, x, h, u, m, q8, k8, v8, stat, hc, pool, pn, logit, out, ws;
    CUgraphExec graph;
    CUevent done[EVA_NBUF];
    void *ipool;                                      // caller's input pool, page-locked at bring-up
    size_t ipoolb;
    float *hout[EVA_NBUF];
    HANDLE boot;                                      // bring-up thread until joined
    float sclh[EVA_NL * EVA_NS];                      // v3 scale table (host copy)
    int devno, bstatus, up;
    const char *path;
    size_t wbytes;
    uint32_t nt, ntp;
    eva_ws wsj;                                       // weight streaming job
#ifdef EVA_PROF
    CUevent pev[256];
    uint8_t pid[256];
    uint32_t pk, pon;
    double wait_ms;
#endif
} eva;

static inline int eva_cu_ck(CUresult r, const char *what)
{
    const char *m;

    if (!r) return 0;
    m = "?";
    cuGetErrorString(r, &m);
    fprintf(stderr, "eva: %s: %s\n", what, m);
    return -1;
}

static inline int eva_cb_ck(int s, const char *what)
{
    if (!s) return 0;
    fprintf(stderr, "eva: %s failed (%d)\n", what, s);
    return -1;
}

#define EVA_CU(call) do { if (eva_cu_ck((call), #call)) return -1; } while (0)
#define EVA_CB(call) do { if (eva_cb_ck((int)(call), #call)) return -1; } while (0)
// Kernel launch on the engine stream; args are addresses of the parameter values.
#define EVA_LAUNCH3(e, f, gx, gy, gz, bx, ...)                                          \
    EVA_CU(cuLaunchKernel((e)->f, (gx), (gy), (gz), (bx), 1, 1, 0, (e)->st, (void *[]){__VA_ARGS__}, NULL))
#define EVA_LAUNCH(e, f, gx, gy, bx, ...) EVA_LAUNCH3(e, f, gx, gy, 1, bx, __VA_ARGS__)
// Next weight tensor: 256-byte aligned, as written by export_eva.py.
#define EVA_TAKE(e, off, dst, bytes)                                                    \
    do {                                                                                \
        (off) = ((off) + 255) & ~(size_t)255;                                           \
        (dst) = (e)->wts + (off);                                                       \
        (off) += (bytes);                                                               \
    } while (0)
#define EVA_F32(p, n) ((p) + (size_t)(n) * 4)
// Row-major Y[rows][n] (op)= X[rows][k] * W[n][k]^T as a column-major TN fp16 GEMM.
#define EVA_LIN(e, W, X, n, k, rows, beta, Y, ct)                                       \
    do {                                                                                \
        if (eva_gemm(e, (n), (rows), (k), (W), (k), 0, (X), (k), 0, (beta), (Y), (ct), (n), 0, 1)) return -1; \
    } while (0)
#define EVA_BLK(e, i, g, W, X, n, k, beta, Y, ct)                                       \
    do {                                                                                \
        if (eva_blk(e, i, g, W, X, n, k, beta, Y, ct)) return -1;                       \
    } while (0)

// C = A^T * B, A/B fp16, fp32 accumulate; ct selects the C type (beta 1 adds into the residual).
static inline int eva_gemm(eva *e, int m, int n, int k, CUdeviceptr A, int lda, long long sa, CUdeviceptr B, int ldb,
                           long long sb, float beta, CUdeviceptr C, cudaDataType ct, int ldc, long long sc, int batch)
{
    float one;

    one = 1.0f;
    EVA_CB(cublasGemmStridedBatchedEx(e->cb, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, (const void *)A, CUDA_R_16F, lda,
                                      sa, (const void *)B, CUDA_R_16F, ldb, sb, &beta, (void *)C, ct, ldc, sc, batch,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return 0;
}

// Block GEMM of layer i, kind g: Y[EVA_M][n] (op)= X[EVA_M][k] * W[n][k]^T.
// E4M3 through the layer's descriptor (scales bound in it) and the shape's chosen algorithm; the calibration build
// runs it in fp16 and records X per column first.
static inline int eva_blk(eva *e, uint32_t i, uint32_t g, CUdeviceptr W, CUdeviceptr X, int n, int k, float beta,
                          CUdeviceptr Y, cudaDataType ct)
{
#ifdef EVA_CALIB
    CUdeviceptr a;
    uint32_t kk, rows;

    kk = (uint32_t)k;
    rows = EVA_M;
    a = e->amax + ((size_t)i * EVA_NG + g) * EVA_HIDP * 4;
    EVA_LAUNCH(e, f_amax, (kk / 8 + 127) / 128, 64, 128, &X, &kk, &rows, &a);
    EVA_LIN(e, W, X, n, k, EVA_M, beta, Y, ct);
#else
    float one;

    (void)n; (void)k; (void)ct;
    one = 1.0f;
    EVA_CB(cublasLtMatmul(e->lt, e->L[i].desc[g], &one, (const void *)W, e->la[g], (const void *)X, e->lb[g], &beta,
                          (void *)Y, e->lc[g], (void *)Y, e->lc[g], &e->algo[g], (void *)e->ws, EVA_WS,
                          (cudaStream_t)e->st));
#endif
    return 0;
}

// The whole forward pass for EVA_NB inputs on e->st (captured once into e->graph).
static inline int eva_forward(eva *e)
{
    eva_layer *L;
    uint32_t M, lnb, total, nb, i;
#ifdef EVA_CALIB
    CUdeviceptr a;
    uint32_t kq, kh;

    kq = 3 * EVA_D;
    kh = EVA_HIDP;
#else
    uint32_t rows;

    rows = EVA_NB * EVA_T;
#endif

    M = EVA_M;
    lnb = M >> 3;
    nb = EVA_NB;
    total = EVA_NB * e->nt;
    if (eva_gemm(e, EVA_D, EVA_NP, EVA_K0P, e->pe, EVA_K0P, 0, e->in, EVA_K0P, (long long)EVA_IN, 0.0f,
                 e->x + (size_t)EVA_D * EVA_RB, EVA_RT, EVA_D, (long long)EVA_T * EVA_D, EVA_NB)) return -1;
    EVA_LAUNCH(e, f_embed, EVA_NB * EVA_T, 1, 256, &e->x, &e->pos);
    EVA_PT(e, 0);
    for (i = 0; i < EVA_NL; i++) {
        L = &e->L[i];
        EVA_LAUNCH(e, f_ln, lnb, 1, 256, &e->x, &L->ln1w, &L->ln1b, &e->h, &M, &L->qs[0]);
        EVA_PT(e, 1);
#ifdef EVA_CALIB
        EVA_BLK(e, i, 0, L->wqkv, e->h, 3 * EVA_D, EVA_D, 0.0f, e->u, CUDA_R_16F);
        EVA_PT(e, 2);
        EVA_LAUNCH(e, f_prep, EVA_NB * EVA_T, 1, 256, &e->u, &e->cs);
        a = e->amax + ((size_t)EVA_NL * EVA_NG * EVA_HIDP + (size_t)i * 3 * EVA_D) * 4;
        EVA_LAUNCH(e, f_amax, 3 * EVA_D / 8 / 128, 64, 128, &e->u, &kq, &M, &a);
        EVA_LAUNCH3(e, f_flash, EVA_NKT, EVA_NH, EVA_NB, 128, &e->u, &e->h);
        EVA_PT(e, 4);
        EVA_BLK(e, i, 1, L->wo, e->h, EVA_D, EVA_D, 1.0f, e->x, EVA_RT);
        EVA_PT(e, 5);
        EVA_LAUNCH(e, f_ln, lnb, 1, 256, &e->x, &L->ln2w, &L->ln2b, &e->h, &M, &L->qs[2]);
        EVA_PT(e, 6);
        EVA_BLK(e, i, 2, L->w12, e->h, 2 * EVA_HIDP, EVA_D, 0.0f, e->u, CUDA_R_16F);
        EVA_PT(e, 7);
        EVA_LAUNCH(e, f_swiglu, M, 1, 352, &e->u, &L->mlnw, &L->mlnb, &e->m, &e->hc);
        a = e->amax + ((size_t)EVA_NL * EVA_NG * EVA_HIDP + (size_t)EVA_NL * 3 * EVA_D + (size_t)i * EVA_HIDP) * 4;
        EVA_LAUNCH(e, f_amax, (EVA_HIDP / 8 + 127) / 128, 64, 128, &e->hc, &kh, &M, &a);
        EVA_PT(e, 8);
        EVA_BLK(e, i, 3, L->w2, e->m, EVA_D, EVA_HIDP, 1.0f, e->x, EVA_RT);
        EVA_PT(e, 9);
#else
        EVA_LAUNCH(e, f_qkv, 3 * EVA_D / 128, (rows + 127) / 128, 128, &e->h, &L->wqkv, &e->q8, &e->k8, &e->v8, &e->cs,
                   &L->a0, &L->cq, &L->ck, &L->cv, &rows);
        EVA_PT(e, 2);
        EVA_LAUNCH3(e, f_flash, EVA_NKT, EVA_NH, EVA_NB, 128, &e->q8, &e->k8, &e->v8, &e->h, &L->sinv, &L->vo);
        EVA_PT(e, 4);
        EVA_BLK(e, i, 1, L->wo, e->h, EVA_D, EVA_D, 1.0f, e->x, EVA_RT);
        EVA_PT(e, 5);
        EVA_LAUNCH(e, f_ln, lnb, 1, 256, &e->x, &L->ln2w, &L->ln2b, &e->h, &M, &L->qs[2]);
        EVA_PT(e, 6);
        EVA_CU(cuMemsetD32Async(e->stat, 0, (size_t)M * 2, e->st));
        EVA_PT(e, 3);
        EVA_LAUNCH(e, f_fc1, (M + 127) / 128, 2 * EVA_HIDP / 96, 128, &e->h, &L->w12, &e->m, &e->stat, &L->rs, &L->a12,
                   &M);
        EVA_PT(e, 7);
        EVA_LAUNCH(e, f_fc2, EVA_D / 128, (M + 127) / 128, 128, &e->m, &L->w2, &e->x, &e->stat, &L->cn, &L->dn, &L->a3,
                   &M);
        EVA_PT(e, 9);
#endif
    }
    EVA_LAUNCH(e, f_pool, EVA_NB, 16, 256, &e->x, &e->pool);
    EVA_LAUNCH(e, f_lnh, (EVA_NB + 7) >> 3, 1, 256, &e->pool, &e->fnw, &e->fnb, &e->pn, &nb);
    EVA_LIN(e, e->hw, e->pn, e->ntp, EVA_D, EVA_NB, 0.0f, e->logit, CUDA_R_32F);
    EVA_LAUNCH(e, f_head, (total + 255) >> 8, 1, 256, &e->logit, &e->hb, &e->out, &e->nt, &e->ntp, &total);
    EVA_PT(e, 10);
    return 0;
}

#ifdef EVA_PROF
// One direct (non-graph) forward with an event after every op, summed per op kind.
static inline int eva_profile(eva *e)
{
    double ms[11], tot;
    float dt;
    uint32_t i;

    for (i = 0; i < 256; i++) EVA_CU(cuEventCreate(&e->pev[i], 0));
    if (eva_forward(e) || eva_forward(e)) return -1; // warm
    e->pon = 1;
    e->pk = 1;
    e->pid[0] = 255;
    EVA_CU(cuEventRecord(e->pev[0], e->st));
    if (eva_forward(e)) return -1;
    EVA_CU(cuStreamSynchronize(e->st));
    e->pon = 0;
    memset(ms, 0, sizeof(ms));
    for (i = 1, tot = 0; i < e->pk; i++) {
        cuEventElapsedTime(&dt, e->pev[i - 1], e->pev[i]);
        ms[e->pid[i]] += dt;
        tot += dt;
    }
    for (i = 0; i < 11; i++) fprintf(stderr, "prof gpu %-12s %8.2f ms  %5.1f%%\n", eva_opname[i], ms[i], 100 * ms[i] / tot);
    fprintf(stderr, "prof gpu batch of %d     %8.2f ms (direct launches)\n", EVA_NB, tot);
    return 0;
}
#endif

// Weight streaming thread: file -> two pinned chunks -> device, disk reads overlapping PCIe copies. Runs while the
// main thread brings up cuBLAS and captures the graph (which only needs the device addresses).
static DWORD WINAPI eva_stream(void *arg)
{
    eva_ws *e;
    FILE *f;
    void *buf[2];
    CUstream ls;
    CUevent ev[2];
    size_t off, n;
    uint32_t i;

    e = arg;
    e->status = -1;
    if (cuCtxSetCurrent(e->ctx) || cuStreamCreate(&ls, CU_STREAM_NON_BLOCKING)) return 1;
    f = fopen(e->path, "rb");
    if (!f) return 1;
    setvbuf(f, NULL, _IONBF, 0);
    fseek(f, 256, SEEK_SET);
    for (i = 0; i < 2; i++)
        if (cuMemAllocHost(&buf[i], EVA_CHUNK) || cuEventCreate(&ev[i], CU_EVENT_DISABLE_TIMING)) return 1;
    for (off = 0, i = 0; off < e->bytes; off += n, i ^= 1) {
        n = e->bytes - off < EVA_CHUNK ? e->bytes - off : EVA_CHUNK;
        if (cuEventSynchronize(ev[i]) || fread(buf[i], 1, n, f) != n) break;
        if (cuMemcpyHtoDAsync(e->dst + off, buf[i], n, ls) || cuEventRecord(ev[i], ls)) break;
    }
    fclose(f);
    if (off >= e->bytes && !cuStreamSynchronize(ls)) e->status = 0;
    cuStreamSynchronize(ls);
    for (i = 0; i < 2; i++) {
        cuEventDestroy(ev[i]);
        cuMemFreeHost(buf[i]);
    }
    cuStreamDestroy(ls);
    return 0;
}

// Device addresses of every tensor in the blob (layout fixed by export_eva.py). v3 starts with the scale table
// [EVA_NL][weight dequant x4, activation dequant x4, q, k, v dequant, pad], whose host copy gives the multipliers.
static inline int eva_layout(eva *e, const float *scl)
{
    size_t off;
    uint32_t i, g;

    off = 0;
    if (EVA_VER == 3) EVA_TAKE(e, off, e->scl, (size_t)EVA_NL * EVA_NS * 4);
    EVA_TAKE(e, off, e->pe, (size_t)EVA_D * EVA_K0P * 2);
    EVA_TAKE(e, off, e->pos, (size_t)EVA_T * EVA_D * 4);
    EVA_TAKE(e, off, e->cs, (size_t)EVA_NP * 32 * 2 * 4);
    for (i = 0; i < EVA_NL; i++) {
        EVA_TAKE(e, off, e->L[i].ln1w, EVA_D * 4);
        EVA_TAKE(e, off, e->L[i].ln1b, EVA_D * 4);
        EVA_TAKE(e, off, e->L[i].wqkv, (size_t)3 * EVA_D * EVA_D * EVA_WB);
        EVA_TAKE(e, off, e->L[i].wo, (size_t)EVA_D * EVA_D * EVA_WB);
        EVA_TAKE(e, off, e->L[i].ln2w, EVA_D * 4);
        EVA_TAKE(e, off, e->L[i].ln2b, EVA_D * 4);
        EVA_TAKE(e, off, e->L[i].w12, (size_t)2 * EVA_HIDP * EVA_D * EVA_WB);
        if (EVA_VER == 3) {
            EVA_TAKE(e, off, e->L[i].rs, EVA_HIDP * 4);
            EVA_TAKE(e, off, e->L[i].w2, (size_t)EVA_D * EVA_HIDP * EVA_WB);
            EVA_TAKE(e, off, e->L[i].cn, EVA_D * 4);
            EVA_TAKE(e, off, e->L[i].dn, EVA_D * 4);
        } else {
            EVA_TAKE(e, off, e->L[i].mlnw, EVA_HIDP * 4);
            EVA_TAKE(e, off, e->L[i].mlnb, EVA_HIDP * 4);
            EVA_TAKE(e, off, e->L[i].w2, (size_t)EVA_D * EVA_HIDP * EVA_WB);
        }
        for (g = 0; g < EVA_NG; g++) {
            e->L[i].qs[g] = scl ? 1.0f / scl[i * EVA_NS + EVA_NG + g] : 1.0f;
#ifndef EVA_CALIB
            e->L[i].sw[g] = e->scl + ((size_t)i * EVA_NS + g) * 4;
            e->L[i].sa[g] = e->scl + ((size_t)i * EVA_NS + EVA_NG + g) * 4;
#endif
        }
        if (scl) {
            e->L[i].cq = 1.0f / scl[i * EVA_NS + 8];
            e->L[i].ck = 1.0f / scl[i * EVA_NS + 9];
            e->L[i].cv = 1.0f / scl[i * EVA_NS + 10];
            e->L[i].sinv = scl[i * EVA_NS + 8] * scl[i * EVA_NS + 9];
            e->L[i].vo = e->L[i].qs[1] * scl[i * EVA_NS + 10];
            e->L[i].a0 = scl[i * EVA_NS] * scl[i * EVA_NS + EVA_NG];
            e->L[i].a12 = scl[i * EVA_NS + 2] * scl[i * EVA_NS + EVA_NG + 2];
            e->L[i].a3 = scl[i * EVA_NS + 3] * scl[i * EVA_NS + EVA_NG + 3];
        }
    }
    EVA_TAKE(e, off, e->fnw, EVA_D * 4);
    EVA_TAKE(e, off, e->fnb, EVA_D * 4);
    EVA_TAKE(e, off, e->hw, (size_t)e->ntp * EVA_D * 2);
    EVA_TAKE(e, off, e->hb, (size_t)e->ntp * 4);
    if (off != e->wbytes) {
        fprintf(stderr, "eva: weight layout mismatch (%zu vs %zu bytes)\n", off, e->wbytes);
        return -1;
    }
    return 0;
}

#ifndef EVA_CALIB
// Shared layouts and one heuristic algorithm per block-GEMM shape; one descriptor per layer and GEMM (it carries
// that layer's scale pointers).
static inline int eva_lt_init(eva *e)
{
    static const int gn[EVA_NG] = {3 * EVA_D, EVA_D, 2 * EVA_HIDP, EVA_D};
    static const int gk[EVA_NG] = {EVA_D, EVA_D, EVA_D, EVA_HIDP};
    static const cudaDataType gt[EVA_NG] = {CUDA_R_16F, EVA_RT, CUDA_R_16F, EVA_RT};
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t res;
    cublasOperation_t opt, opn;
    size_t ws;
    void *p;
    uint32_t i, g;
    int nres;

    opt = CUBLAS_OP_T;
    opn = CUBLAS_OP_N;
    ws = EVA_WS;
    EVA_CB(cublasLtCreate(&e->lt));
    EVA_CB(cublasLtMatmulPreferenceCreate(&pref));
    EVA_CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    for (g = 1; g < 2; g++) { // out-proj only: QKV, fc1, fc2 run on k_qkv / k_fc1 / k_fc2
        EVA_CB(cublasLtMatrixLayoutCreate(&e->la[g], CUDA_R_8F_E4M3, gk[g], gn[g], gk[g]));
        EVA_CB(cublasLtMatrixLayoutCreate(&e->lb[g], CUDA_R_8F_E4M3, gk[g], EVA_M, gk[g]));
        EVA_CB(cublasLtMatrixLayoutCreate(&e->lc[g], gt[g], gn[g], EVA_M, gn[g]));
        for (i = 0; i < EVA_NL; i++) {
            EVA_CB(cublasLtMatmulDescCreate(&e->L[i].desc[g], CUBLAS_COMPUTE_32F, CUDA_R_32F));
            EVA_CB(cublasLtMatmulDescSetAttribute(e->L[i].desc[g], CUBLASLT_MATMUL_DESC_TRANSA, &opt, sizeof(opt)));
            EVA_CB(cublasLtMatmulDescSetAttribute(e->L[i].desc[g], CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn)));
            p = (void *)e->L[i].sw[g];
            EVA_CB(cublasLtMatmulDescSetAttribute(e->L[i].desc[g], CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p, sizeof(p)));
            p = (void *)e->L[i].sa[g];
            EVA_CB(cublasLtMatmulDescSetAttribute(e->L[i].desc[g], CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &p, sizeof(p)));
        }
        EVA_CB(cublasLtMatmulAlgoGetHeuristic(e->lt, e->L[0].desc[g], e->la[g], e->lb[g], e->lc[g], e->lc[g], pref, 1,
                                              &res, &nres));
        if (nres < 1) {
            fprintf(stderr, "eva: no FP8 GEMM algorithm for shape %d x %d x %d\n", gn[g], EVA_M, gk[g]);
            return -1;
        }
        e->algo[g] = res.algo;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    return 0;
}
#endif

// CUDA bring-up, on the boot thread: device wake, context, weight streaming, kernels, buffers, graph, and the host
// buffers page-locked in place.
static inline int eva_up(eva *e)
{
    HANDLE th;
    CUgraph g;
    uint32_t s;
#ifdef EVA_PROF
    LARGE_INTEGER q;
#endif

    EVA_CU(cuInit(0));
    EVA_CU(cuDeviceGet(&e->dev, e->devno));
    EVA_CU(cuDevicePrimaryCtxRetain(&e->ctx, e->dev));
    EVA_CU(cuCtxSetCurrent(e->ctx));
    EVA_CU(cuMemAlloc(&e->wts, e->wbytes));
#ifdef EVA_PROF
    EVA_TMARK(q, "cuInit + context");
#endif
    e->wsj = (eva_ws){e->ctx, e->path, e->wbytes, e->wts, -1};
    th = CreateThread(NULL, 0, eva_stream, &e->wsj, 0, NULL);
    if (!th) return -1;

    EVA_CU(cuStreamCreate(&e->st, CU_STREAM_NON_BLOCKING));
    EVA_CU(cuModuleLoadData(&e->mod, eva_cubin));
    EVA_CU(cuModuleGetFunction(&e->f_embed, e->mod, "k_embed"));
    EVA_CU(cuModuleGetFunction(&e->f_ln, e->mod, "k_ln"));
    EVA_CU(cuModuleGetFunction(&e->f_lnh, e->mod, "k_lnh"));
    EVA_CU(cuModuleGetFunction(&e->f_flash, e->mod, "k_flash"));
#ifdef EVA_CALIB
    EVA_CU(cuModuleGetFunction(&e->f_prep, e->mod, "k_prep"));
    EVA_CU(cuModuleGetFunction(&e->f_swiglu, e->mod, "k_swiglu_ln"));
#else
    EVA_CU(cuModuleGetFunction(&e->f_qkv, e->mod, "k_qkv"));
    EVA_CU(cuModuleGetFunction(&e->f_fc1, e->mod, "k_fc1"));
    EVA_CU(cuModuleGetFunction(&e->f_fc2, e->mod, "k_fc2"));
#endif
    EVA_CU(cuModuleGetFunction(&e->f_pool, e->mod, "k_pool"));
    EVA_CU(cuModuleGetFunction(&e->f_head, e->mod, "k_head"));
    EVA_CU(cuModuleGetFunction(&e->f_amax, e->mod, "k_amax"));
    if (eva_layout(e, EVA_VER == 3 ? e->sclh : NULL)) return -1;
#ifdef EVA_PROF
    EVA_TMARK(q, "module load");
#endif

    EVA_CU(cuMemAlloc(&e->in, EVA_NB * EVA_IN * 2));
    EVA_CU(cuMemAlloc(&e->x, (size_t)EVA_M * EVA_D * EVA_RB));
    EVA_CU(cuMemAlloc(&e->h, (size_t)EVA_M * EVA_D * 2));
    EVA_CU(cuMemAlloc(&e->u, (size_t)EVA_M * 2 * EVA_HIDP * 2));
    EVA_CU(cuMemAlloc(&e->m, (size_t)EVA_M * EVA_HIDP * 2));
#ifdef EVA_CALIB
    EVA_CU(cuMemAlloc(&e->hc, (size_t)EVA_M * EVA_HIDP * 2));
#else
    // E4M3 attention operands; the padded key rows (t >= T) stay zero forever
    EVA_CU(cuMemAlloc(&e->q8, (size_t)EVA_NB * EVA_NH * EVA_TP * 64));
    EVA_CU(cuMemAlloc(&e->k8, (size_t)EVA_NB * EVA_NH * EVA_TP * 64));
    EVA_CU(cuMemAlloc(&e->v8, (size_t)EVA_NB * EVA_NH * EVA_TP * 64));
    EVA_CU(cuMemsetD8(e->q8, 0, (size_t)EVA_NB * EVA_NH * EVA_TP * 64));
    EVA_CU(cuMemsetD8(e->k8, 0, (size_t)EVA_NB * EVA_NH * EVA_TP * 64));
    EVA_CU(cuMemsetD8(e->v8, 0, (size_t)EVA_NB * EVA_NH * EVA_TP * 64));
    EVA_CU(cuMemAlloc(&e->stat, (size_t)EVA_M * 2 * 4));
#endif
    EVA_CU(cuMemAlloc(&e->pool, (size_t)EVA_NB * EVA_D * 4));
    EVA_CU(cuMemAlloc(&e->pn, (size_t)EVA_NB * EVA_D * 2));
    EVA_CU(cuMemAlloc(&e->logit, (size_t)EVA_NB * e->ntp * 4));
    EVA_CU(cuMemAlloc(&e->out, (size_t)EVA_NB * e->nt * 4));
    EVA_CU(cuMemAlloc(&e->ws, EVA_WS));
    EVA_CU(cuMemsetD8(e->x, 0, (size_t)EVA_M * EVA_D * EVA_RB)); // padding rows stay finite
    EVA_CU(cuMemsetD8(e->h, 0, (size_t)EVA_M * EVA_D * 2)); // flash never writes the padding rows
#ifdef EVA_CALIB
    EVA_CU(cuMemAlloc(&e->amax, EVA_CALN * 4));
    EVA_CU(cuMemsetD8(e->amax, 0, EVA_CALN * 4));
#endif
#ifdef EVA_PROF
    EVA_TMARK(q, "device buffers");
#endif
    for (s = 0; s < EVA_NBUF; s++) {
        EVA_CU(cuMemHostRegister(e->hout[s], ((size_t)EVA_NB * e->nt * 4 + 4095) & ~(size_t)4095, 0));
        EVA_CU(cuEventCreate(&e->done[s], CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC)); // waits sleep, not spin
    }
    EVA_CU(cuMemHostRegister(e->ipool, e->ipoolb, 0));

    EVA_CB(cublasCreate(&e->cb));
    EVA_CB(cublasSetStream(e->cb, (cudaStream_t)e->st));
    EVA_CB(cublasSetWorkspace(e->cb, (void *)e->ws, EVA_WS));
#ifdef EVA_PROF
    EVA_TMARK(q, "pinned + cublasCreate");
#endif
#ifndef EVA_CALIB
    if (eva_lt_init(e)) return -1;
#endif
    EVA_CU(cuStreamBeginCapture(e->st, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    if (eva_forward(e)) return -1;
    EVA_CU(cuStreamEndCapture(e->st, &g));
    EVA_CU(cuGraphInstantiate(&e->graph, g, 0));
    EVA_CU(cuGraphDestroy(g));
    EVA_CU(cuGraphUpload(e->graph, e->st));
#ifdef EVA_PROF
    EVA_TMARK(q, "lt init + graph capture");
#endif

    WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
    if (e->wsj.status) {
        fprintf(stderr, "eva: cannot stream weights from %s\n", e->path);
        return -1;
    }
    EVA_CU(cuStreamSynchronize(e->st));
#ifdef EVA_PROF
    EVA_TMARK(q, "weights resident");
    if (eva_profile(e)) return -1;
    EVA_TMARK(q, "eva_up done (+profile)");
#endif
    return 0;
}

static DWORD WINAPI eva_boot(void *arg)
{
    eva *e;

    e = arg;
    e->bstatus = eva_up(e);
    return 0;
}

// Header and host buffers now; the CUDA bring-up starts on its own thread.
static inline int eva_open(eva *e, const char *weights, int device, void *pool, size_t poolb)
{
    FILE *f;
    uint32_t hdr[4], s;

    memset(e, 0, sizeof(*e));
    f = fopen(weights, "rb");
    if (!f || fread(hdr, 4, 4, f) != 4 || memcmp(hdr, "EVA2", 4) || hdr[1] != EVA_VER
        || (EVA_VER == 3 && (fseek(f, 256, SEEK_SET) || fread(e->sclh, sizeof(e->sclh), 1, f) != 1))) {
        fprintf(stderr, "eva: %s is not an eva02.bin v%d (run export_eva.py%s)\n", weights, EVA_VER,
                EVA_VER == 3 ? " with the calibration file" : "");
        if (f) fclose(f);
        return -1;
    }
    fclose(f);
    e->path = weights;
    e->nt = hdr[2];
    e->ntp = (hdr[2] + 7) & ~7u;
    e->wbytes = hdr[3];
    e->devno = device;
    e->ipool = pool;
    e->ipoolb = poolb;
    for (s = 0; s < EVA_NBUF; s++) {
        e->hout[s] = _mm_malloc(((size_t)EVA_NB * e->nt * 4 + 4095) & ~(size_t)4095, 4096);
        if (!e->hout[s]) return -1;
    }
    e->bstatus = -1;
    e->boot = CreateThread(NULL, 0, eva_boot, e, 0, NULL);
    return e->boot ? 0 : -1;
}

// Join the bring-up (once) and make the context current on the calling thread.
static inline int eva_ready(eva *e)
{
    if (e->up) return 0;
    WaitForSingleObject(e->boot, INFINITE);
    CloseHandle(e->boot);
    e->boot = NULL;
    if (e->bstatus) return -1;
    EVA_CU(cuCtxSetCurrent(e->ctx));
    e->up = 1;
    return 0;
}

// Non-blocking: 1 once the bring-up has finished (then eva_ready returns at once).
static inline int eva_poll(eva *e)
{
    return e->up || WaitForSingleObject(e->boot, 0) == WAIT_OBJECT_0;
}

// Launch n inputs (pool pointers, EVA_IN halves each) as batch buf (the graph always runs EVA_NB; extra slots are
// ignored). Runs of inputs adjacent in the pool (an image's crops) go in one DMA.
static inline int eva_submit(eva *e, uint32_t buf, uint16_t *const *in, uint32_t n)
{
    uint32_t i, j;

    if (eva_ready(e)) return -1;
    for (i = 0; i < n; i = j) {
        for (j = i + 1; j < n && in[j] == in[j - 1] + EVA_IN; j++);
        EVA_CU(cuMemcpyHtoDAsync(e->in + (size_t)i * EVA_IN * 2, in[i], (size_t)(j - i) * EVA_IN * 2, e->st));
    }
    EVA_CU(cuGraphLaunch(e->graph, e->st));
    EVA_CU(cuMemcpyDtoHAsync(e->hout[buf], e->out, (size_t)n * e->nt * 4, e->st));
    EVA_CU(cuEventRecord(e->done[buf], e->st));
    return 0;
}

// Non-blocking: 1 once batch buf is done (or failed: eva_wait then reports it).
static inline int eva_query(eva *e, uint32_t buf)
{
    return cuEventQuery(e->done[buf]) != CUDA_ERROR_NOT_READY;
}

static inline const float *eva_wait(eva *e, uint32_t buf)
{
#ifdef EVA_PROF
    LARGE_INTEGER q;
    double t;

    t = EVA_TNOW(q);
    if (cuEventSynchronize(e->done[buf])) return NULL;
    e->wait_ms += EVA_TNOW(q) - t;
    return e->hout[buf];
#else
    return cuEventSynchronize(e->done[buf]) ? NULL : e->hout[buf];
#endif
}

#ifdef EVA_CALIB
// <weights>.calib: float [EVA_NL][EVA_NG][EVA_HIDP] per-channel |max| of the block GEMM inputs, then
// [EVA_NL][3 * EVA_D] of the rotated q | k | v, then [EVA_NL][EVA_HIDP] of the raw SwiGLU output h.
static inline void eva_calib_dump(eva *e)
{
    FILE *f;
    float *a;
    char name[1024];
    size_t n;

    n = EVA_CALN;
    a = malloc(n * 4);
    snprintf(name, sizeof(name), "%s.calib", e->path);
    if (a && !cuMemcpyDtoH(a, e->amax, n * 4) && (f = fopen(name, "wb"))) {
        fwrite(a, 4, n, f);
        fclose(f);
        fprintf(stderr, "eva: wrote %s\n", name);
    }
    free(a);
}
#endif

static inline void eva_close(eva *e)
{
    uint32_t i;

    if (e->boot) {
        WaitForSingleObject(e->boot, INFINITE);
        CloseHandle(e->boot);
        e->boot = NULL;
        if (e->ctx) cuCtxSetCurrent(e->ctx);
    }
    if (!e->ctx) return;
    if (e->st) cuStreamSynchronize(e->st);
#ifdef EVA_PROF
    fprintf(stderr, "prof host blocked in eva_wait %9.1f ms\n", e->wait_ms);
#endif
#ifdef EVA_CALIB
    eva_calib_dump(e);
    cuMemFree(e->amax);
#else
    for (i = 0; i < EVA_NG; i++) {
        if (e->la[i]) cublasLtMatrixLayoutDestroy(e->la[i]);
        if (e->lb[i]) cublasLtMatrixLayoutDestroy(e->lb[i]);
        if (e->lc[i]) cublasLtMatrixLayoutDestroy(e->lc[i]);
    }
    for (i = 0; i < EVA_NL * EVA_NG; i++)
        if (e->L[i / EVA_NG].desc[i % EVA_NG]) cublasLtMatmulDescDestroy(e->L[i / EVA_NG].desc[i % EVA_NG]);
    if (e->lt) cublasLtDestroy(e->lt);
#endif
    if (e->graph) cuGraphExecDestroy(e->graph);
    if (e->cb) cublasDestroy(e->cb);
    for (i = 0; i < EVA_NBUF; i++) {
        if (e->done[i]) {
            cuEventDestroy(e->done[i]);
            cuMemHostUnregister(e->hout[i]);
        }
        _mm_free(e->hout[i]);
    }
    if (e->done[0]) cuMemHostUnregister(e->ipool);
    cuMemFree(e->in); cuMemFree(e->x); cuMemFree(e->h); cuMemFree(e->u); cuMemFree(e->m); cuMemFree(e->q8); cuMemFree(e->k8); cuMemFree(e->v8);
    cuMemFree(e->stat); cuMemFree(e->hc);
    cuMemFree(e->pool); cuMemFree(e->pn); cuMemFree(e->logit); cuMemFree(e->out); cuMemFree(e->ws); cuMemFree(e->wts);
    if (e->mod) cuModuleUnload(e->mod);
    if (e->st) cuStreamDestroy(e->st);
    cuDevicePrimaryCtxRelease(e->dev);
}

#endif // EVA_H
