#ifndef PX_H
#define PX_H
// PixAI tagger (ViTDet-L, 1008^2, 30877 tags) on CUDA: kernels from px_kernels.cu (cubin embedded at build time), the
// E4M3 block GEMM main loop and flash attention shared with eva.h's engine (kern_common.cuh). Fixed batch of PX_NB
// inputs -> one CUDA graph; tokens window-major throughout (see export_pixai.py). Same lifecycle as eva.h: px_open
// returns at once and the CUDA bring-up runs on its own thread; inputs are DMA'd from the caller's page-locked pool.
// -DPX_CALIB builds the fp16 reference engine (pixai_fp16.bin v2): records per-channel |max| of every block GEMM input
// and of rotated q | k | v, plus the GELU block sparsity, and writes <weights>.calib on close.
//
// Input per model input: the patch GEMM operand, fp16 [5184 patches, window-major][592] (RGB c * 196 + ky * 14 + kx,
// normalised to [-1, 1], columns 588..591 zero).
//
//   px_open(p, "pixai.bin", device, pool, bytes)  ->  px_submit(p, s, inputs, n)  ->  px_wait(p, s)

#include "eva.h"

#ifndef PX_CUBIN
#define PX_CUBIN "build/px_kernels.cubin"
#endif
#ifndef PX_NB
#define PX_NB 4 // model inputs per launch (even: rows stay a multiple of 128)
#endif
#ifdef PX_CALIB
#define PX_VER 2
#else
#define PX_VER 3
#endif
#define PX_NBUF   4
#define PX_T      5184
#define PX_LW     576
#define PX_HID    4736
#define PX_NL     32
#define PX_IN     ((size_t)PX_T * EVA_K0P)            // halves per model input
#define PX_M      ((size_t)PX_NB * PX_T)              // token rows
#define PX_WB     (PX_VER == 3 ? 1 : 2)
#define PX_CALN   ((size_t)PX_NL * (4 * PX_HID + 3 * EVA_D)) // calibration floats
#define PX_GLOBAL(i) (((i) & 7) == 7)                 // blocks 7, 15, 23, 31 attend globally
#define PX_STHR   0.01f                               // calibration: |gelu| below this counts as skippable

__asm__(".section .rdata,\"dr\"\n"
        ".balign 64\n"
        ".globl px_cubin\n"
        "px_cubin:\n"
        ".incbin \"" PX_CUBIN "\"\n"
        ".byte 0\n"
        ".text\n");
extern const unsigned char px_cubin[];

typedef struct {
    CUdeviceptr ln1w, ln1b, wqkv, bqkv, wo, bo, ln2w, ln2b, w1, b1, rs, w2, b2;
    float qs[EVA_NG];                                 // activation quant multipliers (448 / amax)
    float cq, ck, cv, sinv, vo, a0, a1, a2, a3;       // attention multipliers; per-GEMM dequant (weight x act)
} px_layer;

typedef struct {
    CUcontext ctx;
    CUdevice dev;
    CUstream st;
    CUmodule mod;
    CUfunction f_embed, f_ln, f_qkv, f_flw, f_flg, f_proj, f_fc1, f_fc2, f_prep, f_addb, f_gelu, f_amax;
    CUfunction f_hln, f_hscore, f_hsoft, f_hv, f_head;
    cublasHandle_t cb;
    CUdeviceptr wts, pe, pos, lnpw, lnpb, csw, csg, hnw, hnb, U, hpos, wv, hw, hb;
    px_layer L[PX_NL];
    CUdeviceptr in, x, h, q8, k8, v8, u, y, s, P, pooled, o, logit, out, ws;
#ifdef PX_CALIB
    CUdeviceptr amax, spar;                           // [32][4][4736] + [32][3072] float bits; [32] counts
#endif
    CUgraphExec graph;
    CUevent done[PX_NBUF];
    float *hout[PX_NBUF];
    void *ipool;
    size_t ipoolb;
    HANDLE boot;
    float sclh[PX_NL * EVA_NS];
    eva_ws wsj;
    const char *path;
    size_t wbytes;
    uint32_t nt, ntp;
    int devno, bstatus, up;
} px;

// C = A^T B with fp16 A, B and fp32 accumulate (column-major cuBLAS view of row-major Y[rows][n] = X[rows][k] W[n][k]^T).
static inline int px_lin(px *p, CUdeviceptr W, CUdeviceptr X, int n, int k, int rows, float beta, CUdeviceptr Y,
                         cudaDataType ct)
{
    float one;

    one = 1.0f;
    EVA_CB(cublasGemmEx(p->cb, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, (const void *)W, CUDA_R_16F, k,
                        (const void *)X, CUDA_R_16F, k, &beta, (void *)Y, ct, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return 0;
}

#ifdef PX_CALIB
// per-column |max| of X [PX_M][k] fp16 into calibration slot a
#define PX_AMAX(p, X, kk, a)                                                            \
    do {                                                                                \
        k32 = (kk);                                                                     \
        av = (a);                                                                       \
        EVA_LAUNCH(p, f_amax, (k32 / 8 + 127) / 128, 64, 128, &(X), &k32, &rows, &av);  \
    } while (0)
#endif

// The forward pass for PX_NB inputs on p->st (captured once into p->graph).
static inline int px_forward(px *p)
{
    px_layer *L;
    CUdeviceptr cs, pv;
    uint32_t rows, lnb, total, nb, i, L1, glob;
    float one, beta, zero;
#ifdef PX_CALIB
    CUdeviceptr av;
    uint32_t k32;
    float sthr;
#endif

    rows = (uint32_t)PX_M;
    lnb = rows >> 3;
    nb = PX_NB;
    total = PX_NB * p->nt;
    one = 1.0f;
    zero = 0.0f;
    if (px_lin(p, p->pe, p->in, EVA_D, EVA_K0P, rows, 0.0f, p->x, EVA_RT)) return -1;
    EVA_LAUNCH(p, f_embed, lnb, 1, 256, &p->x, &p->pos, &p->lnpw, &p->lnpb, &rows);
    for (i = 0; i < PX_NL; i++) {
        L = &p->L[i];
        glob = PX_GLOBAL(i);
        cs = glob ? p->csg : p->csw;
        L1 = glob ? PX_T : PX_LW;
        EVA_LAUNCH(p, f_ln, lnb, 1, 256, &p->x, &L->ln1w, &L->ln1b, &p->h, &rows, &L->qs[0]);
#ifdef PX_CALIB
        sthr = PX_STHR;
        PX_AMAX(p, p->h, EVA_D, p->amax + ((size_t)i * 4 + 0) * PX_HID * 4);
        if (px_lin(p, L->wqkv, p->h, 3 * EVA_D, EVA_D, rows, 0.0f, p->u, CUDA_R_16F)) return -1;
        EVA_LAUNCH(p, f_prep, rows, 1, 384, &p->u, &L->bqkv, &cs, &L1);
        PX_AMAX(p, p->u, 3 * EVA_D, p->amax + ((size_t)PX_NL * 4 * PX_HID + (size_t)i * 3 * EVA_D) * 4);
        if (glob) EVA_LAUNCH3(p, f_flg, PX_T / 64, 16, PX_NB, 128, &p->u, &p->h);
        else EVA_LAUNCH3(p, f_flw, PX_LW / 64, 16, PX_NB * 9, 128, &p->u, &p->h);
        PX_AMAX(p, p->h, EVA_D, p->amax + ((size_t)i * 4 + 1) * PX_HID * 4);
        if (px_lin(p, L->wo, p->h, EVA_D, EVA_D, rows, 1.0f, p->x, EVA_RT)) return -1;
        EVA_LAUNCH(p, f_addb, rows, 1, 256, &p->x, &L->bo);
        EVA_LAUNCH(p, f_ln, lnb, 1, 256, &p->x, &L->ln2w, &L->ln2b, &p->h, &rows, &L->qs[2]);
        PX_AMAX(p, p->h, EVA_D, p->amax + ((size_t)i * 4 + 2) * PX_HID * 4);
        if (px_lin(p, L->w1, p->h, PX_HID, EVA_D, rows, 0.0f, p->u, CUDA_R_16F)) return -1;
        pv = p->spar + (size_t)i * 4;
        EVA_LAUNCH(p, f_gelu, PX_HID / 16, rows / 128, 128, &p->u, &L->b1, &sthr, &pv);
        PX_AMAX(p, p->u, PX_HID, p->amax + ((size_t)i * 4 + 3) * PX_HID * 4);
        if (px_lin(p, L->w2, p->u, EVA_D, PX_HID, rows, 1.0f, p->x, EVA_RT)) return -1;
        EVA_LAUNCH(p, f_addb, rows, 1, 256, &p->x, &L->b2);
#else
        EVA_LAUNCH(p, f_qkv, 24, rows / 128, 128, &p->h, &L->wqkv, &L->bqkv, &p->q8, &p->k8, &p->v8, &cs, &L1, &L->a0,
                   &L->cq, &L->ck, &L->cv, &rows);
        if (glob) EVA_LAUNCH3(p, f_flg, PX_T / 64, 16, PX_NB, 128, &p->q8, &p->k8, &p->v8, &p->h, &L->sinv, &L->vo);
        else EVA_LAUNCH3(p, f_flw, PX_LW / 64, 16, PX_NB * 9, 128, &p->q8, &p->k8, &p->v8, &p->h, &L->sinv, &L->vo);
        EVA_LAUNCH(p, f_proj, EVA_D / 128, rows / 128, 128, &p->h, &L->wo, &p->x, &L->bo, &L->a1, &rows);
        EVA_LAUNCH(p, f_ln, lnb, 1, 256, &p->x, &L->ln2w, &L->ln2b, &p->h, &rows, &L->qs[2]);
        EVA_LAUNCH(p, f_fc1, PX_HID / 128, rows / 128, 128, &p->h, &L->w1, &L->b1, &L->rs, &p->u, &L->a2, &rows);
        EVA_LAUNCH(p, f_fc2, EVA_D / 128, rows / 128, 128, &p->u, &L->w2, &p->x, &L->b2, &L->a3, &rows);
        (void)pv;
#endif
    }
    // attention-pool head: y = LN(x) + pos; scores U y; softmax over tokens; pooled = P y (cuBLAS, per image);
    // o = Wv . pooled per head; logits = Wh' o (+ folded bias in k_head)
    EVA_LAUNCH(p, f_hln, lnb, 1, 256, &p->x, &p->hnw, &p->hnb, &p->hpos, &p->y, &rows);
    EVA_LAUNCH(p, f_hscore, lnb, 1, 256, &p->y, &p->U, &p->s, &rows);
    EVA_LAUNCH(p, f_hsoft, PX_NB * 16, 1, 256, &p->s, &p->P);
    beta = 0.0f;
    EVA_CB(cublasGemmStridedBatchedEx(p->cb, CUBLAS_OP_N, CUBLAS_OP_N, EVA_D, 16, PX_T, &one, (const void *)p->y,
                                      CUDA_R_16F, EVA_D, (long long)PX_T * EVA_D, (const void *)p->P, CUDA_R_16F, PX_T,
                                      (long long)16 * PX_T, &beta, (void *)p->pooled, CUDA_R_32F, EVA_D,
                                      (long long)16 * EVA_D, PX_NB, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    EVA_LAUNCH(p, f_hv, PX_NB * 16, 1, 256, &p->pooled, &p->wv, &p->o);
    if (px_lin(p, p->hw, p->o, (int)p->ntp, EVA_D, PX_NB, zero, p->logit, CUDA_R_32F)) return -1;
    EVA_LAUNCH(p, f_head, (total + 255) >> 8, 1, 256, &p->logit, &p->hb, &p->out, &p->nt, &p->ntp, &total);
    (void)nb;
    return 0;
}

// Device addresses of every tensor in the blob (layout fixed by export_pixai.py); v3 multipliers from the host table.
static inline int px_layout(px *p)
{
    CUdeviceptr tab;
    const float *t;
    size_t off;
    uint32_t i, g;

    off = 0;
    if (PX_VER == 3) EVA_TAKE(p, off, tab, (size_t)PX_NL * EVA_NS * 4);
    EVA_TAKE(p, off, p->pe, (size_t)EVA_D * EVA_K0P * 2);
    EVA_TAKE(p, off, p->pos, (size_t)PX_T * EVA_D * 4);
    EVA_TAKE(p, off, p->lnpw, EVA_D * 4);
    EVA_TAKE(p, off, p->lnpb, EVA_D * 4);
    EVA_TAKE(p, off, p->csw, (size_t)PX_LW * 32 * 2 * 4);
    EVA_TAKE(p, off, p->csg, (size_t)PX_T * 32 * 2 * 4);
    for (i = 0; i < PX_NL; i++) {
        EVA_TAKE(p, off, p->L[i].ln1w, EVA_D * 4);
        EVA_TAKE(p, off, p->L[i].ln1b, EVA_D * 4);
        EVA_TAKE(p, off, p->L[i].wqkv, (size_t)3 * EVA_D * EVA_D * PX_WB);
        EVA_TAKE(p, off, p->L[i].bqkv, 3 * EVA_D * 4);
        EVA_TAKE(p, off, p->L[i].wo, (size_t)EVA_D * EVA_D * PX_WB);
        EVA_TAKE(p, off, p->L[i].bo, EVA_D * 4);
        EVA_TAKE(p, off, p->L[i].ln2w, EVA_D * 4);
        EVA_TAKE(p, off, p->L[i].ln2b, EVA_D * 4);
        EVA_TAKE(p, off, p->L[i].w1, (size_t)PX_HID * EVA_D * PX_WB);
        EVA_TAKE(p, off, p->L[i].b1, PX_HID * 4);
        EVA_TAKE(p, off, p->L[i].rs, PX_HID * 4);
        EVA_TAKE(p, off, p->L[i].w2, (size_t)EVA_D * PX_HID * PX_WB);
        EVA_TAKE(p, off, p->L[i].b2, EVA_D * 4);
        t = p->sclh + (size_t)i * EVA_NS;
        for (g = 0; g < EVA_NG; g++) p->L[i].qs[g] = PX_VER == 3 ? 1.0f / t[EVA_NG + g] : 1.0f;
        if (PX_VER == 3) {
            p->L[i].cq = 1.0f / t[8];
            p->L[i].ck = 1.0f / t[9];
            p->L[i].cv = 1.0f / t[10];
            p->L[i].sinv = t[8] * t[9];
            p->L[i].vo = p->L[i].qs[1] * t[10];
            p->L[i].a0 = t[0] * t[EVA_NG];
            p->L[i].a1 = t[1] * t[EVA_NG + 1];
            p->L[i].a2 = t[2] * t[EVA_NG + 2];
            p->L[i].a3 = t[3] * t[EVA_NG + 3];
        }
    }
    EVA_TAKE(p, off, p->hnw, EVA_D * 4);
    EVA_TAKE(p, off, p->hnb, EVA_D * 4);
    EVA_TAKE(p, off, p->U, (size_t)16 * EVA_D * 4);
    EVA_TAKE(p, off, p->hpos, (size_t)PX_T * EVA_D * 2);
    EVA_TAKE(p, off, p->wv, (size_t)EVA_D * EVA_D * 2);
    EVA_TAKE(p, off, p->hw, (size_t)p->ntp * EVA_D * 2);
    EVA_TAKE(p, off, p->hb, (size_t)p->ntp * 4);
    (void)tab;
    if (off != p->wbytes) {
        fprintf(stderr, "px: weight layout mismatch (%zu vs %zu bytes)\n", off, p->wbytes);
        return -1;
    }
    return 0;
}

// CUDA bring-up on the boot thread (mirrors eva_up).
static inline int px_up(px *p)
{
    HANDLE th;
    CUgraph g;
    uint32_t s;

    EVA_CU(cuInit(0));
    EVA_CU(cuDeviceGet(&p->dev, p->devno));
    EVA_CU(cuDevicePrimaryCtxRetain(&p->ctx, p->dev));
    EVA_CU(cuCtxSetCurrent(p->ctx));
    EVA_CU(cuMemAlloc(&p->wts, p->wbytes));
    p->wsj = (eva_ws){p->ctx, p->path, p->wbytes, p->wts, -1};
    th = CreateThread(NULL, 0, eva_stream, &p->wsj, 0, NULL);
    if (!th) return -1;
    EVA_CU(cuStreamCreate(&p->st, CU_STREAM_NON_BLOCKING));
    EVA_CU(cuModuleLoadData(&p->mod, px_cubin));
    EVA_CU(cuModuleGetFunction(&p->f_embed, p->mod, "k_pembed"));
    EVA_CU(cuModuleGetFunction(&p->f_ln, p->mod, "k_ln"));
    EVA_CU(cuModuleGetFunction(&p->f_flw, p->mod, "k_pflash_w"));
    EVA_CU(cuModuleGetFunction(&p->f_flg, p->mod, "k_pflash_g"));
#ifdef PX_CALIB
    EVA_CU(cuModuleGetFunction(&p->f_prep, p->mod, "k_pprep"));
    EVA_CU(cuModuleGetFunction(&p->f_addb, p->mod, "k_paddb"));
    EVA_CU(cuModuleGetFunction(&p->f_gelu, p->mod, "k_pgelu"));
    EVA_CU(cuModuleGetFunction(&p->f_amax, p->mod, "k_amax"));
#else
    EVA_CU(cuModuleGetFunction(&p->f_qkv, p->mod, "k_pqkv"));
    EVA_CU(cuModuleGetFunction(&p->f_proj, p->mod, "k_pproj"));
    EVA_CU(cuModuleGetFunction(&p->f_fc1, p->mod, "k_pfc1"));
    EVA_CU(cuModuleGetFunction(&p->f_fc2, p->mod, "k_pfc2"));
#endif
    EVA_CU(cuModuleGetFunction(&p->f_hln, p->mod, "k_phln"));
    EVA_CU(cuModuleGetFunction(&p->f_hscore, p->mod, "k_phscore"));
    EVA_CU(cuModuleGetFunction(&p->f_hsoft, p->mod, "k_phsoft"));
    EVA_CU(cuModuleGetFunction(&p->f_hv, p->mod, "k_phv"));
    EVA_CU(cuModuleGetFunction(&p->f_head, p->mod, "k_head"));
    if (px_layout(p)) return -1;

    EVA_CU(cuMemAlloc(&p->in, (size_t)PX_NB * PX_IN * 2));
    EVA_CU(cuMemAlloc(&p->x, PX_M * EVA_D * 2));
    EVA_CU(cuMemAlloc(&p->h, PX_M * EVA_D * 2));
    EVA_CU(cuMemAlloc(&p->y, PX_M * EVA_D * 2));
#ifdef PX_CALIB
    EVA_CU(cuMemAlloc(&p->u, PX_M * PX_HID * 2));
    EVA_CU(cuMemAlloc(&p->amax, PX_CALN * 4));
    EVA_CU(cuMemsetD8(p->amax, 0, PX_CALN * 4));
    EVA_CU(cuMemAlloc(&p->spar, PX_NL * 4));
    EVA_CU(cuMemsetD8(p->spar, 0, PX_NL * 4));
#else
    EVA_CU(cuMemAlloc(&p->u, PX_M * PX_HID));
    EVA_CU(cuMemAlloc(&p->q8, PX_M * EVA_D));
    EVA_CU(cuMemAlloc(&p->k8, PX_M * EVA_D));
    EVA_CU(cuMemAlloc(&p->v8, PX_M * EVA_D));
#endif
    EVA_CU(cuMemAlloc(&p->s, PX_M * 16 * 4));
    EVA_CU(cuMemAlloc(&p->P, (size_t)PX_NB * 16 * PX_T * 2));
    EVA_CU(cuMemAlloc(&p->pooled, (size_t)PX_NB * 16 * EVA_D * 4));
    EVA_CU(cuMemAlloc(&p->o, (size_t)PX_NB * EVA_D * 2));
    EVA_CU(cuMemAlloc(&p->logit, (size_t)PX_NB * p->ntp * 4));
    EVA_CU(cuMemAlloc(&p->out, (size_t)PX_NB * p->nt * 4));
    EVA_CU(cuMemAlloc(&p->ws, EVA_WS));
    for (s = 0; s < PX_NBUF; s++) {
        EVA_CU(cuMemHostRegister(p->hout[s], ((size_t)PX_NB * p->nt * 4 + 4095) & ~(size_t)4095, 0));
        EVA_CU(cuEventCreate(&p->done[s], CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC));
    }
    EVA_CU(cuMemHostRegister(p->ipool, p->ipoolb, 0));
    EVA_CB(cublasCreate(&p->cb));
    EVA_CB(cublasSetStream(p->cb, (cudaStream_t)p->st));
    EVA_CB(cublasSetWorkspace(p->cb, (void *)p->ws, EVA_WS));
    EVA_CU(cuStreamBeginCapture(p->st, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    if (px_forward(p)) return -1;
    EVA_CU(cuStreamEndCapture(p->st, &g));
    EVA_CU(cuGraphInstantiate(&p->graph, g, 0));
    EVA_CU(cuGraphDestroy(g));
    EVA_CU(cuGraphUpload(p->graph, p->st));
    WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
    if (p->wsj.status) {
        fprintf(stderr, "px: cannot stream weights from %s\n", p->path);
        return -1;
    }
    EVA_CU(cuStreamSynchronize(p->st));
    return 0;
}

static DWORD WINAPI px_boot(void *arg)
{
    px *p;

    p = arg;
    p->bstatus = px_up(p);
    return 0;
}

// Header and host buffers now; the CUDA bring-up starts on its own thread.
static inline int px_open(px *p, const char *weights, int device, void *pool, size_t poolb)
{
    FILE *f;
    uint32_t hdr[4], s;

    memset(p, 0, sizeof(*p));
    f = fopen(weights, "rb");
    if (!f || fread(hdr, 4, 4, f) != 4 || memcmp(hdr, "PXAI", 4) || hdr[1] != PX_VER
        || (PX_VER == 3 && (fseek(f, 256, SEEK_SET) || fread(p->sclh, sizeof(p->sclh), 1, f) != 1))) {
        fprintf(stderr, "px: %s is not a pixai.bin v%d (run export_pixai.py%s)\n", weights, PX_VER,
                PX_VER == 3 ? " with the calibration file" : "");
        if (f) fclose(f);
        return -1;
    }
    fclose(f);
    p->path = weights;
    p->nt = hdr[2];
    p->ntp = (hdr[2] + 7) & ~7u;
    p->wbytes = hdr[3];
    p->devno = device;
    p->ipool = pool;
    p->ipoolb = poolb;
    for (s = 0; s < PX_NBUF; s++) {
        p->hout[s] = _mm_malloc(((size_t)PX_NB * p->nt * 4 + 4095) & ~(size_t)4095, 4096);
        if (!p->hout[s]) return -1;
    }
    p->bstatus = -1;
    p->boot = CreateThread(NULL, 0, px_boot, p, 0, NULL);
    return p->boot ? 0 : -1;
}

static inline int px_ready(px *p)
{
    if (p->up) return 0;
    WaitForSingleObject(p->boot, INFINITE);
    CloseHandle(p->boot);
    p->boot = NULL;
    if (p->bstatus) return -1;
    EVA_CU(cuCtxSetCurrent(p->ctx));
    p->up = 1;
    return 0;
}

static inline int px_poll(px *p)
{
    return p->up || WaitForSingleObject(p->boot, 0) == WAIT_OBJECT_0;
}

// Launch n inputs (pool pointers, PX_IN halves each) as batch buf; adjacent pool inputs go in one DMA.
static inline int px_submit(px *p, uint32_t buf, uint16_t *const *in, uint32_t n)
{
    uint32_t i, j;

    if (px_ready(p)) return -1;
    for (i = 0; i < n; i = j) {
        for (j = i + 1; j < n && in[j] == in[j - 1] + PX_IN; j++);
        EVA_CU(cuMemcpyHtoDAsync(p->in + (size_t)i * PX_IN * 2, in[i], (size_t)(j - i) * PX_IN * 2, p->st));
    }
    EVA_CU(cuGraphLaunch(p->graph, p->st));
    EVA_CU(cuMemcpyDtoHAsync(p->hout[buf], p->out, (size_t)n * p->nt * 4, p->st));
    EVA_CU(cuEventRecord(p->done[buf], p->st));
    return 0;
}

static inline int px_query(px *p, uint32_t buf)
{
    return cuEventQuery(p->done[buf]) != CUDA_ERROR_NOT_READY;
}

static inline const float *px_wait(px *p, uint32_t buf)
{
    return cuEventSynchronize(p->done[buf]) ? NULL : p->hout[buf];
}

#ifdef PX_CALIB
// <weights>.calib: float [32][4][4736] per-channel |max| of the block GEMM inputs, then [32][3072] of the rotated
// q | k | v. Also reports, per block, the share of (128-token, 16-unit) GELU blocks that are entirely below PX_STHR.
static inline void px_calib_dump(px *p, uint32_t ninputs)
{
    FILE *f;
    float *a;
    char name[1024];
    uint32_t sp[PX_NL], i;
    double groups;

    a = malloc(PX_CALN * 4);
    snprintf(name, sizeof(name), "%s.calib", p->path);
    if (a && !cuMemcpyDtoH(a, p->amax, PX_CALN * 4) && (f = fopen(name, "wb"))) {
        fwrite(a, 4, PX_CALN, f);
        fclose(f);
        fprintf(stderr, "px: wrote %s\n", name);
    }
    free(a);
    groups = (double)ninputs / PX_NB * (PX_M / 128) * (PX_HID / 16);
    if (!cuMemcpyDtoH(sp, p->spar, sizeof(sp)) && groups > 0)
        for (i = 0; i < PX_NL; i++)
            fprintf(stderr, "px: block %2u gelu blocks below %.2g: %5.1f%%\n", i, PX_STHR, 100.0 * sp[i] / groups);
}
#endif

static inline void px_close(px *p)
{
    uint32_t i;

    if (p->boot) {
        WaitForSingleObject(p->boot, INFINITE);
        CloseHandle(p->boot);
        p->boot = NULL;
        if (p->ctx) cuCtxSetCurrent(p->ctx);
    }
    if (!p->ctx) return;
    if (p->st) cuStreamSynchronize(p->st);
    if (p->graph) cuGraphExecDestroy(p->graph);
    if (p->cb) cublasDestroy(p->cb);
    for (i = 0; i < PX_NBUF; i++) {
        if (p->done[i]) {
            cuEventDestroy(p->done[i]);
            cuMemHostUnregister(p->hout[i]);
        }
        _mm_free(p->hout[i]);
    }
    if (p->done[0]) cuMemHostUnregister(p->ipool);
    cuMemFree(p->in); cuMemFree(p->x); cuMemFree(p->h); cuMemFree(p->y); cuMemFree(p->u);
    cuMemFree(p->q8); cuMemFree(p->k8); cuMemFree(p->v8);
    cuMemFree(p->s); cuMemFree(p->P); cuMemFree(p->pooled); cuMemFree(p->o); cuMemFree(p->logit);
    cuMemFree(p->out); cuMemFree(p->ws); cuMemFree(p->wts);
#ifdef PX_CALIB
    cuMemFree(p->amax); cuMemFree(p->spar);
#endif
    if (p->mod) cuModuleUnload(p->mod);
    if (p->st) cuStreamDestroy(p->st);
    cuDevicePrimaryCtxRelease(p->dev);
}

#endif // PX_H
