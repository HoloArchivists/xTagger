// Kernel pieces shared by the wd-eva02 (eva_kernels.cu) and PixAI (px_kernels.cu) engines: helpers, LayerNorm,
// calibration max, RoPE, the E4M3 block-GEMM main loop and E4M3 / fp16 flash attention templated on the
// sequence length. The including file defines the model shapes first (D, T, QKV, ...).
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifdef EVA_FP8
#define ACTQ 1
#else
#define ACTQ 0
#endif
#define ACT_BYTES(q) ((q) ? 1 : 2)

#define RESCALE 8.0f // k_flash: log2 headroom before the running max must be refreshed
#define PSKIP  10.0f // k_flash (E4M3): p < 2^-10 rounds to zero in E4M3 (half the smallest subnormal, tie to even)
#define EPS  1e-6f


// byte offset of 16-byte chunk c of row r in a 64 x 64 half tile, XOR-swizzled so ldmatrix rows hit distinct banks
#define SWZ(r, c) ((r) * 128 + ((((c) ^ ((r) & 7))) << 4))
#define CP_COMMIT() asm volatile("cp.async.commit_group;\n" ::)
#define CP_WAIT1()  asm volatile("cp.async.wait_group 1;\n" ::)

__device__ __forceinline__ float warp_sum(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ unsigned pack2(float a, float b)
{
    __half2 h;

    h = __floats2half2_rn(a, b);
    return *(unsigned *)&h;
}

__device__ __forceinline__ float tanh_approx(float x)
{
    float y;

    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void cp16(unsigned dst, const void *src, unsigned n)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}

__device__ __forceinline__ void ldsm4(unsigned *r, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ldsm4t(unsigned *r, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void mma16816(float *c, const unsigned *a, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Residual stream element (fp16: halves the residual traffic, accuracy unchanged against fp32). 4 consecutive
// elements in / out as float4; the float overload serves the pooled head input.
typedef __half res_t;

__device__ __forceinline__ float4 ld_res4(const float *p) { return *(const float4 *)p; }

__device__ __forceinline__ float4 ld_res4(const __half *p)
{
    uint2 u;
    float2 a, b;

    u = *(const uint2 *)p;
    a = __half22float2(*(const __half2 *)&u.x);
    b = __half22float2(*(const __half2 *)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ void st_res4(__half *p, float4 v)
{
    *(uint2 *)p = make_uint2(pack2(v.x, v.y), pack2(v.z, v.w));
}

// E4M3 pair, saturating (values were scaled by qs = 448 / calibrated amax)
__device__ __forceinline__ unsigned q2(float a, float b)
{
    return (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

// 4 activations at element i of row p: E4M3 (Q = 1, scaled by qs) or fp16 (Q = 0).
template <int Q>
__device__ __forceinline__ void st4(unsigned char *p, size_t i, float a, float b, float c, float d, float qs)
{
    if (Q) *(unsigned *)(p + i) = q2(a * qs, b * qs) | q2(c * qs, d * qs) << 16;
    else *(uint2 *)(p + 2 * i) = make_uint2(pack2(a, b), pack2(c, d));
}

// 8 fp16 values (one uint4) -> 8 activations at p.
template <int Q>
__device__ __forceinline__ void st8h(unsigned char *p, uint4 v, float qs)
{
    const __half2 *h;
    float2 a, b, c, d;

    if (!Q) {
        *(uint4 *)p = v;
        return;
    }
    h = (const __half2 *)&v;
    a = __half22float2(h[0]);
    b = __half22float2(h[1]);
    c = __half22float2(h[2]);
    d = __half22float2(h[3]);
    *(uint2 *)p = make_uint2(q2(a.x * qs, a.y * qs) | q2(b.x * qs, b.y * qs) << 16,
                             q2(c.x * qs, c.y * qs) | q2(d.x * qs, d.y * qs) << 16);
}

// 64 rows x 64 halfs from g (row stride QKV) starting at token r0 -> swizzled tile; rows >= L are zero-filled.
template <int L>
__device__ __forceinline__ void load_tile(unsigned s, const __half *g, unsigned r0, unsigned tid)
{
    unsigned i, idx, r, c, ok;

    for (i = 0; i < 4; i++) {
        idx = tid + 128 * i;
        r = idx >> 3;
        c = idx & 7;
        ok = r0 + r < L;
        cp16(s + SWZ(r, c), g + (size_t)(ok ? r0 + r : 0) * QKV + c * 8, ok ? 16 : 0);
    }
}

// LayerNorm over 1024 fp32 -> activation. One warp per row, 8 rows per 256-thread block. 128-bit loads,
// 32-bit (E4M3) or 64-bit (fp16) stores. k_ln writes the GEMM activation type, k_lnh always fp16 (head input).
template <int Q, typename R>
__device__ __forceinline__ void ln_row(const R *__restrict__ x, const float *__restrict__ w,
                                       const float *__restrict__ bias, void *__restrict__ y, unsigned rows, float qs)
{
    const R *xr;
    unsigned char *yr;
    float4 v[8], g, bb;
    float s, q, mean, rstd, dx;
    unsigned row, lane, j;

    row = blockIdx.x * 8 + (threadIdx.x >> 5);
    lane = threadIdx.x & 31;
    if (row >= rows) return;
    xr = x + (size_t)row * D;
    for (j = 0, s = 0.f; j < 8; j++) {
        v[j] = ld_res4(xr + 4 * (lane + 32 * j));
        s += v[j].x + v[j].y + v[j].z + v[j].w;
    }
    mean = warp_sum(s) * (1.0f / D);
    for (j = 0, q = 0.f; j < 8; j++) {
        dx = v[j].x - mean; q += dx * dx;
        dx = v[j].y - mean; q += dx * dx;
        dx = v[j].z - mean; q += dx * dx;
        dx = v[j].w - mean; q += dx * dx;
    }
    rstd = rsqrtf(warp_sum(q) * (1.0f / D) + EPS);
    yr = (unsigned char *)y + (size_t)row * D * ACT_BYTES(Q);
    for (j = 0; j < 8; j++) {
        g = ((const float4 *)w)[lane + 32 * j];
        bb = ((const float4 *)bias)[lane + 32 * j];
        st4<Q>(yr, 4 * (lane + 32 * j), (v[j].x - mean) * rstd * g.x + bb.x, (v[j].y - mean) * rstd * g.y + bb.y,
               (v[j].z - mean) * rstd * g.z + bb.z, (v[j].w - mean) * rstd * g.w + bb.w, qs);
    }
}

extern "C" __global__ void k_ln(const res_t *__restrict__ x, const float *__restrict__ w, const float *__restrict__ bias,
                                void *__restrict__ y, unsigned rows, float qs)
{
    ln_row<ACTQ>(x, w, bias, y, rows, qs);
}

extern "C" __global__ void k_lnh(const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ bias,
                                 void *__restrict__ y, unsigned rows)
{
    ln_row<0>(x, w, bias, y, rows, 1.0f);
}

// Calibration: per-column max |x| over rows of an fp16 [rows][K] tensor, atomically into amax (float bits).
// Grid (K / 8 / 128, row groups), one 8-column uint4 per thread.
extern "C" __global__ void k_amax(const __half *__restrict__ x, unsigned K, unsigned rows, unsigned *__restrict__ amax)
{
    uint4 v;
    const __half2 *hv;
    float2 f;
    float m[8];
    unsigned c8, r, j;

    c8 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c8 * 8 >= K) return;
    for (j = 0; j < 8; j++) m[j] = 0.f;
    hv = (const __half2 *)&v;
    for (r = blockIdx.y; r < rows; r += gridDim.y) {
        v = *(const uint4 *)(x + (size_t)r * K + c8 * 8);
        for (j = 0; j < 4; j++) {
            f = __half22float2(hv[j]);
            m[2 * j] = fmaxf(m[2 * j], fabsf(f.x));
            m[2 * j + 1] = fmaxf(m[2 * j + 1], fabsf(f.y));
        }
    }
    for (j = 0; j < 8; j++) atomicMax(amax + c8 * 8 + j, __float_as_uint(m[j]));
}

// Interleaved RoPE of 4 pairs (one uint4 of fp16) with angles (c01, c23) = (cos, sin) x 4, scaled by sc -> o[8].
__device__ __forceinline__ void rot8(uint4 v, float4 c01, float4 c23, float sc, float *o)
{
    const __half2 *h;
    float2 f;

    h = (const __half2 *)&v;
    f = __half22float2(h[0]); o[0] = (f.x * c01.x - f.y * c01.y) * sc; o[1] = (f.y * c01.x + f.x * c01.y) * sc;
    f = __half22float2(h[1]); o[2] = (f.x * c01.z - f.y * c01.w) * sc; o[3] = (f.y * c01.z + f.x * c01.w) * sc;
    f = __half22float2(h[2]); o[4] = (f.x * c23.x - f.y * c23.y) * sc; o[5] = (f.y * c23.x + f.x * c23.y) * sc;
    f = __half22float2(h[3]); o[6] = (f.x * c23.z - f.y * c23.w) * sc; o[7] = (f.y * c23.z + f.x * c23.w) * sc;
}

#ifdef EVA_FP8
// ---- E4M3 block GEMMs and attention on mma.m16n8k32.e4m3 (fp32 accumulate) ----
// Q, K, V carry per-layer static scales cq, ck, cv (448 / calibrated amax), so raw scores are S * cq * ck: the
// softmax runs on raw scores with sinv = 1 / (cq ck) folded into the exp2 FFMA, and 1 / cv into the output scale.
//
// Fragment orders are chosen so that no register ever moves between lanes:
//  - head dims: k_qkv's accumulators hold dim pairs {2tq, 2tq+1} per n8 tile, and the k32 A / B fragments want
//    4 consecutive k per lane. Logical dim p of each 16 is physical dim 2(p>>2) + (p&1) + 8((p>>1)&1) for both
//    Q and K (stored permuted by k_qkv): the dot product is order-free.
//  - keys: S n-tile n, row i of a 64-key tile holds key 16(n>>1) + 4(i>>1) + 2(n&1) + (i&1) (picked by the
//    ldmatrix row addresses), which turns the S accumulators of n-tiles 4kk..4kk+3 into exactly the P A fragment
//    of keys 32kk..32kk+31 in natural order, so V^T is read straight.
#define GS 3    // GEMM cp.async stages
// byte offset of 16-byte chunk c of row r in a 64-byte-row E4M3 tile. K rows are read 8 at a time as
// {0,1,4,5,8,9,12,13} (+2) of a 16-row group, everything else (Q, V^T, GEMM tiles) as 8 consecutive rows: each XOR
// spreads its 8 rows over all 8 bank quads.
#define SWK(r, c) ((r) * 64 + (((c) ^ (((r) >> 2) & 3)) << 4))
#define SWV(r, c) ((r) * 64 + (((c) ^ (((r) >> 1) & 3)) << 4))

// not volatile: a pure register op, free to interleave with the ldmatrix stream
__device__ __forceinline__ void mma16832(float *c, const unsigned *a, unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// fp16 m16n8k16, non-volatile like mma16832. Its A / B fragments sit at the same byte offsets per lane as the E4M3
// k32 ones (a0: row g, bytes 4t..4t+3; a2: row g, bytes 16+4t..; b0 / b1 likewise), so gemm8's byte-addressed
// pipeline runs fp16 operands unchanged with this MMA and K counted in bytes.
__device__ __forceinline__ void mma16816h(float *c, const unsigned *a, unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#define MMAK(H, c, a, b0, b1) ((H) ? mma16816h(c, a, b0, b1) : mma16832(c, a, b0, b1))

// Block GEMM main loop: acc (+)= X[m0 + 32 warp ..][0, K) * W[0, BN)[0, K)^T for this warp's 32 rows x BN columns.
// Block tile 128 x BN, 4 warps, 64-byte k steps through a GS-stage cp.async ring at shared address sb
// (GS * (128 + BN) * 64 bytes). Rows of X past `rows` and k past K load as zero. X and W rows are K bytes apart.
// Every address is precomputed: a thread's copies sit 32 rows (2 KB of shared memory) apart, the stage offsets
// rotate without division, and the K-tail test folds away when K is a multiple of 64.
// Leaves the ring drained and the block synchronised, so the caller may reuse the shared memory.
#define GSB(BN) ((128 + (BN)) * 64) // bytes per stage: A tile | B tile
#define GLOAD(s, k)                                                                     \
    do {                                                                                \
        okk = K % 64 == 0 || (k) * 64 + 16 * (tid & 3) < K;                             \
        for (i = 0; i < 4; i++) {                                                       \
            ok = okk && m0 + (tid >> 2) + 32 * i < rows;                                \
            cp16((s) + da + i * 2048, ok ? xa + (size_t)i * 32 * K + (k) * 64 : x, ok ? 16 : 0); \
        }                                                                               \
        for (i = 0; i < BN / 32; i++)                                                   \
            cp16((s) + 128 * 64 + da + i * 2048, okk ? wa + (size_t)i * 32 * K + (k) * 64 : w, okk ? 16 : 0); \
    } while (0)

// fragments of k half kk of the stage at st -> buffer f; all 2 x BN / 8 MMAs of buffer f
#define GFRAG(f, st, kk)                                                                \
    do {                                                                                \
        for (mt = 0; mt < 2; mt++) ldsm4(a[f][mt], (st) + ao[kk] + mt * 1024);          \
        for (n = 0; n < BN / 16; n++) ldsm4(b[f][n], (st) + bo[kk] + n * 1024);         \
    } while (0)
#define GMMA(f)                                                                         \
    do {                                                                                \
        for (mt = 0; mt < 2; mt++)                                                      \
            for (n = 0; n < BN / 16; n++) {                                             \
                MMAK(H, acc[mt][2 * n], a[f][mt], b[f][n][0], b[f][n][1]);              \
                MMAK(H, acc[mt][2 * n + 1], a[f][mt], b[f][n][2], b[f][n][3]);          \
            }                                                                           \
    } while (0)

// H = 1: fp16 operands (mma16816h), K still in bytes (two halves per byte pair).
template <int BN, int K, int H = 0>
__device__ __forceinline__ void gemm8(const unsigned char *__restrict__ x, const unsigned char *__restrict__ w,
                                      unsigned m0, unsigned rows, unsigned sb, float (&acc)[2][BN / 8][4])
{
    const unsigned char *xa, *wa;
    unsigned ls, cs, da, tid, lane, warp, kt, kk, mt, n, i, r, ok, okk, ao[2], bo[2];
    unsigned a[2][2][4], b[2][BN / 16][4];

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    // per-lane ldmatrix offsets: A rows 32 warp + (lane & 15), chunk 2kk + (lane >> 4); B rows (lane & 7) +
    // 8 (lane >> 4), chunk 2kk + ((lane >> 3) & 1). Row bases are multiples of 8, so the swizzle is per lane.
    for (kk = 0; kk < 2; kk++) {
        r = warp * 32 + (lane & 15);
        ao[kk] = r * 64 + (((2 * kk + (lane >> 4)) ^ ((r >> 1) & 3)) << 4);
        r = (lane & 7) + 8 * (lane >> 4);
        bo[kk] = 128 * 64 + r * 64 + (((2 * kk + ((lane >> 3) & 1)) ^ ((r >> 1) & 3)) << 4);
    }
    // copy i of a thread: row (tid >> 2) + 32 i, chunk tid & 3 (same swizzle for every i)
    xa = x + (size_t)(m0 + (tid >> 2)) * K + 16 * (tid & 3);
    wa = w + (size_t)(tid >> 2) * K + 16 * (tid & 3);
    da = SWV(tid >> 2, tid & 3);
#pragma unroll
    for (mt = 0; mt < 2; mt++)
        for (n = 0; n < BN / 8; n++) acc[mt][n][0] = acc[mt][n][1] = acc[mt][n][2] = acc[mt][n][3] = 0.f;
    for (kt = 0; kt < GS - 1; kt++) {
        GLOAD(sb + kt * GSB(BN), kt);
        CP_COMMIT();
    }
    // Fragments are double-buffered: half kk + 1 (or the next stage's half 0) loads while half kk multiplies, and
    // the one barrier per k step sits between the two halves, after the ring slot it frees was last read.
    asm volatile("cp.async.wait_group %0;\n" ::"n"(GS - 2));
    __syncthreads();
    ls = (GS - 1) * GSB(BN);
    cs = 0;
    GFRAG(0, sb, 0);
    for (kt = 0; kt < (K + 63) / 64; kt++) {
        GFRAG(1, sb + cs, 1);
        GMMA(0);
        GMMA(1);
        if (kt + GS - 1 < (K + 63) / 64) GLOAD(sb + ls, kt + GS - 1);
        CP_COMMIT();
        ls = ls == (GS - 1) * GSB(BN) ? 0 : ls + GSB(BN);
        cs = cs == (GS - 1) * GSB(BN) ? 0 : cs + GSB(BN);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(GS - 2));
        __syncthreads();
        if (kt + 1 < (K + 63) / 64) GFRAG(0, sb + cs, 0);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
}

// Patch embedding on gemm8's pipeline in fp16: residual rows of patch token m (fp16, fp32 accumulate) =
// in [rows][K0P] . pe [1024][K0P]^T. CLS: each image's 1024 patches sit after its cls row (x row m + m / 1024 + 1),
// else row m. Grid (1024 / 128, rows / 128), rows a multiple of 128.
template <int CLS>
__device__ __forceinline__ void patch_gemm(const __half *__restrict__ in, const __half *__restrict__ pe,
                                           res_t *__restrict__ x, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    float acc[2][16][4];
    unsigned lane, warp, m0, mt, nt, i, m, n;

    lane = threadIdx.x & 31;
    warp = threadIdx.x >> 5;
    m0 = blockIdx.y * 128;
    gemm8<128, K0P * 2, 1>((const unsigned char *)in, (const unsigned char *)(pe + (size_t)blockIdx.x * 128 * K0P), m0,
                           rows, (unsigned)__cvta_generic_to_shared(sm), acc);
#pragma unroll
    for (mt = 0; mt < 2; mt++)
#pragma unroll
        for (i = 0; i < 2; i++) {
            m = m0 + warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            m = CLS ? m + (m >> 10) + 1 : m;
#pragma unroll
            for (nt = 0; nt < 16; nt++) {
                n = blockIdx.x * 128 + 8 * nt + 2 * (lane & 3);
                *(__half2 *)(x + (size_t)m * D + n) = __floats2half2_rn(acc[mt][nt][2 * i], acc[mt][nt][2 * i + 1]);
            }
        }
}


// contiguous 4 KB E4M3 tile at g -> swizzled shared tile at s (2 x 16 bytes per thread)
#define LOAD8(s, g, tid, SW)                                                            \
    do {                                                                                \
        for (i = 0; i < 2; i++) {                                                       \
            r = (tid) + 128 * i;                                                        \
            cp16((s) + SW(r >> 2, r & 3), (g) + 16 * r, 16);                            \
        }                                                                               \
    } while (0)

// Fused attention, one head, 64 queries per block (4 warps x 16 rows), 64-key E4M3 tiles double-buffered via
// cp.async; 20 KB of shared memory and <= 128 registers: 4 blocks per SM. P never leaves registers.
// Sequences of L tokens (keys padded to a multiple of 64; the tail tile is masked only when L % 64 != 0):
// q8 / k8 [S][16][ceil(L / 64) * 64][64], vt8 [S][16][ceil(L / 64)][64][64] -> out [S, L, 1024] E4M3 at column
// h * 64. Grid (ceil(L / 64), 16, S); the caller is __launch_bounds__(128, 4).
template <int L>
__device__ __forceinline__ void flash8(const unsigned char *__restrict__ q8,
                                                            const unsigned char *__restrict__ k8,
                                                            const unsigned char *__restrict__ v8, void *__restrict__ out,
                                                            float sinv, float vo)
{
    __shared__ __align__(128) unsigned char sm[4096 + 4 * 4096]; // Q | K0 | K1 | V0 | V1
    const unsigned char *kg, *vg;
    unsigned char *dst;
    unsigned sq, sk, sv, kt, vt, tid, lane, warp, q0, j, kk, n, i, r, c, key, ko, vx0, vx1;
    unsigned qa[2][4], kb[4], vb[4], pa[4];
    float s[8][4], o[8][4], m0, m1, l0, l1, mn0, mn1, tm0, tm1, sc0, sc1, nm0, nm1, thr, dthr;

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    q0 = blockIdx.x * 64;
    kg = k8 + ((size_t)blockIdx.z * 16 + blockIdx.y) * ((L + 63) / 64 * 64) * 64;
    vg = v8 + ((size_t)blockIdx.z * 16 + blockIdx.y) * ((L + 63) / 64) * 4096;
    sq = (unsigned)__cvta_generic_to_shared(sm);
    sk = sq + 4096;
    sv = sk + 8192;
    // per-lane ldmatrix row offsets: K rows 4(i>>1) + (i&1) of the n-tile's group, V^T rows i of the dim tile
    i = lane & 7;
    c = lane >> 3;
    ko = 64 * (4 * (i >> 1) + (i & 1)) + ((c ^ (i >> 1)) << 4);
    vx0 = 512 * (c >> 1) + 64 * i + (((c & 1) ^ (i >> 1)) << 4);
    vx1 = 512 * (c >> 1) + 64 * i + (((2 + (c & 1)) ^ (i >> 1)) << 4);
    thr = RESCALE / sinv;
    dthr = PSKIP / sinv;
    LOAD8(sq, q8 + ((size_t)blockIdx.z * 16 + blockIdx.y) * ((L + 63) / 64 * 64) * 64 + q0 * 64, tid, SWV);
    LOAD8(sk, kg, tid, SWK);
    LOAD8(sv, vg, tid, SWV);
    CP_COMMIT();
    for (n = 0; n < 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    m0 = m1 = -1e30f;
    l0 = l1 = 0.f;
    for (j = 0; j < (L + 63) / 64; j++) {
        if (j + 1 < (L + 63) / 64) {
            LOAD8(sk + ((j + 1) & 1) * 4096, kg + (j + 1) * 4096, tid, SWK);
            LOAD8(sv + ((j + 1) & 1) * 4096, vg + (j + 1) * 4096, tid, SWV);
        }
        CP_COMMIT();
        CP_WAIT1();
        __syncthreads();
        if (!j)
            for (kk = 0; kk < 2; kk++) {
                r = warp * 16 + (lane & 15);
                ldsm4(qa[kk], sq + SWV(r, 2 * kk + (lane >> 4)));
            }
        kt = sk + (j & 1) * 4096;
        vt = sv + (j & 1) * 4096;

        for (n = 0; n < 8; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
            ldsm4(kb, kt + 64 * (16 * (n >> 1) + 2 * (n & 1)) + ko);
            mma16832(s[n], qa[0], kb[0], kb[1]);
            mma16832(s[n], qa[1], kb[2], kb[3]);
        }
        if (L % 64 && j == (L + 63) / 64 - 1)
            for (n = 0; n < 8; n++) {
                key = j * 64 + 16 * (n >> 1) + 2 * (n & 1) + 4 * (lane & 3);
                if (key >= L) s[n][0] = s[n][2] = __int_as_float(0xff800000);
                if (key + 1 >= L) s[n][1] = s[n][3] = __int_as_float(0xff800000);
            }

        tm0 = tm1 = __int_as_float(0xff800000);
        for (n = 0; n < 8; n++) {
            tm0 = fmaxf(tm0, fmaxf(s[n][0], s[n][1]));
            tm1 = fmaxf(tm1, fmaxf(s[n][2], s[n][3]));
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        mn0 = fmaxf(m0, tm0);
        mn1 = fmaxf(m1, tm1);
        // Predict the running max holds: keep the stale max (p <= 2^RESCALE, inside E4M3 and exact in fp32) and
        // skip the O / l rescale unless some row of the warp grew past it. Warp-uniform branch.
        if (__any_sync(0xffffffffu, mn0 > m0 + thr || mn1 > m1 + thr)) {
            sc0 = exp2f((m0 - mn0) * sinv);
            sc1 = exp2f((m1 - mn1) * sinv);
            m0 = mn0;
            m1 = mn1;
            l0 *= sc0;
            l1 *= sc1;
            for (n = 0; n < 8; n++) {
                o[n][0] *= sc0; o[n][1] *= sc0;
                o[n][2] *= sc1; o[n][3] *= sc1;
            }
        }
        // Dead tile: every p of the warp's 16 x 64 tile is below 2^-PSKIP, which E4M3 rounds to zero, so P V adds
        // exactly nothing: skip the exp2s and the P V MMAs. Warp-uniform; the block barrier below is still met.
        if (__all_sync(0xffffffffu, tm0 < m0 - dthr && tm1 < m1 - dthr)) {
            __syncthreads();
            continue;
        }
        nm0 = -m0 * sinv;
        nm1 = -m1 * sinv;
        for (n = 0; n < 8; n++) {
            s[n][0] = exp2f(fmaf(s[n][0], sinv, nm0)); s[n][1] = exp2f(fmaf(s[n][1], sinv, nm0));
            s[n][2] = exp2f(fmaf(s[n][2], sinv, nm1)); s[n][3] = exp2f(fmaf(s[n][3], sinv, nm1));
            l0 += s[n][0] + s[n][1];
            l1 += s[n][2] + s[n][3];
        }

        for (kk = 0; kk < 2; kk++) {
            pa[0] = q2(s[4 * kk][0], s[4 * kk][1]) | q2(s[4 * kk + 1][0], s[4 * kk + 1][1]) << 16;
            pa[1] = q2(s[4 * kk][2], s[4 * kk][3]) | q2(s[4 * kk + 1][2], s[4 * kk + 1][3]) << 16;
            pa[2] = q2(s[4 * kk + 2][0], s[4 * kk + 2][1]) | q2(s[4 * kk + 3][0], s[4 * kk + 3][1]) << 16;
            pa[3] = q2(s[4 * kk + 2][2], s[4 * kk + 2][3]) | q2(s[4 * kk + 3][2], s[4 * kk + 3][3]) << 16;
            for (n = 0; n < 4; n++) {
                ldsm4(vb, vt + 1024 * n + (kk ? vx1 : vx0));
                mma16832(o[2 * n], pa, vb[0], vb[1]);
                mma16832(o[2 * n + 1], pa, vb[2], vb[3]);
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    l0 = vo / l0;
    l1 = vo / l1;
    // output staging (64 x 64 fp16, 128-byte rows) in the drained K buffers
    r = warp * 16 + (lane >> 2);
    for (n = 0; n < 8; n++) {
        *(unsigned *)(sm + 4096 + SWZ(r, n) + 4 * (lane & 3)) = pack2(o[n][0] * l0, o[n][1] * l0);
        *(unsigned *)(sm + 4096 + SWZ(r + 8, n) + 4 * (lane & 3)) = pack2(o[n][2] * l1, o[n][3] * l1);
    }
    __syncthreads();
    dst = (unsigned char *)out + (size_t)blockIdx.z * L * D + blockIdx.y * 64;
    for (j = 0; j < 4; j++) {
        kk = tid + 128 * j;
        r = kk >> 3;
        c = kk & 7;
        if (q0 + r < L) st8h<1>(dst + (size_t)(q0 + r) * D + c * 8, *(const uint4 *)(sm + 4096 + SWZ(r, c)), 1.0f);
    }
}
#else
// Fused attention, one head, 64 queries per block (4 warps x 16 rows), 64-key tiles double-buffered via cp.async.
// S = Q K^T and O += P V on mma.m16n8k16 (fp32 accumulate); online softmax in registers; P never leaves registers.
// qkv [S, L, 3072] (q | k | v, head-major 64 each, q and k rotated) -> out [S, L, 1024] fp16 at column h * 64.
// Grid (ceil(L / 64), 16, S); the caller is __launch_bounds__(128).
template <int L>
__device__ __forceinline__ void flash16(const __half *__restrict__ qkv, void *__restrict__ out)
{
    __shared__ __align__(128) unsigned char sm[5 * 8192]; // Q | K0 | K1 | V0 | V1
    const __half *base;
    unsigned sq, sk, sv, kt, vt, tid, lane, warp, q0, j, kk, n, h, r, c, key;
    unsigned qa[4][4], kb[4], vb[4], pa[4];
    float s[8][4], o[8][4], m0, m1, l0, l1, mn0, mn1, sc0, sc1;

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    q0 = blockIdx.x * 64;
    base = qkv + (size_t)blockIdx.z * L * QKV + blockIdx.y * 64;
    sq = (unsigned)__cvta_generic_to_shared(sm);
    sk = sq + 8192;
    sv = sq + 3 * 8192;
    load_tile<L>(sq, base, q0, tid);
    load_tile<L>(sk, base + D, 0, tid);
    load_tile<L>(sv, base + 2 * D, 0, tid);
    CP_COMMIT();
    for (n = 0; n < 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    m0 = m1 = -1e30f;
    l0 = l1 = 0.f;
    for (j = 0; j < (L + 63) / 64; j++) {
        if (j + 1 < (L + 63) / 64) {
            load_tile<L>(sk + ((j + 1) & 1) * 8192, base + D, (j + 1) * 64, tid);
            load_tile<L>(sv + ((j + 1) & 1) * 8192, base + 2 * D, (j + 1) * 64, tid);
        }
        CP_COMMIT();
        CP_WAIT1();
        __syncthreads();
        if (!j)
            for (kk = 0; kk < 4; kk++) ldsm4(qa[kk], sq + SWZ(warp * 16 + (lane & 15), 2 * kk + (lane >> 4)));
        kt = sk + (j & 1) * 8192;
        vt = sv + (j & 1) * 8192;

        for (n = 0; n < 8; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
            for (h = 0; h < 2; h++) {
                ldsm4(kb, kt + SWZ(n * 8 + (lane & 7), 4 * h + (lane >> 3)));
                mma16816(s[n], qa[2 * h], kb[0], kb[1]);
                mma16816(s[n], qa[2 * h + 1], kb[2], kb[3]);
            }
        }
        if (L % 64 && j == (L + 63) / 64 - 1)
            for (n = 0; n < 8; n++) {
                key = j * 64 + n * 8 + 2 * (lane & 3);
                if (key >= L) s[n][0] = s[n][2] = __int_as_float(0xff800000);
                if (key + 1 >= L) s[n][1] = s[n][3] = __int_as_float(0xff800000);
            }

        mn0 = m0;
        mn1 = m1;
        for (n = 0; n < 8; n++) {
            mn0 = fmaxf(mn0, fmaxf(s[n][0], s[n][1]));
            mn1 = fmaxf(mn1, fmaxf(s[n][2], s[n][3]));
        }
        mn0 = fmaxf(mn0, __shfl_xor_sync(0xffffffffu, mn0, 1));
        mn0 = fmaxf(mn0, __shfl_xor_sync(0xffffffffu, mn0, 2));
        mn1 = fmaxf(mn1, __shfl_xor_sync(0xffffffffu, mn1, 1));
        mn1 = fmaxf(mn1, __shfl_xor_sync(0xffffffffu, mn1, 2));
        if (__any_sync(0xffffffffu, mn0 > m0 + RESCALE || mn1 > m1 + RESCALE)) {
            sc0 = exp2f(m0 - mn0);
            sc1 = exp2f(m1 - mn1);
            m0 = mn0;
            m1 = mn1;
            l0 *= sc0;
            l1 *= sc1;
            for (n = 0; n < 8; n++) {
                o[n][0] *= sc0; o[n][1] *= sc0;
                o[n][2] *= sc1; o[n][3] *= sc1;
            }
        }
        for (n = 0; n < 8; n++) {
            s[n][0] = exp2f(s[n][0] - m0); s[n][1] = exp2f(s[n][1] - m0);
            s[n][2] = exp2f(s[n][2] - m1); s[n][3] = exp2f(s[n][3] - m1);
            l0 += s[n][0] + s[n][1];
            l1 += s[n][2] + s[n][3];
        }

        for (kk = 0; kk < 4; kk++) {
            pa[0] = pack2(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = pack2(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = pack2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            for (n = 0; n < 4; n++) {
                ldsm4t(vb, vt + SWZ(16 * kk + (lane & 15), 2 * n + (lane >> 4)));
                mma16816(o[2 * n], pa, vb[0], vb[1]);
                mma16816(o[2 * n + 1], pa, vb[2], vb[3]);
            }
        }
        __syncthreads();
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    l0 = 1.0f / l0;
    l1 = 1.0f / l1;
    r = warp * 16 + (lane >> 2);
    for (n = 0; n < 8; n++) {
        *(unsigned *)(sm + SWZ(r, n) + 4 * (lane & 3)) = pack2(o[n][0] * l0, o[n][1] * l0);
        *(unsigned *)(sm + SWZ(r + 8, n) + 4 * (lane & 3)) = pack2(o[n][2] * l1, o[n][3] * l1);
    }
    __syncthreads();
    for (j = 0; j < 4; j++) {
        kk = tid + 128 * j;
        r = kk >> 3;
        c = kk & 7;
        if (q0 + r < L)
            *(uint4 *)((__half *)out + ((size_t)blockIdx.z * L + q0 + r) * D + blockIdx.y * 64 + c * 8) = *(const uint4 *)(sm + SWZ(r, c));
    }
}
#endif

// out [B, nt] = sigmoid(logit [B, ntp] + bias).
extern "C" __global__ void k_head(const float *__restrict__ logit, const float *__restrict__ bias, float *__restrict__ out,
                                  unsigned nt, unsigned ntp, unsigned total)
{
    unsigned i, b, j;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    b = i / nt;
    j = i - b * nt;
    out[i] = 1.0f / (1.0f + __expf(-(logit[(size_t)b * ntp + j] + bias[j])));
}

// Tag head as a batched GEMV with the sigmoid fused: out [nb][nt] = sigmoid(a [nb][1024] . w[n]^T + bias[n]), fp16
// a and w, nb <= 8. The weights stream once per batch; every row of a sits in shared memory. One warp per column,
// 8 columns per warp (q = 4 column-quarters x 8 columns, one flat loop), 64 columns per 256-thread block.
#define HMV_COLS 64
#define HMV_DOT(wv, av, s)                                                              \
    do {                                                                                \
        wf = __half22float2(((const __half2 *)&(wv))[0]); af = __half22float2(((const __half2 *)&(av))[0]); \
        s = fmaf(wf.x, af.x, fmaf(wf.y, af.y, s));                                      \
        wf = __half22float2(((const __half2 *)&(wv))[1]); af = __half22float2(((const __half2 *)&(av))[1]); \
        s = fmaf(wf.x, af.x, fmaf(wf.y, af.y, s));                                      \
        wf = __half22float2(((const __half2 *)&(wv))[2]); af = __half22float2(((const __half2 *)&(av))[2]); \
        s = fmaf(wf.x, af.x, fmaf(wf.y, af.y, s));                                      \
        wf = __half22float2(((const __half2 *)&(wv))[3]); af = __half22float2(((const __half2 *)&(av))[3]); \
        s = fmaf(wf.x, af.x, fmaf(wf.y, af.y, s));                                      \
    } while (0)

extern "C" __global__ void __launch_bounds__(256) k_headmv(const __half *__restrict__ a, const __half *__restrict__ w,
                                                          const float *__restrict__ bias, float *__restrict__ out,
                                                          unsigned nt, unsigned nb)
{
    __shared__ __align__(16) __half sa[8 * D];
    uint4 wv, av;
    float2 wf, af;
    float acc[8], s, o;
    unsigned lane, n, b, q, j;

    for (q = threadIdx.x; q < nb * D / 8; q += 256) ((uint4 *)sa)[q] = ((const uint4 *)a)[q];
    __syncthreads();
    lane = threadIdx.x & 31;
    for (b = 0; b < 8; b++) acc[b] = 0.f;
    for (q = 0; q < 32; q++) {
        n = blockIdx.x * HMV_COLS + (threadIdx.x >> 5) * 8 + (q >> 2);
        j = q & 3;
        if (n >= nt) break; // warp-uniform: later columns of this warp are past nt too
        wv = __ldg((const uint4 *)(w + (size_t)n * D) + lane + 32 * j);
        for (b = 0; b < 8; b++) {
            if (b >= nb) break;
            av = ((const uint4 *)(sa + b * D))[lane + 32 * j];
            HMV_DOT(wv, av, acc[b]);
        }
        if (j != 3) continue;
        for (b = 0, o = 0.f; b < 8; b++) {
            s = warp_sum(acc[b]);
            o = lane == b ? s : o;
            acc[b] = 0.f;
        }
        if (lane < nb) out[(size_t)lane * nt + n] = 1.0f / (1.0f + __expf(-(o + bias[n])));
    }
}
