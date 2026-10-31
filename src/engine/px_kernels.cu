// PixAI tagger (ViTDet-L, 1008^2) kernels. Compiled to a cubin at build time by evacc (NVRTC), embedded into the
// binary. Tokens are window-major for the whole network: 72 x 72 patches = 9 windows of 24 x 24, window w = tokens
// 576w .. 576w + 575. Windowed blocks attend within 576-token sequences, global blocks (7, 15, 23, 31) over all 5184;
// every count is a multiple of 64, so no attention tile is ever masked. q arrives pre-scaled by log2(e) / 8.
// GEMM activations are E4M3 with -DEVA_FP8 (static calibrated scales), fp16 otherwise (the calibration build).
#define D    1024
#define T    5184   // tokens per image
#define LW   576    // tokens per window
#define HID  4736
#define K0   588
#define K0P  592
#define QKV  3072

#include "kern_common.cuh"

// x = LN_pre(patch_embed + pos[t]) (tiled abs pos, window-major), residual stream fp16. One warp per row.
extern "C" __global__ void k_pembed(res_t *__restrict__ x, const float *__restrict__ pos, const float *__restrict__ w,
                                    const float *__restrict__ bias, unsigned rows)
{
    res_t *xr;
    const float4 *pr;
    float4 v[8], p, g, bb;
    float s, q, mean, rstd, dx;
    unsigned row, lane, j;

    row = blockIdx.x * 8 + (threadIdx.x >> 5);
    lane = threadIdx.x & 31;
    if (row >= rows) return;
    xr = x + (size_t)row * D;
    pr = (const float4 *)(pos + (size_t)(row % T) * D);
    for (j = 0, s = 0.f; j < 8; j++) {
        v[j] = ld_res4(xr + 4 * (lane + 32 * j));
        p = pr[lane + 32 * j];
        v[j] = make_float4(v[j].x + p.x, v[j].y + p.y, v[j].z + p.z, v[j].w + p.w);
        s += v[j].x + v[j].y + v[j].z + v[j].w;
    }
    mean = warp_sum(s) * (1.0f / D);
    for (j = 0, q = 0.f; j < 8; j++) {
        dx = v[j].x - mean; q += dx * dx;
        dx = v[j].y - mean; q += dx * dx;
        dx = v[j].z - mean; q += dx * dx;
        dx = v[j].w - mean; q += dx * dx;
    }
    rstd = rsqrtf(warp_sum(q) * (1.0f / D) + 1e-5f);
    for (j = 0; j < 8; j++) {
        g = ((const float4 *)w)[lane + 32 * j];
        bb = ((const float4 *)bias)[lane + 32 * j];
        st_res4(xr + 4 * (lane + 32 * j), make_float4((v[j].x - mean) * rstd * g.x + bb.x, (v[j].y - mean) * rstd * g.y + bb.y,
                                                      (v[j].z - mean) * rstd * g.z + bb.z, (v[j].w - mean) * rstd * g.w + bb.w));
    }
}

#ifdef EVA_FP8
// QKV projection with the attention operand layouts as its epilogue. Block (column tile, row tile) covers 2 heads of
// one of q | k | v: (alpha X Wqkv^T + bias) * cq / ck / cv, X = LN1 output E4M3.
//  q, k: RoPE (every token; table row m % L: window-local or global), dims permuted -> q8 / k8 [S][16][L][64]
//  v:    transposed -> vt8 [S][16][L / 64][64 dims][64 keys], 16 keys per store (row tiles start on 64-key tiles)
// S sequences of L tokens (L = 576 windowed, 5184 global). Grid (24, rows / 128).
extern "C" __global__ void __launch_bounds__(128, 2) k_pqkv(const unsigned char *__restrict__ x,
                                                           const unsigned char *__restrict__ w,
                                                           const float *__restrict__ bias, unsigned char *__restrict__ q8,
                                                           unsigned char *__restrict__ k8, unsigned char *__restrict__ v8,
                                                           const float *__restrict__ cs, unsigned L, float alpha,
                                                           float cq, float ck, float cv, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    unsigned char *dst;
    float acc[2][16][4], f0, f1, g0, sc;
    float2 a2, bv;
    unsigned tid, lane, warp, m0, p, hh, mt, nt, i, r, m, s, t, d, idx, pr;

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    m0 = blockIdx.y * 128;
    p = blockIdx.x >> 3;
    hh = (blockIdx.x & 7) * 2;
    gemm8<128, D>(x, w + (size_t)blockIdx.x * 128 * D, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
    sc = p == 0 ? cq : p == 1 ? ck : cv;
    for (mt = 0; mt < 2; mt++)
        for (i = 0; i < 2; i++) {
            r = warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            m = m0 + r;
            t = m % L;
            for (nt = 0; nt < 16; nt++) {
                d = 8 * (nt & 7) + 2 * (lane & 3);
                bv = *(const float2 *)(bias + blockIdx.x * 128 + 8 * nt + 2 * (lane & 3));
                f0 = (acc[mt][nt][2 * i] * alpha + bv.x) * sc;
                f1 = (acc[mt][nt][2 * i + 1] * alpha + bv.y) * sc;
                if (p < 2) {
                    a2 = *(const float2 *)(cs + ((size_t)t * 32 + (d >> 1)) * 2);
                    g0 = f0 * a2.x - f1 * a2.y;
                    f1 = f1 * a2.x + f0 * a2.y;
                    f0 = g0;
                    *(unsigned short *)(sm + r * 128 + 64 * (nt >> 3) + 16 * ((nt & 7) >> 1) + 4 * (lane & 3) + 2 * (nt & 1)) =
                        (unsigned short)q2(f0, f1);
                } else {
                    pr = q2(f0, f1);
                    sm[(64 * (nt >> 3) + d) * 144 + r] = (unsigned char)pr;
                    sm[(64 * (nt >> 3) + d + 1) * 144 + r] = (unsigned char)(pr >> 8);
                }
            }
        }
    __syncthreads();
    if (p < 2) {
        dst = p ? k8 : q8;
        for (i = 0; i < 8; i++) {
            idx = tid + 128 * i;
            r = idx >> 3;
            m = m0 + r;
            s = m / L;
            t = m - s * L;
            *(uint4 *)(dst + (((size_t)s * 16 + hh + ((idx >> 2) & 1)) * L + t) * 64 + 16 * (idx & 3)) =
                *(const uint4 *)(sm + r * 128 + 16 * (idx & 7));
        }
    } else {
        for (i = 0; i < 8; i++) { // 128 dim rows x 8 runs of 16 keys
            idx = tid + 128 * i;
            d = idx >> 3;
            r = 16 * (idx & 7);
            m = m0 + r;
            s = m / L;
            t = m - s * L;
            *(uint4 *)(v8 + (((size_t)s * 16 + hh + (d >> 6)) * (L / 64) + (t >> 6)) * 4096 + (d & 63) * 64 + (t & 63)) =
                *(const uint4 *)(sm + d * 144 + r);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) k_pflash_w(const unsigned char *__restrict__ q8,
                                                               const unsigned char *__restrict__ k8,
                                                               const unsigned char *__restrict__ v8,
                                                               void *__restrict__ out, float sinv, float vo)
{
    flash8<LW>(q8, k8, v8, out, sinv, vo);
}

extern "C" __global__ void __launch_bounds__(128, 4) k_pflash_g(const unsigned char *__restrict__ q8,
                                                               const unsigned char *__restrict__ k8,
                                                               const unsigned char *__restrict__ v8,
                                                               void *__restrict__ out, float sinv, float vo)
{
    flash8<T>(q8, k8, v8, out, sinv, vo);
}

// x [M][1024] += alpha A W^T + bias (A E4M3 [M][K]): attention out-proj (K = 1024) and fc2 (K = 4736).
// Grid (1024 / 128, rows / 128).
template <int K>
__device__ __forceinline__ void resgemm(const unsigned char *__restrict__ a, const unsigned char *__restrict__ w,
                                        res_t *__restrict__ x, const float *__restrict__ bias, float alpha, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    __half2 *xp;
    float2 bv, xf;
    float acc[2][16][4];
    unsigned lane, warp, m0, mt, nt, i, m, n;

    lane = threadIdx.x & 31;
    warp = threadIdx.x >> 5;
    m0 = blockIdx.y * 128;
    gemm8<128, K>(a, w + (size_t)blockIdx.x * 128 * K, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
    for (mt = 0; mt < 2; mt++)
        for (i = 0; i < 2; i++) {
            m = m0 + warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            for (nt = 0; nt < 16; nt++) {
                n = blockIdx.x * 128 + 8 * nt + 2 * (lane & 3);
                bv = *(const float2 *)(bias + n);
                xp = (__half2 *)(x + (size_t)m * D + n);
                xf = __half22float2(*xp);
                *xp = __floats2half2_rn(xf.x + alpha * acc[mt][nt][2 * i] + bv.x, xf.y + alpha * acc[mt][nt][2 * i + 1] + bv.y);
            }
        }
}

extern "C" __global__ void __launch_bounds__(128, 2) k_pproj(const unsigned char *__restrict__ a,
                                                            const unsigned char *__restrict__ w, res_t *__restrict__ x,
                                                            const float *__restrict__ bias, float alpha, unsigned rows)
{
    resgemm<D>(a, w, x, bias, alpha, rows);
}

extern "C" __global__ void __launch_bounds__(128, 2) k_pfc2(const unsigned char *__restrict__ a,
                                                           const unsigned char *__restrict__ w, res_t *__restrict__ x,
                                                           const float *__restrict__ bias, float alpha, unsigned rows)
{
    resgemm<HID>(a, w, x, bias, alpha, rows);
}

// fc1 + GELU: h8 [M][4736] = E4M3(rs * gelu(alpha X W1^T + b1)), exact erf GELU; rs = per-unit quant multiplier with
// the fc2 smoothing folded in. Grid (4736 / 128, rows / 128).
extern "C" __global__ void __launch_bounds__(128, 2) k_pfc1(const unsigned char *__restrict__ x,
                                                           const unsigned char *__restrict__ w,
                                                           const float *__restrict__ bias, const float *__restrict__ rs,
                                                           unsigned char *__restrict__ h8, float alpha, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    float2 bv, rv;
    float acc[2][16][4], v0, v1;
    unsigned tid, lane, warp, m0, mt, nt, i, r, n;

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    m0 = blockIdx.y * 128;
    gemm8<128, D>(x, w + (size_t)blockIdx.x * 128 * D, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
    for (mt = 0; mt < 2; mt++)
        for (i = 0; i < 2; i++) {
            r = warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            for (nt = 0; nt < 16; nt++) {
                n = blockIdx.x * 128 + 8 * nt + 2 * (lane & 3);
                bv = *(const float2 *)(bias + n);
                rv = *(const float2 *)(rs + n);
                v0 = acc[mt][nt][2 * i] * alpha + bv.x;
                v1 = acc[mt][nt][2 * i + 1] * alpha + bv.y;
                v0 *= 0.5f * (1.0f + erff(v0 * 0.70710678f));
                v1 *= 0.5f * (1.0f + erff(v1 * 0.70710678f));
                *(unsigned short *)(sm + r * 128 + 8 * nt + 2 * (lane & 3)) = (unsigned short)q2(v0 * rv.x, v1 * rv.y);
            }
        }
    __syncthreads();
    for (i = 0; i < 8; i++) {
        r = tid + 128 * i;
        *(uint4 *)(h8 + (size_t)(m0 + (r >> 3)) * HID + blockIdx.x * 128 + 16 * (r & 7)) = *(const uint4 *)(sm + (r >> 3) * 128 + 16 * (r & 7));
    }
}

#else
// ---- calibration engine (fp16, cuBLAS GEMMs) ----

// qkv [M][3072] += bias, then interleaved RoPE on q and k (table row m % L), in place. One block (384 threads, one
// uint4 = 8 columns each) per row.
extern "C" __global__ void k_pprep(__half *__restrict__ qkv, const float *__restrict__ bias, const float *__restrict__ cs,
                                   unsigned L)
{
    uint4 *p;
    uint4 v;
    const float4 *tb, *bb;
    __half2 *hv;
    float4 b0, b1, c01, c23;
    float o[8];
    unsigned i, t;

    i = threadIdx.x;
    t = blockIdx.x % L;
    p = (uint4 *)(qkv + (size_t)blockIdx.x * QKV) + i;
    bb = (const float4 *)(bias + 8 * i);
    b0 = bb[0];
    b1 = bb[1];
    v = *p;
    hv = (__half2 *)&v;
    o[0] = __low2float(hv[0]) + b0.x; o[1] = __high2float(hv[0]) + b0.y;
    o[2] = __low2float(hv[1]) + b0.z; o[3] = __high2float(hv[1]) + b0.w;
    o[4] = __low2float(hv[2]) + b1.x; o[5] = __high2float(hv[2]) + b1.y;
    o[6] = __low2float(hv[3]) + b1.z; o[7] = __high2float(hv[3]) + b1.w;
    hv[0] = __floats2half2_rn(o[0], o[1]);
    hv[1] = __floats2half2_rn(o[2], o[3]);
    hv[2] = __floats2half2_rn(o[4], o[5]);
    hv[3] = __floats2half2_rn(o[6], o[7]);
    if (i < 256) { // q and k: 4 pairs at pair index (8 i mod 64) / 2
        tb = (const float4 *)(cs + ((size_t)t * 32 + ((4 * i) & 31)) * 2);
        c01 = tb[0];
        c23 = tb[1];
        rot8(v, c01, c23, 1.0f, o);
        hv[0] = __floats2half2_rn(o[0], o[1]);
        hv[1] = __floats2half2_rn(o[2], o[3]);
        hv[2] = __floats2half2_rn(o[4], o[5]);
        hv[3] = __floats2half2_rn(o[6], o[7]);
    }
    *p = v;
}

extern "C" __global__ void __launch_bounds__(128) k_pflash_w(const __half *__restrict__ qkv, void *__restrict__ out)
{
    flash16<LW>(qkv, out);
}

extern "C" __global__ void __launch_bounds__(128) k_pflash_g(const __half *__restrict__ qkv, void *__restrict__ out)
{
    flash16<T>(qkv, out);
}

// x [M][1024] += bias (after a cuBLAS beta = 1 GEMM into the residual). 256 threads x 4 columns per row.
extern "C" __global__ void k_paddb(res_t *__restrict__ x, const float *__restrict__ bias)
{
    float4 v, b;

    v = ld_res4(x + (size_t)blockIdx.x * D + 4 * threadIdx.x);
    b = ((const float4 *)bias)[threadIdx.x];
    st_res4(x + (size_t)blockIdx.x * D + 4 * threadIdx.x, make_float4(v.x + b.x, v.y + b.y, v.z + b.z, v.w + b.w));
}

// u [M][4736] = gelu(u + b1) in place (exact erf), and per (128-row tile, 16-unit group) the count of groups whose
// every |gelu| < sthr (the block sparsity a speculative fc1 / fc2 could skip) into spar[0]. Grid (4736 / 16, M / 128),
// 128 threads: one row each, 16 units (2 uint4).
extern "C" __global__ void k_pgelu(__half *__restrict__ u, const float *__restrict__ bias, float sthr,
                                   unsigned *__restrict__ spar)
{
    uint4 *p;
    uint4 v[2];
    __half2 *hv;
    float a, b;
    unsigned j, k, small;

    p = (uint4 *)(u + ((size_t)blockIdx.y * 128 + threadIdx.x) * HID + blockIdx.x * 16);
    v[0] = p[0];
    v[1] = p[1];
    small = 1;
    for (k = 0; k < 2; k++) {
        hv = (__half2 *)&v[k];
        for (j = 0; j < 4; j++) {
            a = __low2float(hv[j]) + bias[blockIdx.x * 16 + 8 * k + 2 * j];
            b = __high2float(hv[j]) + bias[blockIdx.x * 16 + 8 * k + 2 * j + 1];
            a *= 0.5f * (1.0f + erff(a * 0.70710678f));
            b *= 0.5f * (1.0f + erff(b * 0.70710678f));
            small &= fabsf(a) < sthr && fabsf(b) < sthr;
            hv[j] = __floats2half2_rn(a, b);
        }
    }
    p[0] = v[0];
    p[1] = v[1];
    small = __syncthreads_and(small);
    if (!threadIdx.x && small) atomicAdd(spar, 1u);
}
#endif

// ---- attention-pool head (fp16 / fp32, both builds) ----

// y = LN_head(x) + hpos[t] (fp16): the pool's keys and values both read it. One warp per row.
extern "C" __global__ void k_phln(const res_t *__restrict__ x, const float *__restrict__ w, const float *__restrict__ bias,
                                  const __half *__restrict__ hpos, __half *__restrict__ y, unsigned rows)
{
    const res_t *xr;
    float4 v[8], g, bb, hp;
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
    rstd = rsqrtf(warp_sum(q) * (1.0f / D) + 1e-5f);
    for (j = 0; j < 8; j++) {
        g = ((const float4 *)w)[lane + 32 * j];
        bb = ((const float4 *)bias)[lane + 32 * j];
        hp = ld_res4(hpos + (size_t)(row % T) * D + 4 * (lane + 32 * j));
        st_res4(y + (size_t)row * D + 4 * (lane + 32 * j),
                make_float4((v[j].x - mean) * rstd * g.x + bb.x + hp.x, (v[j].y - mean) * rstd * g.y + bb.y + hp.y,
                            (v[j].z - mean) * rstd * g.z + bb.z + hp.z, (v[j].w - mean) * rstd * g.w + bb.w + hp.w));
    }
}

// Pool scores (exp2 domain): s [M][16] = U y (the constant query folded into the key projection). One warp per row.
extern "C" __global__ void k_phscore(const __half *__restrict__ y, const float *__restrict__ U, float *__restrict__ s,
                                     unsigned rows)
{
    float4 v[8], u;
    float a;
    unsigned row, lane, j, h;

    row = blockIdx.x * 8 + (threadIdx.x >> 5);
    lane = threadIdx.x & 31;
    if (row >= rows) return;
    for (j = 0; j < 8; j++) v[j] = ld_res4(y + (size_t)row * D + 4 * (lane + 32 * j));
    for (h = 0; h < 16; h++) {
        for (j = 0, a = 0.f; j < 8; j++) {
            u = __ldg((const float4 *)(U + (size_t)h * D) + lane + 32 * j);
            a += v[j].x * u.x + v[j].y * u.y + v[j].z * u.z + v[j].w * u.w;
        }
        a = warp_sum(a);
        if (!lane) s[(size_t)row * 16 + h] = a;
    }
}

// Softmax over the 5184 tokens of each (image, head): P [B][16][T] fp16. One block (256 threads) per (image, head).
extern "C" __global__ void k_phsoft(const float *__restrict__ s, __half *__restrict__ P)
{
    __shared__ float red[8];
    const float *sb;
    float m, l, v;
    unsigned t, lane, wid, b, h;

    b = blockIdx.x >> 4;
    h = blockIdx.x & 15;
    lane = threadIdx.x & 31;
    wid = threadIdx.x >> 5;
    sb = s + (size_t)b * T * 16 + h;
    for (t = threadIdx.x, m = -1e30f; t < T; t += 256) m = fmaxf(m, sb[(size_t)t * 16]);
    for (t = 16; t; t >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, t));
    if (!lane) red[wid] = m;
    __syncthreads();
    for (t = 0, m = red[0]; t < 8; t++) m = fmaxf(m, red[t]);
    __syncthreads();
    for (t = threadIdx.x, l = 0.f; t < T; t += 256) l += exp2f(sb[(size_t)t * 16] - m);
    l = warp_sum(l);
    if (!lane) red[wid] = l;
    __syncthreads();
    for (t = 0, l = 0.f; t < 8; t++) l += red[t];
    l = 1.0f / l;
    for (t = threadIdx.x; t < T; t += 256) {
        v = exp2f(sb[(size_t)t * 16] - m) * l;
        P[(size_t)blockIdx.x * T + t] = __float2half_rn(v);
    }
}

// o [B][1024] (fp16) = per head h: Wv[64h .. 64h + 63] . pooled[b][h] (pooled [B][16][1024] fp32 from cuBLAS P Y).
// Value bias, out-proj and their biases live in the folded tag head. One block (256 threads) per (image, head).
extern "C" __global__ void k_phv(const float *__restrict__ pooled, const __half *__restrict__ wv, __half *__restrict__ o)
{
    const float *pv;
    float4 x[8], wf;
    float a;
    unsigned lane, wid, j, k, b, h, n;

    b = blockIdx.x >> 4;
    h = blockIdx.x & 15;
    lane = threadIdx.x & 31;
    wid = threadIdx.x >> 5;
    pv = pooled + (size_t)blockIdx.x * D;
    for (j = 0; j < 8; j++) x[j] = ((const float4 *)pv)[lane + 32 * j];
    for (k = 0; k < 8; k++) { // 8 warps x 8 outputs
        n = 64 * h + 8 * wid + k;
        for (j = 0, a = 0.f; j < 8; j++) {
            wf = ld_res4(wv + (size_t)n * D + 4 * (lane + 32 * j));
            a += x[j].x * wf.x + x[j].y * wf.y + x[j].z * wf.z + x[j].w * wf.w;
        }
        a = warp_sum(a);
        if (!lane) o[(size_t)b * D + n] = __float2half_rn(a);
    }
}
