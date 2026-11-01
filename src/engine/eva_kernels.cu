// EVA02-L tagger kernels. Compiled to a cubin at build time by evacc (NVRTC), embedded into the binary.
// Shapes are fixed by the model: D = 1024, T = 1025 tokens, 32 x 32 patches of 14 x 14, hidden 2730 (padded 2736).
// q arrives pre-scaled by log2(e) / 8 (folded into Wq), so attention scores are already in the exp2 domain.
// GEMM activations (LN outputs, attention output, SwiGLU-LN output) are E4M3 with -DEVA_FP8 (static calibrated
// scales passed as qs), fp16 otherwise (the calibration build).
#define D    1024
#define T    1025
#define HID  2730
#define HIDP 2736
#define K0   588
#define K0P  592
#define QKV  3072
#define NKT  17      // ceil(T / 64) key tiles
#define SWT  352     // k_swiglu_ln threads: 11 warps, 342 active (HIDP / 8)
#define TP   1088    // keys padded to NKT * 64
#define FC1_GM 8     // k_fc1 raster group: row tiles walked together

#include "kern_common.cuh"

// x[b, t, :] = (t ? patch_embed : 0) + pos[t]   (pos carries cls and the conv bias). One block per token.
extern "C" __global__ void k_embed(res_t *__restrict__ x, const float *__restrict__ pos)
{
    const float4 *pr;
    float4 a, p;
    unsigned t, i;

    t = blockIdx.x % T;
    pr = (const float4 *)(pos + (size_t)t * D);
    i = threadIdx.x; // 256 threads x 4 = 1024
    p = pr[i];
    a = t ? ld_res4(x + (size_t)blockIdx.x * D + 4 * i) : make_float4(0.f, 0.f, 0.f, 0.f);
    st_res4(x + (size_t)blockIdx.x * D + 4 * i, make_float4(a.x + p.x, a.y + p.y, a.z + p.z, a.w + p.w));
}

#ifdef EVA_FP8
// QKV projection with the attention operand layouts as its epilogue. Block (column tile, row tile) covers 2 heads of
// one of q | k | v (alpha * X Wqkv^T, X = LN1 output E4M3):
//  q, k: RoPE (patch tokens; a rotary pair sits in one lane), * cq / ck, dims permuted -> q8 / k8 [B][16][TP][64]
//  v:    * cv, transposed -> vt8 [B][16][NKT][64 dims][64 keys]
// Rows of the padded key ranges are zeroed once at init and never written. Grid (24, ceil(rows / 128)).
extern "C" __global__ void __launch_bounds__(128, 2) k_qkv(const unsigned char *__restrict__ x,
                                                          const unsigned char *__restrict__ w,
                                                          unsigned char *__restrict__ q8, unsigned char *__restrict__ k8,
                                                          unsigned char *__restrict__ v8, const float *__restrict__ cs,
                                                          float alpha, float cq, float ck, float cv, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    unsigned char *dst;
    float acc[2][16][4], f0, f1, g0, sc;
    float2 a2;
    unsigned tid, lane, warp, m0, p, hh, mt, nt, i, r, m, b, t, d, idx, pr;

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    m0 = blockIdx.y * 128;
    p = blockIdx.x >> 3;
    hh = (blockIdx.x & 7) * 2;
    gemm8<128, D>(x, w + (size_t)blockIdx.x * 128 * D, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
    sc = alpha * (p == 0 ? cq : p == 1 ? ck : cv);
#pragma unroll
    for (mt = 0; mt < 2; mt++)
#pragma unroll
        for (i = 0; i < 2; i++) {
            r = warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            m = m0 + r;
            t = m % T;
#pragma unroll
            for (nt = 0; nt < 16; nt++) {
                f0 = acc[mt][nt][2 * i] * sc;
                f1 = acc[mt][nt][2 * i + 1] * sc;
                d = 8 * (nt & 7) + 2 * (lane & 3);
                if (p < 2) {
                    if (t && m < rows) {
                        a2 = *(const float2 *)(cs + ((size_t)(t - 1) * 32 + (d >> 1)) * 2);
                        g0 = f0 * a2.x - f1 * a2.y;
                        f1 = f1 * a2.x + f0 * a2.y;
                        f0 = g0;
                    }
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
            if (m >= rows) continue;
            b = m / T;
            t = m - b * T;
            *(uint4 *)(dst + (((size_t)b * 16 + hh + ((idx >> 2) & 1)) * TP + t) * 64 + 16 * (idx & 3)) =
                *(const uint4 *)(sm + r * 128 + 16 * (idx & 7));
        }
    } else {
        for (i = 0; i < 128; i++) {
            idx = tid + 128 * i;
            r = idx & 127;
            d = idx >> 7;
            m = m0 + r;
            if (m >= rows) continue;
            b = m / T;
            t = m - b * T;
            v8[(((size_t)b * 16 + hh + (d >> 6)) * NKT + (t >> 6)) * 4096 + (d & 63) * 64 + (t & 63)] = sm[d * 144 + r];
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 4) k_flash(const unsigned char *__restrict__ q8,
                                                            const unsigned char *__restrict__ k8,
                                                            const unsigned char *__restrict__ v8, void *__restrict__ out,
                                                            float sinv, float vo)
{
    flash8<T>(q8, k8, v8, out, sinv, vo);
}

// Patch embedding (fp16 on gemm8): x rows 1..1024 of every image = in . pe^T; k_embed adds pos / cls after.
// Grid (1024 / 128, B * 1024 / 128).
extern "C" __global__ void __launch_bounds__(128, 2) k_pe(const __half *__restrict__ in, const __half *__restrict__ pe,
                                                         res_t *__restrict__ x, unsigned rows)
{
    patch_gemm<1>(in, pe, x, rows);
}

// Attention out-projection: x [rows][1024] += alpha a W^T (a: flash output E4M3, alpha = weight x activation dequant).
// Grid (1024 / 128, ceil(rows / 128)).
extern "C" __global__ void __launch_bounds__(128, 2) k_oproj(const unsigned char *__restrict__ a,
                                                            const unsigned char *__restrict__ w, res_t *__restrict__ x,
                                                            float alpha, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    __half2 *xp;
    float2 xf;
    float acc[2][16][4];
    unsigned lane, warp, m0, mt, nt, i, m, n;

    lane = threadIdx.x & 31;
    warp = threadIdx.x >> 5;
    m0 = blockIdx.y * 128;
    gemm8<128, D>(a, w + (size_t)blockIdx.x * 128 * D, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
#pragma unroll
    for (mt = 0; mt < 2; mt++)
#pragma unroll
        for (i = 0; i < 2; i++) {
            m = m0 + warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            if (m >= rows) continue;
#pragma unroll
            for (nt = 0; nt < 16; nt++) {
                n = blockIdx.x * 128 + 8 * nt + 2 * (lane & 3);
                xp = (__half2 *)(x + (size_t)m * D + n);
                xf = __half22float2(*xp);
                *xp = __floats2half2_rn(fmaf(alpha, acc[mt][nt][2 * i], xf.x), fmaf(alpha, acc[mt][nt][2 * i + 1], xf.y));
            }
        }
}

#else
// ---- fp16 attention (calibration engine) ----

// Interleaved RoPE on q and k of patch tokens (t >= 1), in place in qkv [M, 3072]. One block (256 threads) per row:
// threads 0..127 rotate q, 128..255 k, one uint4 (4 pairs) each.
extern "C" __global__ void k_prep(__half *__restrict__ qkv, const float *__restrict__ cs)
{
    uint4 *p;
    uint4 v;
    const float4 *tb;
    __half2 *hv;
    float o[8];
    unsigned t, i;

    t = blockIdx.x % T;
    if (!t) return;
    i = threadIdx.x;
    p = (uint4 *)(qkv + (size_t)blockIdx.x * QKV) + i;
    tb = (const float4 *)(cs + ((size_t)(t - 1) * 32 + ((4 * i) & 31)) * 2);
    v = *p;
    rot8(v, tb[0], tb[1], 1.0f, o);
    hv = (__half2 *)&v;
    hv[0] = __floats2half2_rn(o[0], o[1]);
    hv[1] = __floats2half2_rn(o[2], o[3]);
    hv[2] = __floats2half2_rn(o[4], o[5]);
    hv[3] = __floats2half2_rn(o[6], o[7]);
    *p = v;
}

extern "C" __global__ void __launch_bounds__(128) k_flash(const __half *__restrict__ qkv, void *__restrict__ out)
{
    flash16<T>(qkv, out);
}
#endif

#ifdef EVA_FP8
// ---- MLP: fc1 + SwiGLU, then the inner LayerNorm folded into fc2 ----
// LN(h) W2^T = rstd (h W~^T - mean c) + d with W~ = gamma W2, c = W~ 1, d = W2 beta (export_eva.py folds them, with
// a smoothing s~ moved from h into W~'s columns). So h is quantized raw: k_fc1 writes h / s~ as E4M3 plus each row's
// sum and sum of squares; k_fc2 applies mean / rstd per row in its epilogue. No LN pass, no fp16 intermediate.
//
// k_fc1: h8 [M][HIDP] = E4M3(rs * silu(g) * x), (g | x) = alpha * X W12^T, stat [NCT][rows] = (sum h, sum h^2) over
// the column tile's 48 units. W12 rows are g / x interleaved in 16-unit chunks, so a 96-column block tile holds g
// and x of 48 whole units and every lane's accumulators pair up g (n-tile 4c + a) with x (n-tile 4c + 2 + a) of the
// same unit. rs = per-unit quant multiplier (448 / amax over 1 / s~). Grid (ceil(rows / 128), NCT). Every (row,
// column tile) partial has exactly one writer and k_fc2 sums them in a fixed order: no atomics, no clearing, and
// the result is bit-identical run to run (float atomics added the 57 partials in arrival order).
extern "C" __global__ void __launch_bounds__(128, 2) k_fc1(const unsigned char *__restrict__ x,
                                                          const unsigned char *__restrict__ w,
                                                          unsigned char *__restrict__ h8, float *__restrict__ stat,
                                                          const float *__restrict__ rs, float alpha, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * (128 + 96) * 64];
    float acc[2][12][4], ru[3][2][2], g0, g1, x0, x1, s, q;
    unsigned tid, lane, warp, m0, mt, c, kk, i, r, row, lin, grp, gm, rem, ct;

    tid = threadIdx.x;
    lane = tid & 31;
    warp = tid >> 5;
    // Grouped raster: launch order walks FC1_GM row tiles x every column tile, so the ~32 resident blocks share
    // a few X row tiles and W column tiles in L2 instead of streaming all of X once per column tile. Each row's
    // stat still gets few concurrent adders (FC1_GM rows fastest).
    lin = blockIdx.y * gridDim.x + blockIdx.x;
    grp = lin / (FC1_GM * gridDim.y);
    rem = lin - grp * FC1_GM * gridDim.y;
    gm = gridDim.x - grp * FC1_GM < FC1_GM ? gridDim.x - grp * FC1_GM : FC1_GM;
    m0 = (grp * FC1_GM + rem % gm) * 128;
    ct = rem / gm;
    gemm8<96, D>(x, w + (size_t)ct * 96 * D, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
#pragma unroll
    for (c = 0; c < 3; c++)
#pragma unroll
        for (kk = 0; kk < 2; kk++) {
            ru[c][kk][0] = rs[ct * 48 + 16 * c + 8 * kk + 2 * (lane & 3)];
            ru[c][kk][1] = rs[ct * 48 + 16 * c + 8 * kk + 2 * (lane & 3) + 1];
        }
    // unit 16c + 8kk + 2(lane & 3) + e of chunk c: g in n-tile 4c + kk, x in 4c + 2 + kk; every index is constant
    // after unrolling, so acc never leaves registers
#pragma unroll
    for (mt = 0; mt < 2; mt++)
#pragma unroll
        for (i = 0; i < 2; i++) {
            r = warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            s = q = 0.f;
#pragma unroll
            for (c = 0; c < 3; c++)
#pragma unroll
                for (kk = 0; kk < 2; kk++) {
                    g0 = acc[mt][4 * c + kk][2 * i] * alpha;
                    g1 = acc[mt][4 * c + kk][2 * i + 1] * alpha;
                    x0 = acc[mt][4 * c + 2 + kk][2 * i] * alpha;
                    x1 = acc[mt][4 * c + 2 + kk][2 * i + 1] * alpha;
                    x0 *= 0.5f * g0 * (1.0f + tanh_approx(0.5f * g0));
                    x1 *= 0.5f * g1 * (1.0f + tanh_approx(0.5f * g1));
                    s += x0 + x1;
                    q += x0 * x0 + x1 * x1;
                    *(unsigned short *)(sm + r * 48 + 16 * c + 8 * kk + 2 * (lane & 3)) =
                        (unsigned short)q2(x0 * ru[c][kk][0], x1 * ru[c][kk][1]);
                }
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            q += __shfl_xor_sync(0xffffffffu, q, 1);
            q += __shfl_xor_sync(0xffffffffu, q, 2);
            if (!(lane & 3) && m0 + r < rows) *(float2 *)(stat + 2 * ((size_t)ct * rows + m0 + r)) = make_float2(s, q);
        }
    __syncthreads();
    // h8 (22 MB a layer, read once by fc2) leaves evict-first: it must not push X and W out of L2
#pragma unroll
    for (i = 0; i < 3; i++) {
        r = tid + 128 * i;
        row = r / 3;
        c = r - row * 3;
        if (m0 + row < rows)
            __stcs((uint4 *)(h8 + (size_t)(m0 + row) * HIDP + ct * 48 + 16 * c), *(const uint4 *)(sm + row * 48 + 16 * c));
    }
}

// k_fc2: x [M][1024] += rstd (alpha h8 W~^T - mean c) + d, mean / rstd from k_fc1's row sums over the 2730 units
// (pad units are exactly zero). Grid (1024 / 128, ceil(rows / 128)). After the GEMM each thread folds one row's NCT
// partials in tile order (coalesced: consecutive threads, consecutive rows; the 3.7 MB of partials sit in L2) into
// shared memory for the epilogue.
#define NCT (2 * HIDP / 96)
extern "C" __global__ void __launch_bounds__(128, 2) k_fc2(const unsigned char *__restrict__ h8,
                                                          const unsigned char *__restrict__ w, res_t *__restrict__ x,
                                                          const float *__restrict__ stat, const float *__restrict__ cn,
                                                          const float *__restrict__ dn, float alpha, unsigned rows)
{
    __shared__ __align__(128) unsigned char sm[GS * 256 * 64];
    __half2 *xp;
    float2 st, c2, d2, xf;
    float acc[2][16][4], mean, rstd, a, b;
    unsigned lane, warp, m0, mt, nt, i, m, n, c;

    lane = threadIdx.x & 31;
    warp = threadIdx.x >> 5;
    m0 = blockIdx.y * 128;
    gemm8<128, HIDP>(h8, w + (size_t)blockIdx.x * 128 * HIDP, m0, rows, (unsigned)__cvta_generic_to_shared(sm), acc);
    __syncthreads(); // the GEMM's last tile reads are done: sm is free
    m = m0 + threadIdx.x;
    st = make_float2(0.f, 0.f);
#pragma unroll 8
    for (c = 0; c < NCT && m < rows; c++) {
        xf = *(const float2 *)(stat + 2 * ((size_t)c * rows + m));
        st.x += xf.x;
        st.y += xf.y;
    }
    ((float2 *)sm)[threadIdx.x] = st;
    __syncthreads();
#pragma unroll
    for (mt = 0; mt < 2; mt++)
#pragma unroll
        for (i = 0; i < 2; i++) {
            m = m0 + warp * 32 + mt * 16 + (lane >> 2) + 8 * i;
            if (m >= rows) continue;
            st = ((const float2 *)sm)[m - m0];
            mean = st.x * (1.0f / HID);
            rstd = rsqrtf(st.y * (1.0f / HID) - mean * mean + EPS);
            a = alpha * rstd;
            b = -mean * rstd;
#pragma unroll
            for (nt = 0; nt < 16; nt++) {
                n = blockIdx.x * 128 + 8 * nt + 2 * (lane & 3);
                c2 = *(const float2 *)(cn + n);
                d2 = *(const float2 *)(dn + n);
                xp = (__half2 *)(x + (size_t)m * D + n);
                xf = __half22float2(*xp);
                *xp = __floats2half2_rn(xf.x + a * acc[mt][nt][2 * i] + b * c2.x + d2.x,
                                        xf.y + a * acc[mt][nt][2 * i + 1] + b * c2.y + d2.y);
            }
        }
}

#else
// SwiGLU + inner LayerNorm (calibration engine): u [M][2 * HIDP] from cuBLAS, g and x interleaved in 16-unit chunks
// (chunk c: g at 32c, x at 32c + 16) -> y [M][HIDP] = LN(silu(g) * x) over 2730, and hv [M][HIDP] = silu(g) * x
// (fp16, for its range). One block per row, one uint4 (8 units) per thread. Pad units have zero weights, so
// g = x = 0 there and v = 0 exactly: they add nothing to sum / sum of squares, and w = b = 0 writes 0 back.
extern "C" __global__ void __launch_bounds__(SWT) k_swiglu_ln(const __half *__restrict__ u, const float *__restrict__ w,
                                                             const float *__restrict__ bias, void *__restrict__ y,
                                                             __half *__restrict__ hv)
{
    __shared__ float2 red[SWT / 32];
    const __half *ur;
    uint4 gv, xv;
    const __half2 *gh, *xh;
    __half *yr;
    float4 w0, w1, b0, b1;
    float2 gf, xf, pr;
    float v[8], s, q, mean, rstd;
    unsigned i, j, lane, wid;

    i = threadIdx.x;
    lane = i & 31;
    wid = i >> 5;
    s = q = 0.f;
    if (i < HIDP / 8) {
        ur = u + (size_t)blockIdx.x * 2 * HIDP + (i >> 1) * 32 + (i & 1) * 8;
        gv = *(const uint4 *)ur;
        xv = *(const uint4 *)(ur + 16);
        gh = (const __half2 *)&gv;
        xh = (const __half2 *)&xv;
        for (j = 0; j < 4; j++) {
            gf = __half22float2(gh[j]);
            xf = __half22float2(xh[j]);
            v[2 * j] = 0.5f * gf.x * (1.0f + tanh_approx(0.5f * gf.x)) * xf.x;
            v[2 * j + 1] = 0.5f * gf.y * (1.0f + tanh_approx(0.5f * gf.y)) * xf.y;
            s += v[2 * j] + v[2 * j + 1];
            q += v[2 * j] * v[2 * j] + v[2 * j + 1] * v[2 * j + 1];
        }
        *(uint4 *)(hv + (size_t)blockIdx.x * HIDP + 8 * i) =
            make_uint4(pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7]));
    }
    s = warp_sum(s);
    q = warp_sum(q);
    if (!lane) red[wid] = make_float2(s, q);
    __syncthreads();
    for (j = 0, s = q = 0.f; j < SWT / 32; j++) {
        pr = red[j];
        s += pr.x;
        q += pr.y;
    }
    mean = s * (1.0f / HID);
    rstd = rsqrtf(q * (1.0f / HID) - mean * mean + EPS);
    if (i < HIDP / 8) {
        yr = (__half *)y + (size_t)blockIdx.x * HIDP;
        w0 = ((const float4 *)w)[2 * i];
        w1 = ((const float4 *)w)[2 * i + 1];
        b0 = ((const float4 *)bias)[2 * i];
        b1 = ((const float4 *)bias)[2 * i + 1];
        st4<0>((unsigned char *)yr, 8 * i, (v[0] - mean) * rstd * w0.x + b0.x, (v[1] - mean) * rstd * w0.y + b0.y,
               (v[2] - mean) * rstd * w0.z + b0.z, (v[3] - mean) * rstd * w0.w + b0.w, 1.0f);
        st4<0>((unsigned char *)yr, 8 * i + 4, (v[4] - mean) * rstd * w1.x + b1.x, (v[5] - mean) * rstd * w1.y + b1.y,
               (v[6] - mean) * rstd * w1.z + b1.z, (v[7] - mean) * rstd * w1.w + b1.w, 1.0f);
    }
}
#endif


// Mean over patch tokens 1..1024: x [B, T, D] -> p [B, D]. Grid (B, 16): 64 columns per block as float4,
// 16 token groups of 64 tokens, 4 independent accumulators each, reduced through shared memory.
extern "C" __global__ void k_pool(const res_t *__restrict__ x, float *__restrict__ p)
{
    __shared__ float4 red[16][16];
    const res_t *xr;
    float4 a0, a1, a2, a3, v;
    unsigned c, g, t;

    c = threadIdx.x & 15;
    g = threadIdx.x >> 4;
    xr = x + ((size_t)blockIdx.x * T + 1 + g * 64) * D + blockIdx.y * 64 + 4 * c;
    a0 = a1 = a2 = a3 = make_float4(0.f, 0.f, 0.f, 0.f);
    for (t = 0; t < 64; t += 4) {
        v = ld_res4(xr + (size_t)(t + 0) * D); a0.x += v.x; a0.y += v.y; a0.z += v.z; a0.w += v.w;
        v = ld_res4(xr + (size_t)(t + 1) * D); a1.x += v.x; a1.y += v.y; a1.z += v.z; a1.w += v.w;
        v = ld_res4(xr + (size_t)(t + 2) * D); a2.x += v.x; a2.y += v.y; a2.z += v.z; a2.w += v.w;
        v = ld_res4(xr + (size_t)(t + 3) * D); a3.x += v.x; a3.y += v.y; a3.z += v.z; a3.w += v.w;
    }
    red[g][c] = make_float4(a0.x + a1.x + a2.x + a3.x, a0.y + a1.y + a2.y + a3.y,
                            a0.z + a1.z + a2.z + a3.z, a0.w + a1.w + a2.w + a3.w);
    __syncthreads();
    if (g) return;
    for (t = 1, v = red[0][c]; t < 16; t++) {
        a0 = red[t][c];
        v.x += a0.x; v.y += a0.y; v.z += a0.z; v.w += a0.w;
    }
    ((float4 *)(p + (size_t)blockIdx.x * D + blockIdx.y * 64))[c] =
        make_float4(v.x * (1.0f / (T - 1)), v.y * (1.0f / (T - 1)), v.z * (1.0f / (T - 1)), v.w * (1.0f / (T - 1)));
}

