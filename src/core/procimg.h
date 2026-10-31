#ifndef PROCIMG_H
#define PROCIMG_H

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "px.h"
#include "hp.h"

// Define Tile Size
#define TILE_SIZE 256 // 256 x 256 tile size

// Naiive variance is done by xorfolding vectors in on themselves.
// If the whole image is self-similar, then the xorvariance is near zero.
// This avoids the retarded way of doing it, which is to compute standard deviations and etcetera.
// Loading the image in grayscale colorspace means every pixel is between 0 and 255, so char sized.
// The resulting xorthresh then, if pixel in the image is the same, is 0.
// If an image has higher variance, we can just use a popcnt. So the thresh should be between 0-8
#define XORTHRESH 2 // Set xorthresh to

// ---------------------------------------------------------------- xorfold
// One pass over the image, 64 bytes per iteration: prefetch, NT load, xor into
// the band accumulators (one ymm per row chunk, no cross-chunk dependency).
//   xf_fold_*: also NT-stores every chunk to the staging buffer (rows of
//              plan.dpitch bytes), then one sfence per image.
//   xf_scan_*: read-only, for when the decoder already wrote into the buffer
//              the GPU reads. dst is ignored (pass src).
// Per band the accumulators collapse per tile, butterfly-reduce to one byte per
// tile and classify in the same register.
//
// Requirements: src and dst 32-byte aligned, stride a multiple of 32 and
// >= row bytes rounded up to 32, 64 readable bytes past the last row, ts a
// power of two >= 32, row bytes >= 32.
// Output per band: plan.pitch tile bytes, plan.words violation words (1 bit per tile).

#define XF_MAX_CHUNKS 2048 // 32-byte chunks per row
#define XF_MAX_COLS   1024
#define XF_PITCH(cols) (((cols) + 31u) & ~31u)
#ifndef XF_PF_DIST
#define XF_PF_DIST 4096 // bytes ahead, 0 disables (T0 4096 best on the corpus sweep)
#endif
#ifndef XF_PF_HINT
#define XF_PF_HINT _MM_HINT_T0
#endif

#define XF_X(a, b) _mm256_xor_si256(a, b)
#define XF_NTLD(p) _mm256_stream_load_si256((__m256i const *)(p))
#define XF_NTST(p, v) _mm256_stream_si256((__m256i *)(p), v)
#define XF_NOST(p, v) ((void)(p), (void)(v))
#define XF_PAIR(a, b, W) XF_X(_mm256_unpacklo_epi##W(a, b), _mm256_unpackhi_epi##W(a, b))
#define XF_ID(v) (v)
// RGBA/BGRA dword -> avg(avg(R, B), G) in byte 0, junk in bytes 1-3 (dropped by plan.lowmask)
#define XF_GRAY4(v) _mm256_avg_epu8(_mm256_avg_epu8(v, _mm256_srli_epi32(v, 16)), _mm256_srli_epi32(v, 8))
#define XF_GRAY1(p) ((uint32_t)(((((p)[0] + (p)[2] + 1) >> 1) + (p)[1] + 1) >> 1))
#if XF_PF_DIST
#define XF_PF(p) _mm_prefetch((const char *)(p) + XF_PF_DIST, XF_PF_HINT)
#else
#define XF_PF(p) ((void)0)
#endif

typedef struct {
    __m256i acc[XF_MAX_CHUNKS + 2]; // vertical fold of the current band, one ymm per row chunk
    __m256i tacc[XF_MAX_COLS + 32]; // per-tile fold (+1 group: empty trailing chunks may map to cols)
    __m256i nib, low4, vlo, vhi;
    uint8_t keep8[64];              // keep8[i] = i < 32 ? 0xFF : 0
} __attribute__((aligned(32))) xf_ctx;

typedef struct {
    __m256i m0, m1;  // masks for the final chunk pair of each row (partial / empty chunks -> 0)
    __m256i lowmask; // all ones for planes, 0xFF per dword for rgba gray
    size_t stride, band_src, band_dst;
    uint32_t ts, npair, dpitch, cols, pitch, words, bands, lasth, lastoff, oddw, oddsel, lastword;
    uint16_t tix[XF_MAX_CHUNKS + 2]; // row chunk -> tile column
} __attribute__((aligned(32))) xf_plan;

// Once per job: thresholds are popcount <= lo or >= hi.
static inline void xf_init(xf_ctx *x, uint32_t lo, uint32_t hi)
{
    uint32_t i;

    for (i = 0; i < 64; i++) x->keep8[i] = (uint8_t)-(i < 32);
    for (i = 0; i < XF_MAX_CHUNKS + 2; i++) x->acc[i] = _mm256_setzero_si256();
    x->nib = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                              0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    x->low4 = _mm256_set1_epi8(0x0F);
    x->vlo = _mm256_set1_epi8((char)(lo + 1));
    x->vhi = _mm256_set1_epi8((char)(hi - 1));
}

// Once per image geometry. w in pixels, stride in bytes, bpp 1/3 (planes) or 4 (rgba).
static inline void xf_plan_init(xf_plan *p, const xf_ctx *x, uint32_t w, uint32_t h, size_t stride,
                                uint32_t ts, uint32_t bpp, uint32_t rgba)
{
    __m256i tmask, ones;
    uint32_t wb, twb, tsh, full, odd, k, t, off, nextb, adv;

    wb = w * bpp;
    twb = ts * bpp;
    tsh = (uint32_t)__builtin_ctz(ts);
    full = wb >> 5;
    odd = full & 1;
    tmask = _mm256_loadu_si256((const __m256i *)(x->keep8 + 32 - (wb & 31)));
    ones = _mm256_set1_epi32(-1);
    // final pair = [last full chunk, tail] when full is odd, else [tail, nothing]
    p->m0 = odd ? ones : tmask;
    p->m1 = odd ? tmask : _mm256_setzero_si256();
    p->npair = (full >> 1) + 1;
    p->lowmask = _mm256_set1_epi32(rgba ? 0xFF : -1);
    p->dpitch = p->npair << 6;
    p->stride = stride;
    p->band_src = stride << tsh;
    p->band_dst = (size_t)p->dpitch << tsh;
    p->ts = ts;

    for (k = 0, t = 0, nextb = twb; k < 2 * p->npair; k++) {
        p->tix[k] = (uint16_t)t;
        off = (k + 1) << 5;
        adv = off >= nextb;
        t += adv;
        nextb += twb & -adv;
    }
    p->cols = p->tix[(wb - 1) >> 5] + 1u;
    p->pitch = XF_PITCH(p->cols);
    p->words = p->pitch >> 5;
    p->lastword = ~0u >> ((32 - (p->cols & 31)) & 31);
    p->bands = (h + ts - 1) >> tsh;
    p->lasth = h - ((p->bands - 1) << tsh);
    p->lastoff = (p->cols - 1) * twb;
    p->oddw = ((wb - p->lastoff) >> __builtin_ctz(bpp)) & 1;
    p->oddsel = (uint32_t)((1ull << (8 * bpp)) - 1);
}

// Collapse a[0..31] into one ymm whose byte i is the XOR of all 32 bytes of a[i]. Destroys a.
static inline __m256i xf_reduce32(__m256i *a)
{
    uint32_t i;

    for (i = 0; i < 16; i++) a[i] = XF_PAIR(a[2 * i], a[2 * i + 1], 8);
    for (i = 0; i < 8; i++)  a[i] = XF_PAIR(a[2 * i], a[2 * i + 1], 16);
    for (i = 0; i < 4; i++)  a[i] = XF_PAIR(a[2 * i], a[2 * i + 1], 32);
    for (i = 0; i < 2; i++)  a[i] = XF_PAIR(a[2 * i], a[2 * i + 1], 64);
    return XF_X(_mm256_permute2x128_si256(a[0], a[1], 0x20), _mm256_permute2x128_si256(a[0], a[1], 0x31));
}

// bh rows, 64 bytes per iteration, one prefetch per line. The final pair of every
// row always runs under plan.m0/m1, so partial rows need no branch.
#define XF_DEF_BAND(name, XFORM, STORE)                                                 \
static inline void name(xf_ctx *x, const xf_plan *p, const uint8_t *s, uint8_t *d,      \
                        uint32_t bh)                                                    \
{                                                                                       \
    __m256i v0, v1, m0, m1;                                                             \
    size_t stride, dpitch;                                                              \
    uint32_t y, k, n;                                                                   \
                                                                                        \
    m0 = p->m0;                                                                         \
    m1 = p->m1;                                                                         \
    stride = p->stride;                                                                 \
    dpitch = p->dpitch;                                                                 \
    n = p->npair - 1;                                                                   \
    for (y = 0; y < bh; y++, s += stride, d += dpitch) {                                \
        for (k = 0; k < n; k++) {                                                       \
            XF_PF(s + 64 * k);                                                          \
            v0 = XF_NTLD(s + 64 * k);                                                   \
            v1 = XF_NTLD(s + 64 * k + 32);                                              \
            STORE(d + 64 * k, v0);                                                      \
            STORE(d + 64 * k + 32, v1);                                                 \
            x->acc[2 * k] = XF_X(x->acc[2 * k], XFORM(v0));                             \
            x->acc[2 * k + 1] = XF_X(x->acc[2 * k + 1], XFORM(v1));                     \
        }                                                                               \
        v0 = _mm256_and_si256(XF_NTLD(s + 64 * k), m0);                                 \
        v1 = _mm256_and_si256(XF_NTLD(s + 64 * k + 32), m1);                            \
        STORE(d + 64 * k, v0);                                                          \
        STORE(d + 64 * k + 32, v1);                                                     \
        x->acc[2 * k] = XF_X(x->acc[2 * k], XFORM(v0));                                 \
        x->acc[2 * k + 1] = XF_X(x->acc[2 * k + 1], XFORM(v1));                         \
    }                                                                                   \
}

XF_DEF_BAND(xf_band_plane, XF_ID, XF_NTST)
XF_DEF_BAND(xf_band_rgba, XF_GRAY4, XF_NTST)
XF_DEF_BAND(xf_band_plane_ro, XF_ID, XF_NOST)
XF_DEF_BAND(xf_band_rgba_ro, XF_GRAY4, XF_NOST)

// Band accumulators -> per-tile ymm -> one byte per tile -> violation bits.
// odd is xored into the last tile (flat odd-area correction, already masked).
static inline void xf_finish(xf_ctx *x, const xf_plan *p, uint32_t odd, uint8_t *tiles, uint32_t *mask)
{
    __m256i v, pc, lowmask, zero;
    uint32_t k, g;

    lowmask = p->lowmask;
    zero = _mm256_setzero_si256();
    for (k = 0; k < p->pitch; k++) x->tacc[k] = zero;
    for (k = 0; k < 2 * p->npair; k++) {
        x->tacc[p->tix[k]] = XF_X(x->tacc[p->tix[k]], _mm256_and_si256(x->acc[k], lowmask));
        x->acc[k] = zero;
    }
    x->tacc[p->cols - 1] = XF_X(x->tacc[p->cols - 1], _mm256_zextsi128_si256(_mm_cvtsi32_si128((int)odd)));
    for (g = 0; g < p->words; g++) {
        v = xf_reduce32(x->tacc + 32 * g);
        _mm256_storeu_si256((__m256i *)(tiles + 32 * g), v);
        pc = _mm256_add_epi8(_mm256_shuffle_epi8(x->nib, _mm256_and_si256(v, x->low4)),
                             _mm256_shuffle_epi8(x->nib, _mm256_and_si256(_mm256_srli_epi16(v, 4), x->low4)));
        mask[g] = (uint32_t)_mm256_movemask_epi8(
            _mm256_or_si256(_mm256_cmpgt_epi8(x->vlo, pc), _mm256_cmpgt_epi8(pc, x->vhi)));
    }
    mask[p->words - 1] &= p->lastword;
}

// First pixel of the band's last tile, folded to one byte (bpp bytes xored).
static inline uint32_t xf_odd_plane(const xf_plan *p, const uint8_t *src)
{
    uint32_t v;

    memcpy(&v, src + p->lastoff, 4);
    v &= p->oddsel;
    v ^= v >> 16;
    v ^= v >> 8;
    return v & 0xFF;
}

#define XF_ODD_RGBA(p, src) XF_GRAY1((src) + (p)->lastoff)
#define XF_ODD_PLANE(p, src) xf_odd_plane(p, src)

#define XF_DEF_FOLD(name, band, ODD, FENCE)                                             \
static inline void name(xf_ctx *x, const xf_plan *p, const uint8_t *src, uint8_t *dst,  \
                        uint8_t *tiles, uint32_t *mask)                                 \
{                                                                                       \
    uint32_t b, bh;                                                                     \
                                                                                        \
    for (b = 0; b < p->bands; b++) {                                                    \
        bh = b + 1 < p->bands ? p->ts : p->lasth;                                       \
        band(x, p, src, dst, bh);                                                       \
        xf_finish(x, p, ODD(p, src) & -(p->oddw & bh & 1), tiles, mask);                \
        src += p->band_src;                                                             \
        dst += p->band_dst;                                                             \
        tiles += p->pitch;                                                              \
        mask += p->words;                                                               \
    }                                                                                   \
    FENCE;                                                                              \
}

// Gray plane (bpp 1) or raw interleaved bytes (bpp 3, parity over all channels).
XF_DEF_FOLD(xf_fold_plane, xf_band_plane, XF_ODD_PLANE, _mm_sfence())
XF_DEF_FOLD(xf_scan_plane, xf_band_plane_ro, XF_ODD_PLANE, (void)0)
// RGBA/BGRA folded as gray avg(avg(R, B), G) without materialising the gray plane.
XF_DEF_FOLD(xf_fold_rgba, xf_band_rgba, XF_ODD_RGBA, _mm_sfence())
XF_DEF_FOLD(xf_scan_rgba, xf_band_rgba_ro, XF_ODD_RGBA, (void)0)

// ---------------------------------------------------------------- MVI selector
// Tile grid up to 64 x 64, one u64 per tile row. Prime rectangles are the
// maximal all-allowed boxes within [min, max] size that hold at least one
// required tile; the cover picks the box hitting the most uncovered required
// tiles (ties: smaller box, i.e. tighter crop) until done or maxmvi.

#define PI_MAX_GRID  64
#define PI_MAX_RECTS 4096
#define PI_MAX_MVI   32
#define PI_RUN(a, len) ((~0ull >> (64 - (len))) << (a)) // len in 1..64

enum { PI_ZERO, PI_DONTCARE, PI_INVERSE };

typedef struct { uint8_t r, c, h, w; } pi_rect;

typedef struct {
    uint64_t allow[PI_MAX_GRID], req[PI_MAX_GRID], unc[PI_MAX_GRID];
    pi_rect rect[PI_MAX_RECTS];
    uint32_t nrect, rows, cols, minw, minh, maxw, maxh;
} pi_sel;

// Every maximal run of m (allowed across rows r0..r0+h-1) becomes one prime, or
// sliding maxw windows when longer. upx/dnx: allowed rows above/below, 0 if the
// box cannot grow that way.
static inline void pi_emit_runs(pi_sel *s, uint32_t r0, uint32_t h, uint64_t m, uint64_t qor,
                                uint64_t upx, uint64_t dnx)
{
    uint64_t x, wm;
    uint32_t a, len, wl, c, last;

    while (m) {
        a = (uint32_t)__builtin_ctzll(m);
        x = ~(m >> a);
        len = x ? (uint32_t)__builtin_ctzll(x) : 64 - a;
        m &= ~PI_RUN(a, len);
        wl = len < s->maxw ? len : s->maxw;
        last = len >= s->minw ? a + len - wl : a - 1;
        for (c = a; c + 1 <= last + 1 && s->nrect < PI_MAX_RECTS; c++) {
            wm = PI_RUN(c, wl);
            if (!(qor & wm) || (upx & wm) == wm || (dnx & wm) == wm) continue;
            s->rect[s->nrect++] = (pi_rect){(uint8_t)r0, (uint8_t)c, (uint8_t)h, (uint8_t)wl};
        }
    }
}

static inline void pi_primes(pi_sel *s)
{
    uint64_t m, qor, up, dn;
    uint32_t r0, h, hmax;

    s->nrect = 0;
    for (r0 = 0; r0 < s->rows; r0++) {
        hmax = s->rows - r0 < s->maxh ? s->rows - r0 : s->maxh;
        up = r0 ? s->allow[r0 - 1] : 0;
        for (h = 1, m = ~0ull, qor = 0; h <= hmax && (m &= s->allow[r0 + h - 1]); h++) {
            qor |= s->req[r0 + h - 1];
            dn = r0 + h < s->rows ? s->allow[r0 + h] : 0;
            if (h >= s->minh) pi_emit_runs(s, r0, h, m, qor, h < s->maxh ? up : 0, h < s->maxh ? dn : 0);
        }
    }
}

static inline uint32_t pi_gain(const pi_sel *s, pi_rect q)
{
    uint64_t wm;
    uint32_t r, g;

    wm = PI_RUN(q.c, q.w);
    for (r = q.r, g = 0; r < q.r + q.h; r++) g += (uint32_t)__builtin_popcountll(s->unc[r] & wm);
    return g;
}

// unc[r] op= rect bits: op 0 sets, 1 clears.
static inline void pi_paint(uint64_t *rowbits, pi_rect q, uint32_t op)
{
    uint64_t wm;
    uint32_t r;

    wm = PI_RUN(q.c, q.w);
    for (r = q.r; r < q.r + q.h; r++) rowbits[r] = op ? rowbits[r] & ~wm : rowbits[r] | wm;
}

// Greedy cover; returns the MVI count, *uncovered gets required tiles no prime can reach.
static inline uint32_t pi_cover(pi_sel *s, pi_rect *out, uint32_t maxmvi, uint32_t *uncovered)
{
    uint32_t i, r, n, g, bg, ba, best, a;

    memset(s->unc, 0, sizeof(s->unc));
    for (i = 0; i < s->nrect; i++) pi_paint(s->unc, s->rect[i], 0);
    for (r = 0, *uncovered = 0; r < s->rows; r++) {
        *uncovered += (uint32_t)__builtin_popcountll(s->req[r] & ~s->unc[r]);
        s->unc[r] &= s->req[r];
    }
    for (n = 0; n < maxmvi; n++) {
        for (i = 0, bg = 0, ba = ~0u, best = 0; i < s->nrect; i++) {
            g = pi_gain(s, s->rect[i]);
            a = (uint32_t)s->rect[i].h * s->rect[i].w;
            best = g > bg || (g == bg && g && a < ba) ? i : best;
            ba = g > bg || (g == bg && g && a < ba) ? a : ba;
            bg = g > bg ? g : bg;
        }
        if (!bg) break;
        out[n] = s->rect[best];
        pi_paint(s->unc, out[n], 1);
    }
    return n;
}

// ---------------------------------------------------------------- tagger pipeline
// Two engines resident on one GPU: wd (EVA02-L, 448^2, cheap) and, with --px, PixAI (ViTDet-L, 1008^2, the heavy
// hitter). Each image takes one of four routes:
//   D  wd full image + wd crops               A  PixAI full image + wd crops
//   B  wd full image + PixAI crops            C  PixAI full image + PixAI crops
// A crop set fed to the same model as the full image is combined per tag and gated exactly:
//   final[t] = full[t] + sum_i c_i * crop_i[t],   c_i = gain * (w_i - weight_thresh)   (ratings: full only)
// with w_i = mean tile popcount of MVI i / 8; crops run only when some decision could flip (crop outputs lie in
// [0, 1]). A crop set fed to the other model contributes its own tags (max over crops). Output: the tags of both,
// one list, deduplicated by name.
//
// Router (hp.h): two online hashed perceptrons predict, per route D / A / B, "this route's tags agree with route C's"
// (F1 >= agree over the names both models know). Stage 1 sees what the CPU knows at push (geometry, MVIs, xorfold
// statistics); a job sent down D is checked again by stage 2 once its wd full pass is back (+ margins of that pass)
// and escalated to A / B / C if stage 2 no longer trusts D. No weights ship: a decaying random sample of images
// (AES-NI counter draws) runs every component, and their component outputs label all four routes for training.
//
// Scheduling, like an out-of-order core: light work (wd) issues as soon as it is ready; heavy work (PixAI) is held
// while the GPU has light work, its inputs prepared by the CPU in the time it would otherwise spend waiting on the
// GPU, and issues when the wd queue drains, when the pinned PixAI slots fill, or at the end. Jobs retire as they
// complete (each output line names its image).
//
// Model inputs: alpha over white; wd: white-padded square, antialiased bicubic to 448, BGR; PixAI: aspect resize to
// 1008 (PIL bilinear), centred on black, RGB, window-major patches. Both normalised to [-1, 1] and written directly
// as the patch GEMM operand (fp16, column c * 196 + ky * 14 + kx), streamed into pinned memory.

#define PI_JOBS  32  // images in flight (job bits fit a u32)
#define PI_PXS   16  // pinned PixAI input slots (PX_IN halves each)
#define PI_F1    14  // router features known at push
#define PI_F2    22  // + features of the wd full pass
#define PI_HROWS 8   // log2 perceptron rows per feature
#define PI_WARM  32  // exploration probability max(explore, PI_WARM / (PI_WARM + images routed))
#define PI_OH    2048 // per-image name hash tables (output dedupe, router sets)

enum { PI_D, PI_A, PI_B, PI_C, PI_X }; // routes; X explores: every component, output C, labels for the router
enum { PI_FREE, PI_LIVE };

typedef struct {
    uint32_t ts, lo, hi, mode, minw, minh, maxw, maxh, maxmvi, nocrops;
    int32_t route, rconf;    // route: -1 router, else PI_D..PI_C forced; rconf: perceptron sum to trust a route (-1: theta / 2)
    float weight_thresh, gain, gen_thresh, char_thresh, agree, explore;
    int device;
    const char *px;          // pixai.bin (NULL: wd only); vocabulary in <px>.tags
} pi_params;

static inline pi_params pi_defaults(void)
{
    pi_params p;

    p.ts = TILE_SIZE;
    p.lo = XORTHRESH - 1;
    p.hi = 9;
    p.mode = PI_ZERO;
    p.minw = 2;
    p.minh = 2;
    p.maxw = 0; // 0: grid width
    p.maxh = 0;
    p.maxmvi = 8;
    p.nocrops = 0;
    p.route = -1;
    p.rconf = -1;
    p.weight_thresh = 0.5f;
    p.gain = 1.0f;
    p.gen_thresh = 0.35f;
    p.char_thresh = 0.85f;
    p.agree = 0.8f;
    p.explore = 1.0f / 32.0f;
    p.device = 0;
    p.px = NULL;
    return p;
}

typedef struct { const char *name; float score; } pi_tag;

typedef struct {
    uint32_t n, nmvi, ncrop, uncovered, ts, route; // ncrop: crops run on either model
    uint32_t runm, prun;     // bit i: crop i ran on wd / PixAI
    const char *rating;
    pi_tag *tag;             // n tags over threshold, deduplicated by name (valid during the callback)
    float tvar, req;         // mean tile popcount / 8 over the grid (-1: no grid), share of required tiles
    pi_rect mvi[PI_MAX_MVI];
    float weight[PI_MAX_MVI];
    float area[PI_MAX_MVI];  // crop area / grid area
} pi_result;

typedef struct {
    pi_result res;
    uint16_t *in;            // wd inputs in the pinned pool: [0] full image, [1 + i] crop i (EVA_IN halves)
    float *out, *pout;       // wd / PixAI outputs: [0] full image, [1 + i] crop i
    uint8_t *rgba;           // source pixels (owned), kept while PixAI inputs may still be prepared from them
    void *user;
    size_t istride;
    uint32_t w, h, alpha, state, route, explore;
    uint32_t pend;           // inputs queued or in flight
    uint32_t prep;           // PixAI inputs still to prepare: bit 0 full image, bit 1 + i crop i
    uint32_t wait;           // decisions pending: 1 wd full pass (crop gate, router stage 2), 2 PixAI crop gate
    uint32_t i1[PI_F1], i2[PI_F2]; // perceptron rows
} pi_job;

typedef struct { uint16_t job, idx; } pi_owner;
typedef struct { uint16_t job, idx, slot; } pi_pent;

// One model's vocabulary. Decision: score >= thr (PixAI's strict > is folded in as the next float up; ratings have
// thr 2 and are decided by argmax instead).
typedef struct { uint8_t *p; size_t n; HANDLE f, m; } pi_mapping;
typedef struct {
    uint32_t nt, nr;
    uint32_t rix[8];         // rating tags
    pi_mapping map;          // the vocabulary file, mapped copy-on-write (names are terminated in place)
    const char **name;
    uint8_t *cat, *shared;   // shared: the other model knows the name (router comparisons only)
    float *thr, *mvimask, *chr; // mvimask 0 for ratings (full image only); chr 1 for character tags
    uint32_t *hash;          // CRC32C of the name
} pi_voc;

typedef struct pi_tagger pi_tagger;
typedef void (*pi_done_fn)(pi_tagger *t, void *user, pi_result *res);

struct pi_tagger {
    pi_params prm;
    eva e;
    px p;
    pi_done_fn done;
    pi_voc wv, pv;           // wd / PixAI vocabularies
    hp r1, r2;               // router stage 1 (push) / stage 2 (+ wd full pass)
    __m128i rng, rkey;       // AES-NI counter: exploration draws
    int32_t c1, c2;          // stage confidence (perceptron sum)
    uint32_t S, mt_cap, maxmvi, haspx, routed, owner, err; // owner: job whose composite is in rgbx (PI_JOBS: none)
    xf_ctx *x;
    xf_plan *plan;
    pi_sel *sel;
    uint8_t *rgbx;           // composited RGBX, 32-aligned rows, 64 bytes slack
    size_t rgbx_cap, stride, tmp_cap;
    float *tmp, *wx, *wy, *kx, *ky;
    int32_t *sx, *nx, *sy, *ny;
    uint32_t *jo;            // output column j -> half offset of its pixel inside a patch row band
    uint16_t *band;          // one band of the widest model (72 patches, 14 rows) staged in cache, streamed out
    uint16_t *pool, *ppool;  // pinned inputs: wd PI_JOBS x (1 + maxmvi), PixAI PI_PXS
    size_t poolb, ppoolb;
    float *sc[4];            // combine / router scratch scores (max vocabulary)
    pi_tag *otag;            // output tags of the retiring image
    uint32_t *oh, *ch, *uh;  // name hash tables: output dedupe, route C's set, a candidate route's set
    pi_job job[PI_JOBS];
    uint32_t jfree, pfree, live; // free job / PixAI slot bits; jobs live
    pi_owner own[EVA_NBUF][EVA_NB];
    uint16_t *src[EVA_NBUF][EVA_NB]; // pool input of each slot
    uint32_t cnt[EVA_NBUF];  // inputs in each queued wd batch
    uint32_t batch, fill, sub, col; // wd: filling batch, next to submit / collect
    pi_pent pown[PX_NBUF][PX_NB];
    uint32_t pcnt[PX_NBUF], psub, pcol; // PixAI batches
    pi_pent rdy[PI_PXS];     // prepared PixAI inputs not yet issued
    uint32_t rh, rt;
    uint32_t defer[PI_JOBS], ndefer, fin[PI_JOBS], nfin; // wd crops to queue; jobs to retire
    uint32_t nroute[5], nok[3], hit1[3], hit2[3]; // router statistics (numbers only)
    uint8_t tiles[PI_MAX_GRID * PI_MAX_GRID];
    uint32_t mask[PI_MAX_GRID * 2];
};

#define PI_A32(n) (((size_t)(n) + 31) & ~(size_t)31)
#define PI_GROW(ptr, cap, need, T)                                                      \
    do {                                                                                \
        if ((need) > (cap)) {                                                           \
            _mm_free(ptr);                                                              \
            (ptr) = (T *)_mm_malloc((need) * sizeof(T) + 64, 64);                       \
            (cap) = (need);                                                             \
        }                                                                               \
    } while (0)

// x / 255 rounded, exact for x in [0, 65025]
#define PI_DIV255(x) (((x) + 128 + (((x) + 128) >> 8)) >> 8)
// 16 u16 channel values (4 pixels per lane pair) -> alpha over white, /255 rounded. ashuf broadcasts each alpha.
#define PI_OVER(v, ashuf, c255, k128)                                                   \
    _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(PI_OVER_X(v, ashuf, c255), k128), \
        _mm256_srli_epi16(_mm256_add_epi16(PI_OVER_X(v, ashuf, c255), k128), 8)), 8)
#define PI_OVER_X(v, ashuf, c255)                                                       \
    _mm256_add_epi16(_mm256_mullo_epi16(v, _mm256_shuffle_epi8(v, ashuf)),              \
        _mm256_mullo_epi16(c255, _mm256_sub_epi16(c255, _mm256_shuffle_epi8(v, ashuf))))

// Alpha over white into the aligned RGBX buffer; X = 255. 8 pixels per ymm, exact PI_DIV255 rounding.
static inline void pi_row_rgbx(uint8_t *d, const uint8_t *s, uint32_t w, uint32_t alpha)
{
    __m256i v, lo, hi, ff, zero, ashuf, c255, k128;
    uint32_t i, a, ia;

    ff = _mm256_set1_epi32((int)0xFF000000);
    zero = _mm256_setzero_si256();
    ashuf = _mm256_setr_epi8(6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15,
                             6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15);
    c255 = _mm256_set1_epi16(255);
    k128 = _mm256_set1_epi16(128);
    if (alpha) {
        for (i = 0; i + 8 <= w; i += 8) {
            v = _mm256_loadu_si256((const __m256i *)(s + 4 * i));
            lo = _mm256_unpacklo_epi8(v, zero);
            hi = _mm256_unpackhi_epi8(v, zero);
            v = _mm256_packus_epi16(PI_OVER(lo, ashuf, c255, k128), PI_OVER(hi, ashuf, c255, k128));
            _mm256_store_si256((__m256i *)(d + 4 * i), _mm256_or_si256(v, ff));
        }
    } else {
        for (i = 0; i + 8 <= w; i += 8)
            _mm256_store_si256((__m256i *)(d + 4 * i), _mm256_or_si256(_mm256_loadu_si256((const __m256i *)(s + 4 * i)), ff));
    }
    for (; i < w; i++) {
        a = alpha ? s[4 * i + 3] : 255;
        ia = 255 * (255 - a);
        d[4 * i] = (uint8_t)PI_DIV255(s[4 * i] * a + ia);
        d[4 * i + 1] = (uint8_t)PI_DIV255(s[4 * i + 1] * a + ia);
        d[4 * i + 2] = (uint8_t)PI_DIV255(s[4 * i + 2] * a + ia);
        d[4 * i + 3] = 255;
    }
}

static inline void pi_load(pi_tagger *t, const uint8_t *rgba, uint32_t w, uint32_t h, size_t in_stride, uint32_t alpha)
{
    uint32_t y;
    size_t need;

    t->stride = PI_A32((size_t)w * 4);
    need = t->stride * h + 64;
    PI_GROW(t->rgbx, t->rgbx_cap, need, uint8_t);
    for (y = 0; y < h; y++) {
        _mm_prefetch((const char *)(rgba + (y + 1) * in_stride), _MM_HINT_T0);
        pi_row_rgbx(t->rgbx + y * t->stride, rgba + y * in_stride, w, alpha);
    }
    memset(t->rgbx + t->stride * h, 255, 64);
}

static inline float pi_cubic(float x)
{
    x = x < 0 ? -x : x;
    return x < 1.0f ? (1.5f * x - 2.5f) * x * x + 1.0f
         : x < 2.0f ? ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f : 0.0f;
}

// One axis: virtual white square [0, side) -> [0, S), image content at [off, off + len).
// Output i reads content [s[i], s[i] + n[i]) with weights w[i * mt..], plus k[i] (white taps).
// Weights come out scaled by sc and k offset by add (the x axis folds the [-1, 1] normalisation in).
static inline void pi_axis(uint32_t S, uint32_t side, uint32_t off, uint32_t len, uint32_t mt, float sc, float add,
                           float *w, int32_t *s, int32_t *n, float *k)
{
    float scale, fs, support, ss, center, sum, white, wt, *wi;
    int32_t i, j, lo, hi, a, b, end;

    scale = (float)side / (float)S;
    fs = scale > 1.0f ? scale : 1.0f;
    support = 2.0f * fs;
    ss = 1.0f / fs;
    end = (int32_t)(off + len);
    for (i = 0; i < (int32_t)S; i++) {
        wi = w + (size_t)i * mt;
        center = ((float)i + 0.5f) * scale;
        lo = (int32_t)(center - support + 0.5f);
        lo = lo < 0 ? 0 : lo;
        hi = (int32_t)(center + support + 0.5f);
        hi = hi > (int32_t)side ? (int32_t)side : hi;
        a = lo > (int32_t)off ? lo : (int32_t)off;
        b = hi < end ? hi : end;
        b = b < a ? a : b;
        for (j = lo, sum = 0.0f, white = 0.0f; j < hi; j++) {
            wt = pi_cubic(((float)j - center + 0.5f) * ss);
            sum += wt;
            white += j < a || j >= b ? wt : 0.0f;
            wi[j - lo] = wt;
        }
        memmove(wi, wi + (a - lo), (size_t)(b - a) * sizeof(float));
        for (j = 0, sum = sc / sum; j < b - a; j++) wi[j] *= sum;
        s[i] = a - (int32_t)off;
        n[i] = b - a;
        k[i] = 255.0f * white * sum + add;
    }
}

// Vertical pass for one output row: tmp[x] = k + sum_j w[j] * content_row(s + j)[x]. Register-blocked: 16 pixels
// (64 source bytes, 8 ymm accumulators) take all n taps before tmp is written once, so tmp - cw * 16 bytes, past L1
// on any large crop - is never read back per tap, and each tap row is walked one cache line per block with the next
// line prefetched. Same FMA order per element as the tap-outer loop, so results are bit-identical.
#define PI_U8F(p) _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(p))))
static inline void pi_vrow(const uint8_t *src, size_t stride, uint32_t n8, const float *w,
                           int32_t s, int32_t n, float k, float *tmp)
{
    const uint8_t *row;
    __m256 a0, a1, a2, a3, a4, a5, a6, a7, kv, wv;
    uint32_t c;
    int32_t j;

    kv = _mm256_set1_ps(k);
    src += (size_t)s * stride;
    for (c = 0; c + 8 <= n8; c += 8) {
        a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = kv;
        for (j = 0, row = src + 8 * c; j < n; j++, row += stride) {
            _mm_prefetch((const char *)row + 128, _MM_HINT_T0);
            wv = _mm256_set1_ps(w[j]);
            a0 = _mm256_fmadd_ps(wv, PI_U8F(row), a0);
            a1 = _mm256_fmadd_ps(wv, PI_U8F(row + 8), a1);
            a2 = _mm256_fmadd_ps(wv, PI_U8F(row + 16), a2);
            a3 = _mm256_fmadd_ps(wv, PI_U8F(row + 24), a3);
            a4 = _mm256_fmadd_ps(wv, PI_U8F(row + 32), a4);
            a5 = _mm256_fmadd_ps(wv, PI_U8F(row + 40), a5);
            a6 = _mm256_fmadd_ps(wv, PI_U8F(row + 48), a6);
            a7 = _mm256_fmadd_ps(wv, PI_U8F(row + 56), a7);
        }
        _mm256_storeu_ps(tmp + 8 * c, a0);
        _mm256_storeu_ps(tmp + 8 * c + 8, a1);
        _mm256_storeu_ps(tmp + 8 * c + 16, a2);
        _mm256_storeu_ps(tmp + 8 * c + 24, a3);
        _mm256_storeu_ps(tmp + 8 * c + 32, a4);
        _mm256_storeu_ps(tmp + 8 * c + 40, a5);
        _mm256_storeu_ps(tmp + 8 * c + 48, a6);
        _mm256_storeu_ps(tmp + 8 * c + 56, a7);
    }
    for (; c < n8; c++) {
        a0 = kv;
        for (j = 0, row = src + 8 * c; j < n; j++, row += stride) a0 = _mm256_fmadd_ps(_mm256_set1_ps(w[j]), PI_U8F(row), a0);
        _mm256_storeu_ps(tmp + 8 * c, a0);
    }
}

// Horizontal pass for one output row: RGBX taps in one xmm each, clamp to [-1, 1], F16C convert, and the channels go
// straight to their planes of the patch operand (pixel j -> out + jo[j] + c * 196). ord picks the plane order with
// one vpermilps: PI_BGR (wd) or PI_RGB (PixAI).
#define PI_BGR _mm_setr_epi32(0, 1, 2, 3)
#define PI_RGB _mm_setr_epi32(2, 1, 0, 3)
static inline void pi_hrow(const float *tmp, uint32_t S, uint32_t mt, const float *w, const int32_t *s,
                           const int32_t *n, const float *k, const uint32_t *jo, __m128i ord, uint16_t *out)
{
    const float *wj, *tj;
    __m128 acc, lo, hi;
    __m128i hv;
    uint16_t *o;
    uint32_t j;
    int32_t q;

    lo = _mm_set1_ps(-1.0f);
    hi = _mm_set1_ps(1.0f);
    for (j = 0; j < S; j++) {
        acc = _mm_set1_ps(k[j]);
        wj = w + (size_t)j * mt;
        tj = tmp + 4 * s[j];
        for (q = 0; q < n[j]; q++) acc = _mm_fmadd_ps(_mm_set1_ps(wj[q]), _mm_loadu_ps(tj + 4 * q), acc);
        hv = _mm_cvtps_ph(_mm_permutevar_ps(_mm_min_ps(_mm_max_ps(acc, lo), hi), ord), _MM_FROUND_TO_NEAREST_INT);
        o = out + jo[j];
        o[0] = (uint16_t)_mm_extract_epi16(hv, 2);
        o[196] = (uint16_t)_mm_extract_epi16(hv, 1);
        o[392] = (uint16_t)_mm_extract_epi16(hv, 0);
    }
}

// 14 output rows complete one band: 32 whole patches, 37888 contiguous bytes of the operand. Stream it out
// (non-temporal: the GPU reads it, the CPU never will) from the cache-resident staging band.
#define PI_BANDH (32 * EVA_K0P)
#define PI_SMAX  1008            // largest model side (PixAI)
#define PI_PXRUN (24 * EVA_K0P)  // PixAI: one band = 3 runs of 24 patches, one per window
// n32 x 32 bytes, 4 per step (n32 % 4 == 0)
static inline void pi_stream(uint16_t *dst, const uint16_t *band, uint32_t n32)
{
    uint32_t c;

    for (c = 0; c < n32; c += 4) {
        _mm256_stream_si256((__m256i *)dst + c, _mm256_load_si256((const __m256i *)band + c));
        _mm256_stream_si256((__m256i *)dst + c + 1, _mm256_load_si256((const __m256i *)band + c + 1));
        _mm256_stream_si256((__m256i *)dst + c + 2, _mm256_load_si256((const __m256i *)band + c + 2));
        _mm256_stream_si256((__m256i *)dst + c + 3, _mm256_load_si256((const __m256i *)band + c + 3));
    }
}

// Crop (x0, y0, cw, ch) of the RGBX buffer -> one model input (EVA_IN halves) at out (32-byte aligned, pinned):
// written once, band by band, with streaming stores, fenced at the end so the DMA sees all of it.
static inline void pi_prep(pi_tagger *t, uint32_t x0, uint32_t y0, uint32_t cw, uint32_t ch, uint16_t *out)
{
    const uint8_t *src;
    uint32_t S, side, mt, i, n8;
    size_t need;

    S = t->S;
    side = cw > ch ? cw : ch;
    mt = 2 * (uint32_t)(2.0f * (side > S ? (float)side / (float)S : 1.0f) + 1.0f) + 2;
    if (mt > t->mt_cap) {
        _mm_free(t->wx);
        _mm_free(t->wy);
        t->wx = _mm_malloc((size_t)PI_SMAX * mt * sizeof(float) + 64, 64);
        t->wy = _mm_malloc((size_t)PI_SMAX * mt * sizeof(float) + 64, 64);
        t->mt_cap = mt;
    }
    n8 = (cw * 4 + 7) >> 3;
    need = (size_t)n8 * 8 + 8;
    PI_GROW(t->tmp, t->tmp_cap, need, float);
    pi_axis(S, side, (side - cw) >> 1, cw, mt, 1.0f / 127.5f, -1.0f, t->wx, t->sx, t->nx, t->kx);
    pi_axis(S, side, (side - ch) >> 1, ch, mt, 1.0f, 0.0f, t->wy, t->sy, t->ny, t->ky);
    src = t->rgbx + (size_t)y0 * t->stride + (size_t)x0 * 4;
    for (i = 0; i < S; i++) {
        pi_vrow(src, t->stride, n8, t->wy + (size_t)i * mt, t->sy[i], t->ny[i], t->ky[i], t->tmp);
        pi_hrow(t->tmp, S, mt, t->wx, t->sx, t->nx, t->kx, t->jo, PI_BGR, t->band + (i % 14) * 14);
        if (i % 14 == 13) pi_stream(out + (size_t)(i / 14) * PI_BANDH, t->band, PI_BANDH * 2 / 32);
    }
    _mm_sfence();
}

static inline float pi_tri(float x)
{
    x = x < 0 ? -x : x;
    return x < 1.0f ? 1.0f - x : 0.0f;
}

// PixAI axis (RescalePadProcessor): content [0, len) resized to nl output pixels placed at [lo, lo + nl) (PIL
// bilinear, antialiased: triangle of support max(scale, 1)), pad elsewhere. Output i reads content [s[i], s[i] + n[i])
// with weights w[i * mt..] scaled by sc, plus k[i] (add inside the content, pad outside).
static inline void pi_axis_px(uint32_t S, uint32_t len, uint32_t nl, uint32_t lo, uint32_t mt, float sc, float add,
                              float pad, float *w, int32_t *s, int32_t *n, float *k)
{
    float scale, fs, ss, center, sum, wt, *wi;
    int32_t i, j, a, b;

    scale = (float)len / (float)nl;
    fs = scale > 1.0f ? scale : 1.0f;
    ss = 1.0f / fs;
    for (i = 0; i < (int32_t)S; i++) {
        s[i] = n[i] = 0;
        k[i] = pad;
        if (i < (int32_t)lo || i >= (int32_t)(lo + nl)) continue;
        wi = w + (size_t)i * mt;
        center = ((float)(i - (int32_t)lo) + 0.5f) * scale;
        a = (int32_t)(center - fs + 0.5f);
        a = a < 0 ? 0 : a;
        b = (int32_t)(center + fs + 0.5f);
        b = b > (int32_t)len ? (int32_t)len : b;
        for (j = a, sum = 0.0f; j < b; j++) {
            wt = pi_tri(((float)j - center + 0.5f) * ss);
            sum += wt;
            wi[j - a] = wt;
        }
        for (j = 0, sum = sc / sum; j < b - a; j++) wi[j] *= sum;
        s[i] = a;
        n[i] = b - a;
        k[i] = add;
    }
}

// Crop (x0, y0, cw, ch) -> one PixAI input (PX_IN halves, window-major patches) at out: aspect-preserving resize to
// 1008 (int(len * r) like the reference), centred on black, RGB, [-1, 1]. Each 14-row band of 72 patches leaves as
// 3 streamed runs of 24, one into each window of that band row.
static inline void pi_prep_px(pi_tagger *t, uint32_t x0, uint32_t y0, uint32_t cw, uint32_t ch, uint16_t *out)
{
    const uint8_t *src;
    double r;
    uint32_t S, side, mt, i, n8, nw, nh, py, base;
    size_t need;

    S = PI_SMAX;
    side = cw > ch ? cw : ch;
    mt = 2 * (uint32_t)(2.0f * (side > S ? (float)side / (float)S : 1.0f) + 1.0f) + 2;
    if (mt > t->mt_cap) {
        _mm_free(t->wx);
        _mm_free(t->wy);
        t->wx = _mm_malloc((size_t)PI_SMAX * mt * sizeof(float) + 64, 64);
        t->wy = _mm_malloc((size_t)PI_SMAX * mt * sizeof(float) + 64, 64);
        t->mt_cap = mt;
    }
    r = (double)S / (double)ch < (double)S / (double)cw ? (double)S / (double)ch : (double)S / (double)cw;
    nw = (uint32_t)(cw * r);
    nh = (uint32_t)(ch * r);
    nw = nw ? nw : 1;
    nh = nh ? nh : 1;
    n8 = (cw * 4 + 7) >> 3;
    need = (size_t)n8 * 8 + 8;
    PI_GROW(t->tmp, t->tmp_cap, need, float);
    pi_axis_px(S, cw, nw, (S - nw) >> 1, mt, 1.0f / 127.5f, -1.0f, -1.0f, t->wx, t->sx, t->nx, t->kx);
    pi_axis_px(S, ch, nh, (S - nh) >> 1, mt, 1.0f, 0.0f, 0.0f, t->wy, t->sy, t->ny, t->ky);
    src = t->rgbx + (size_t)y0 * t->stride + (size_t)x0 * 4;
    for (i = 0; i < S; i++) {
        pi_vrow(src, t->stride, n8, t->wy + (size_t)i * mt, t->sy[i], t->ny[i], t->ky[i], t->tmp);
        pi_hrow(t->tmp, S, mt, t->wx, t->sx, t->nx, t->kx, t->jo, PI_RGB, t->band + (i % 14) * 14);
        if (i % 14 != 13) continue;
        py = i / 14;
        base = (py / 24) * 3 * 576 + (py % 24) * 24;
        pi_stream(out + (size_t)base * EVA_K0P, t->band, PI_PXRUN * 2 / 32);
        pi_stream(out + (size_t)(base + 576) * EVA_K0P, t->band + PI_PXRUN, PI_PXRUN * 2 / 32);
        pi_stream(out + (size_t)(base + 1152) * EVA_K0P, t->band + 2 * PI_PXRUN, PI_PXRUN * 2 / 32);
    }
    _mm_sfence();
}

// Mean tile popcount / 8 over a rect.

// Mean tile popcount / 8 over a rect.
static inline float pi_rect_var(const uint8_t *tiles, uint32_t pitch, pi_rect q)
{
    uint32_t r, c, sum;

    for (r = q.r, sum = 0; r < q.r + q.h; r++)
        for (c = q.c; c < q.c + q.w; c++) sum += (uint32_t)__builtin_popcount(tiles[r * pitch + c]);
    return (float)sum / (8.0f * q.h * q.w);
}

// Fold + select. Returns the MVI count; 0 means a single full-image pass. Also leaves the grid statistics the
// router reads (tvar, req).
static inline uint32_t pi_find_mvis(pi_tagger *t, uint32_t w, uint32_t h, pi_result *res)
{
    pi_sel *s;
    uint64_t v, colmask, hisel;
    uint32_t ts, rows, cols, r, n, words, small, nreq;

    ts = t->prm.ts;
    rows = (h + ts - 1) / ts;
    cols = (w + ts - 1) / ts;
    res->uncovered = 0;
    res->ts = ts;
    res->tvar = -1.0f;
    res->req = 0.0f;
    small = rows < t->prm.minh || cols < t->prm.minw || rows > PI_MAX_GRID || cols > PI_MAX_GRID
          || w < 8 || (w <= t->S && h <= t->S);
    if (small || t->prm.nocrops) return 0;

    xf_plan_init(t->plan, t->x, w, h, t->stride, ts, 4, 1);
    xf_scan_rgba(t->x, t->plan, t->rgbx, t->rgbx, t->tiles, t->mask);

    s = t->sel;
    s->rows = rows;
    s->cols = cols;
    s->minw = t->prm.minw;
    s->minh = t->prm.minh;
    s->maxw = t->prm.maxw ? t->prm.maxw : cols;
    s->maxh = t->prm.maxh ? t->prm.maxh : rows;
    words = t->plan->words;
    hisel = words > 1 ? ~0ull : 0;
    colmask = ~0ull >> (64 - cols);
    for (r = 0, nreq = 0; r < rows; r++) {
        v = (t->mask[r * words] | (((uint64_t)t->mask[r * words + words - 1] << 32) & hisel)) & colmask;
        s->allow[r] = t->prm.mode == PI_ZERO ? ~v & colmask : t->prm.mode == PI_DONTCARE ? colmask : v;
        s->req[r] = t->prm.mode == PI_INVERSE ? v : ~v & colmask;
        nreq += (uint32_t)__builtin_popcountll(s->req[r]);
    }
    res->tvar = pi_rect_var(t->tiles, t->plan->pitch, (pi_rect){0, 0, (uint8_t)rows, (uint8_t)cols});
    res->req = (float)nreq / (float)(rows * cols);
    pi_primes(s);
    n = pi_cover(s, res->mvi, t->maxmvi, &res->uncovered);
    if (n == 1 && res->mvi[0].h == rows && res->mvi[0].w == cols) return 0;
    for (r = 0; r < n; r++) {
        res->weight[r] = pi_rect_var(t->tiles, t->plan->pitch, res->mvi[r]);
        res->area[r] = (float)(res->mvi[r].w * res->mvi[r].h) / (float)(rows * cols);
    }
    return n;
}

// Pixel geometry of MVI m (tiles of ts, clipped to the image).
#define PI_CROPGEO(m, ts, iw, ih, x0, y0, cw, ch)                                       \
    ((x0) = (m)->c * (ts), (y0) = (m)->r * (ts),                                        \
     (cw) = ((m)->c + (m)->w) * (ts) < (iw) ? (m)->w * (ts) : (iw) - (x0),              \
     (ch) = ((m)->r + (m)->h) * (ts) < (ih) ? (m)->h * (ts) : (ih) - (y0))

// Exact crop gate: 1 if the crops could flip some non-rating decision of vocabulary v and must run. Crop outputs
// lie in [0, 1], so the crop terms move a score by at most up = sum c_i > 0 upward and dn = sum -c_i > 0 downward:
// with d = full - thr, a decision can only flip inside -up <= d <= dn. (Clamping to [0, 1] never moves a score
// across a threshold in (0, 1].)
static inline uint32_t pi_crops_matter(const pi_tagger *t, const pi_voc *v, const float *full, const pi_result *res)
{
    __m256 vu, vd, d, zero;
    float up, dn, c;
    uint32_t i, j;

    for (i = 0, up = dn = 0.0f; i < res->nmvi; i++) {
        c = t->prm.gain * (res->weight[i] - t->prm.weight_thresh);
        up += c > 0.0f ? c : 0.0f;
        dn += c < 0.0f ? -c : 0.0f;
    }
    vu = _mm256_set1_ps(-up);
    vd = _mm256_set1_ps(dn);
    zero = _mm256_setzero_ps();
    for (j = 0; j + 8 <= v->nt; j += 8) {
        d = _mm256_sub_ps(_mm256_loadu_ps(full + j), _mm256_loadu_ps(v->thr + j));
        if (_mm256_movemask_ps(_mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(d, vu, _CMP_GE_OQ),
                                                           _mm256_cmp_ps(d, vd, _CMP_LE_OQ)),
                                             _mm256_cmp_ps(_mm256_loadu_ps(v->mvimask + j), zero, _CMP_GT_OQ))))
            return 1;
    }
    for (; j < v->nt; j++) {
        c = full[j] - v->thr[j];
        if (v->mvimask[j] > 0.0f && c >= -up && c <= dn) return 1;
    }
    return 0;
}

// final = full + sum_i c_i * crop_i over the crops in runm, ratings masked out of the crop terms; clamp [0, 1].
static inline void pi_combine(const pi_tagger *t, const pi_voc *v, const float *out, const pi_result *res,
                              uint32_t runm, float *score)
{
    const float *p;
    __m256 cv, lo, hi;
    float c;
    uint32_t i, j, nt;

    nt = v->nt;
    memcpy(score, out, nt * sizeof(float));
    for (i = 0; i < res->nmvi; i++) {
        if (!(runm >> i & 1)) continue;
        p = out + (size_t)(i + 1) * nt;
        c = t->prm.gain * (res->weight[i] - t->prm.weight_thresh);
        cv = _mm256_set1_ps(c);
        for (j = 0; j + 8 <= nt; j += 8)
            _mm256_storeu_ps(score + j, _mm256_fmadd_ps(_mm256_mul_ps(cv, _mm256_loadu_ps(v->mvimask + j)),
                                                        _mm256_loadu_ps(p + j), _mm256_loadu_ps(score + j)));
        for (; j < nt; j++) score[j] += c * v->mvimask[j] * p[j];
    }
    lo = _mm256_setzero_ps();
    hi = _mm256_set1_ps(1.0f);
    for (j = 0; j + 8 <= nt; j += 8)
        _mm256_storeu_ps(score + j, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(score + j), lo), hi));
    for (; j < nt; j++) score[j] = score[j] < 0.0f ? 0.0f : score[j] > 1.0f ? 1.0f : score[j];
}

// A crop set without its model's full pass: max over the crops in runm, ratings 0.
static inline void pi_cropmax(const pi_voc *v, const float *out, uint32_t nmvi, uint32_t runm, float *score)
{
    const float *p;
    uint32_t i, j, nt;

    nt = v->nt;
    memset(score, 0, nt * sizeof(float));
    for (i = 0; i < nmvi; i++) {
        if (!(runm >> i & 1)) continue;
        p = out + (size_t)(i + 1) * nt;
        for (j = 0; j + 8 <= nt; j += 8)
            _mm256_storeu_ps(score + j, _mm256_max_ps(_mm256_loadu_ps(score + j),
                                                      _mm256_mul_ps(_mm256_loadu_ps(p + j), _mm256_loadu_ps(v->mvimask + j))));
        for (; j < nt; j++) score[j] = score[j] > p[j] * v->mvimask[j] ? score[j] : p[j] * v->mvimask[j];
    }
}

// Open-addressing set of name hashes (never 0), PI_OH slots, filled to at most half.
static inline uint32_t pi_hput(uint32_t *tab, uint32_t h)
{
    uint32_t k;

    for (k = h & (PI_OH - 1); tab[k]; k = (k + 1) & (PI_OH - 1))
        if (tab[k] == h) return 0;
    tab[k] = h;
    return 1;
}

static inline uint32_t pi_hhas(const uint32_t *tab, uint32_t h)
{
    uint32_t k;

    for (k = h & (PI_OH - 1); tab[k]; k = (k + 1) & (PI_OH - 1))
        if (tab[k] == h) return 1;
    return 0;
}

// Indices of v's tags over threshold, 8 compares per movemask, set bits walked with tzcnt / blsr.
#define PI_OVERTHR(v, score, j, k, m, BODY)                                                \
    do {                                                                                \
        for ((j) = 0; (j) + 8 <= (v)->nt; (j) += 8)                                     \
            for ((m) = (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps((score) + (j)), \
                                         _mm256_loadu_ps((v)->thr + (j)), _CMP_GE_OQ)); (m); (m) &= (m) - 1) { \
                (k) = (j) + (uint32_t)__builtin_ctz(m);                                 \
                BODY;                                                                   \
            }                                                                           \
        for ((k) = (j); (k) < (v)->nt; (k)++)                                           \
            if ((score)[k] >= (v)->thr[k]) { BODY; }                                    \
    } while (0)

// Append v's tags over threshold to the output, skipping names already emitted.
static inline void pi_emit(pi_tagger *t, const pi_voc *v, const float *score, pi_result *res)
{
    uint32_t j, k, m;

    PI_OVERTHR(v, score, j, k, m,
            if (res->n >= PI_OH / 2 || pi_hput(t->oh, v->hash[k])) (t->otag[res->n++] = (pi_tag){v->name[k], score[k]}));
}

static inline const char *pi_rating(const pi_voc *v, const float *score)
{
    uint32_t i, best;

    for (i = 1, best = v->rix[0]; i < v->nr; i++) best = score[v->rix[i]] > score[best] ? v->rix[i] : best;
    return v->nr ? v->name[best] : "";
}

// ---------------------------------------------------------------- vocabularies (mapped)
static inline void pi_unmap(pi_mapping *mp)
{
    if (mp->p) UnmapViewOfFile(mp->p);
    if (mp->m) CloseHandle(mp->m);
    if (mp->f) CloseHandle(mp->f);
    memset(mp, 0, sizeof(*mp));
}

// Whole file as a view: read-only, or copy-on-write (writes stay private to the process). NULL if missing or empty.
static inline uint8_t *pi_map(pi_mapping *mp, const char *path, uint32_t cow)
{
    LARGE_INTEGER sz;

    memset(mp, 0, sizeof(*mp));
    mp->f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (mp->f == INVALID_HANDLE_VALUE) {
        mp->f = NULL;
        return NULL;
    }
    if (!GetFileSizeEx(mp->f, &sz) || !sz.QuadPart
        || !(mp->m = CreateFileMappingA(mp->f, NULL, cow ? PAGE_WRITECOPY : PAGE_READONLY, 0, 0, NULL))
        || !(mp->p = MapViewOfFile(mp->m, cow ? FILE_MAP_COPY : FILE_MAP_READ, 0, 0, 0))) {
        pi_unmap(mp);
        return NULL;
    }
    mp->n = (size_t)sz.QuadPart;
    return mp->p;
}

static inline uint32_t pi_hash(const char *s)
{
    uint64_t v;
    size_t n, i;
    uint32_t h;

    n = strlen(s);
    for (i = 0, h = 0xFFFFFFFFu; i + 8 <= n; i += 8) {
        memcpy(&v, s + i, 8);
        h = (uint32_t)_mm_crc32_u64(h, v);
    }
    for (; i < n; i++) h = _mm_crc32_u8(h, (uint8_t)s[i]);
    return h ? h : 1;
}

// Lines of the mapped view; each line is terminated in place (copy-on-write). The byte after a final line without
// a newline is the zero fill of the view's last page, unless the file ends exactly on a page.
#define PI_LINES(v, line, nl, end, BODY)                                                \
    for ((line) = (char *)(v)->map.p, (end) = (line) + (v)->map.n; (line) < (end); (line) = (nl) + 1) { \
        (nl) = memchr((line), '\n', (size_t)((end) - (line)));                          \
        if (!(nl)) {                                                                    \
            if (!((v)->map.n & 4095)) break;                                            \
            (nl) = (end);                                                               \
        }                                                                               \
        *(nl) = 0;                                                                      \
        if ((nl) > (line) && (nl)[-1] == '\r') (nl)[-1] = 0;                            \
        BODY;                                                                           \
    }

static inline int pi_voc_alloc(pi_voc *v, uint32_t n)
{
    v->name = malloc(sizeof(char *) * n);
    v->cat = malloc(n);
    v->shared = calloc(n, 1);
    v->thr = malloc(n * sizeof(float) + 32);
    v->mvimask = malloc(n * sizeof(float) + 32);
    v->chr = malloc(n * sizeof(float) + 32);
    v->hash = malloc(n * sizeof(uint32_t));
    return v->name && v->cat && v->shared && v->thr && v->mvimask && v->chr && v->hash ? 0 : -1;
}

// Per-tag tables once the names and categories are in: hash, masks, ratings.
static inline void pi_voc_finish(pi_voc *v)
{
    uint32_t j;

    for (j = 0, v->nr = 0; j < v->nt; j++) {
        v->hash[j] = pi_hash(v->name[j]);
        v->mvimask[j] = v->cat[j] == 9 ? 0.0f : 1.0f;
        v->chr[j] = v->cat[j] == 4 ? 1.0f : 0.0f;
        if (v->cat[j] == 9 && v->nr < 8) v->rix[v->nr++] = j;
    }
}

// selected_tags.csv: tag_id,name,category,count (names may hold commas: split on first and last two).
static inline int pi_load_csv(pi_voc *v, const char *path, const pi_params *prm)
{
    char *line, *nl, *end, *c1, *c2, *c3;
    uint32_t n, hdr;

    if (!pi_map(&v->map, path, 1)) return -1;
    for (n = 1, line = (char *)v->map.p; (line = memchr(line, '\n', v->map.n - (size_t)(line - (char *)v->map.p))); line++) n++;
    if (pi_voc_alloc(v, n)) return -1;
    hdr = 1;
    v->nt = 0;
    PI_LINES(v, line, nl, end, {
        if (hdr) { hdr = 0; continue; }
        c1 = strchr(line, ',');
        c3 = strrchr(line, ',');
        if (!c1 || c3 == c1) continue;
        *c3 = 0;
        c2 = strrchr(line, ',');
        *c2 = 0;
        v->name[v->nt] = c1 + 1;
        v->cat[v->nt] = (uint8_t)atoi(c2 + 1);
        v->thr[v->nt] = v->cat[v->nt] == 9 ? 2.0f : v->cat[v->nt] == 4 ? prm->char_thresh : prm->gen_thresh;
        v->nt++;
    });
    pi_voc_finish(v);
    return v->nt ? 0 : -1;
}

// <pixai.bin>.tags (export_pixai.py): name \t category \t threshold, decided with a strict >.
static inline int pi_load_tags(pi_voc *v, const char *path)
{
    char *line, *nl, *end, *t1, *t2;
    uint32_t n;

    if (!pi_map(&v->map, path, 1)) return -1;
    for (n = 1, line = (char *)v->map.p; (line = memchr(line, '\n', v->map.n - (size_t)(line - (char *)v->map.p))); line++) n++;
    if (pi_voc_alloc(v, n)) return -1;
    v->nt = 0;
    PI_LINES(v, line, nl, end, {
        t1 = strchr(line, '\t');
        t2 = t1 ? strchr(t1 + 1, '\t') : NULL;
        if (!t2) continue;
        *t1 = *t2 = 0;
        v->name[v->nt] = line;
        v->cat[v->nt] = (uint8_t)atoi(t1 + 1);
        v->thr[v->nt] = v->cat[v->nt] == 9 ? 2.0f : nextafterf((float)atof(t2 + 1), 2.0f);
        v->nt++;
    });
    pi_voc_finish(v);
    return v->nt ? 0 : -1;
}

// Mark names both vocabularies hold (one hash table of b, probed with a).
static inline int pi_shared(pi_voc *a, pi_voc *b)
{
    uint32_t *tab, m, i, k;

    m = 1u << (32 - __builtin_clz(2 * b->nt));
    tab = calloc(m, sizeof(uint32_t));
    if (!tab) return -1;
    for (i = 0; i < b->nt; i++) {
        for (k = b->hash[i] & (m - 1); tab[k]; k = (k + 1) & (m - 1));
        tab[k] = i + 1;
    }
    for (i = 0; i < a->nt; i++)
        for (k = a->hash[i] & (m - 1); tab[k]; k = (k + 1) & (m - 1))
            if (b->hash[tab[k] - 1] == a->hash[i]) a->shared[i] = b->shared[tab[k] - 1] = 1;
    free(tab);
    return 0;
}

static inline void pi_voc_free(pi_voc *v)
{
    free(v->name); free(v->cat); free(v->shared); free(v->thr); free(v->mvimask); free(v->chr); free(v->hash);
    pi_unmap(&v->map);
}

// ---------------------------------------------------------------- router
// Stage 1 features (quantized, hashed by hp.h):
//  0 bias   1, 2 4 log2 w, h   3 aspect   4 nmvi   5 crop area   6 mean MVI weight   7, 8 crop push up / down
//  9 uncovered tiles   10 alpha   11 megapixels x nmvi   12 grid mean tile popcount   13 share of required tiles
static inline void pi_feat1(const pi_tagger *t, const pi_job *jb, uint32_t *x)
{
    const pi_result *r;
    float up, dn, c, as, ws;
    uint32_t i;

    r = &jb->res;
    for (i = 0, up = dn = as = ws = 0.0f; i < r->nmvi; i++) {
        c = t->prm.gain * (r->weight[i] - t->prm.weight_thresh);
        up += c > 0.0f ? c : 0.0f;
        dn += c < 0.0f ? -c : 0.0f;
        as += r->area[i];
        ws += r->weight[i];
    }
    x[0] = 0;
    x[1] = HP_LOG4(jb->w);
    x[2] = HP_LOG4(jb->h);
    x[3] = x[1] + 128 - x[2];
    x[4] = r->nmvi;
    x[5] = (uint32_t)(16.0f * as);
    x[6] = r->nmvi ? (uint32_t)(16.0f * ws / (float)r->nmvi) : 255;
    x[7] = (uint32_t)(8.0f * up);
    x[8] = (uint32_t)(8.0f * dn);
    x[9] = r->uncovered < 31 ? r->uncovered : 31;
    x[10] = jb->alpha;
    x[11] = HP_LOG4((uint32_t)(((uint64_t)jb->w * jb->h) >> 10)) << 5 | (r->nmvi < 31 ? r->nmvi : 31);
    x[12] = r->tvar < 0.0f ? 255 : (uint32_t)(32.0f * r->tvar);
    x[13] = r->tvar < 0.0f ? 255 : (uint32_t)(16.0f * r->req);
}

// Stage 2 adds the wd full pass: 14 tags within 0.1 of threshold   15 tags over   16 character tags over
// 17 top character score   18 rating   19 exact crop gate   20 tags within 0.1 under   21 top score
static inline void pi_feat2(const pi_tagger *t, const pi_job *jb, uint32_t matter, uint32_t *x)
{
    const pi_voc *v;
    const float *f;
    __m256 s, d, ms, mc, vnear, vnn, zero, mx, mxc;
    __m256i cn, ca, cc, cb;
    int32_t c[4][8];
    float m1[8], m2[8], top, tc, e;
    uint32_t j, k, best;

    pi_feat1(t, jb, x);
    v = &t->wv;
    f = jb->out;
    vnear = _mm256_set1_ps(0.1f);
    vnn = _mm256_set1_ps(-0.1f);
    zero = _mm256_setzero_ps();
    mx = mxc = zero;
    cn = ca = cc = cb = _mm256_setzero_si256();
    for (j = 0; j + 8 <= v->nt; j += 8) {
        s = _mm256_loadu_ps(f + j);
        d = _mm256_sub_ps(s, _mm256_loadu_ps(v->thr + j));
        ms = _mm256_cmp_ps(_mm256_loadu_ps(v->mvimask + j), zero, _CMP_GT_OQ);
        mc = _mm256_cmp_ps(_mm256_loadu_ps(v->chr + j), zero, _CMP_GT_OQ);
        cn = _mm256_sub_epi32(cn, _mm256_castps_si256(_mm256_and_ps(ms, _mm256_and_ps(_mm256_cmp_ps(d, vnn, _CMP_GE_OQ),
                                                                                     _mm256_cmp_ps(d, vnear, _CMP_LT_OQ)))));
        ca = _mm256_sub_epi32(ca, _mm256_castps_si256(_mm256_and_ps(ms, _mm256_cmp_ps(d, zero, _CMP_GE_OQ))));
        cc = _mm256_sub_epi32(cc, _mm256_castps_si256(_mm256_and_ps(mc, _mm256_cmp_ps(d, zero, _CMP_GE_OQ))));
        cb = _mm256_sub_epi32(cb, _mm256_castps_si256(_mm256_and_ps(ms, _mm256_and_ps(_mm256_cmp_ps(d, vnn, _CMP_GE_OQ),
                                                                                     _mm256_cmp_ps(d, zero, _CMP_LT_OQ)))));
        mx = _mm256_max_ps(mx, _mm256_and_ps(ms, s));
        mxc = _mm256_max_ps(mxc, _mm256_and_ps(mc, s));
    }
    _mm256_storeu_si256((__m256i *)c[0], cn);
    _mm256_storeu_si256((__m256i *)c[1], ca);
    _mm256_storeu_si256((__m256i *)c[2], cc);
    _mm256_storeu_si256((__m256i *)c[3], cb);
    _mm256_storeu_ps(m1, mx);
    _mm256_storeu_ps(m2, mxc);
    for (k = 1, top = m1[0], tc = m2[0]; k < 8; k++) {
        c[0][0] += c[0][k];
        c[1][0] += c[1][k];
        c[2][0] += c[2][k];
        c[3][0] += c[3][k];
        top = m1[k] > top ? m1[k] : top;
        tc = m2[k] > tc ? m2[k] : tc;
    }
    for (; j < v->nt; j++) {
        if (v->mvimask[j] <= 0.0f) continue;
        e = f[j] - v->thr[j];
        c[0][0] += e >= -0.1f && e < 0.1f;
        c[1][0] += e >= 0.0f;
        c[2][0] += e >= 0.0f && v->chr[j] > 0.0f;
        c[3][0] += e >= -0.1f && e < 0.0f;
        top = f[j] > top ? f[j] : top;
        tc = v->chr[j] > 0.0f && f[j] > tc ? f[j] : tc;
    }
    for (k = 1, best = 0; k < v->nr; k++) best = f[v->rix[k]] > f[v->rix[best]] ? k : best;
    x[14] = HP_LOG4((uint32_t)c[0][0] + 1);
    x[15] = HP_LOG4((uint32_t)c[1][0] + 1);
    x[16] = c[2][0] < 7 ? (uint32_t)c[2][0] : 7;
    x[17] = (uint32_t)(16.0f * tc);
    x[18] = best;
    x[19] = matter;
    x[20] = HP_LOG4((uint32_t)c[3][0] + 1);
    x[21] = (uint32_t)(16.0f * top);
}

// Cheapest route the perceptron trusts (ok bit 0 D, 1 A, 2 B), else C. Cost order for n crops at wd / PixAI
// per-input cost: D < A (PixAI once) <= B (PixAI n times) < C.
static inline uint32_t pi_pick(uint32_t ok)
{
    return ok & 1 ? PI_D : ok & 2 ? PI_A : ok & 4 ? PI_B : PI_C;
}

// Stage 1, at push. Explores with probability max(explore, PI_WARM / (PI_WARM + routed)); the draw is one AES round
// of a counter.
static inline uint32_t pi_route1(pi_tagger *t, pi_job *jb)
{
    uint32_t x[PI_F1];
    float p;

    if (!t->haspx) return PI_D;
    if (t->prm.route >= 0) return (uint32_t)t->prm.route;
    pi_feat1(t, jb, x);
    hp_index(&t->r1, x, jb->i1);
    p = (float)PI_WARM / (float)(PI_WARM + t->routed++);
    p = p > t->prm.explore ? p : t->prm.explore;
    t->rng = _mm_aesenc_si128(_mm_add_epi64(t->rng, _mm_set_epi64x(0, 1)), t->rkey);
    if ((float)(uint32_t)_mm_cvtsi128_si32(t->rng) < p * 4294967296.0f) {
        jb->explore = 1;
        return PI_X;
    }
    return pi_pick(hp_ge(hp_sum(&t->r1, jb->i1), t->c1));
}

// Route C's tags against a candidate's (a list from one or two score vectors), over names both models know:
// 1 if F1 >= agree. C's set is already in t->ch (nc names).
#define PI_SETADD(v, score, tab, n, j, k, m)                                            \
    PI_OVERTHR(v, score, j, k, m, if ((v)->shared[k] && (n) < PI_OH / 2) (n) += pi_hput(tab, (v)->hash[k]))
static inline uint32_t pi_agree(pi_tagger *t, uint32_t nc, const pi_voc *va, const float *sa, const pi_voc *vb,
                                const float *sb)
{
    uint32_t n, hit, j, k, m;

    memset(t->uh, 0, PI_OH * sizeof(uint32_t));
    n = 0;
    PI_SETADD(va, sa, t->uh, n, j, k, m);
    if (vb) PI_SETADD(vb, sb, t->uh, n, j, k, m);
    for (k = 0, hit = 0; k < PI_OH; k++) hit += t->uh[k] && pi_hhas(t->ch, t->uh[k]);
    return n + nc == 0 || 2.0f * (float)hit >= t->prm.agree * (float)(n + nc);
}

// An explored job is back with every component: label routes D, A, B against C (t->sc[1] holds C's scores) and
// train both stages. Statistics count what each stage predicted before learning.
static inline void pi_learn(pi_tagger *t, pi_job *jb)
{
    const pi_result *r;
    uint32_t nc, j, k, m, lab, o;

    r = &jb->res;
    memset(t->ch, 0, PI_OH * sizeof(uint32_t));
    nc = 0;
    PI_SETADD(&t->pv, t->sc[1], t->ch, nc, j, k, m);
    pi_combine(t, &t->wv, jb->out, r, r->runm, t->sc[0]);
    pi_cropmax(&t->wv, jb->out, r->nmvi, r->runm, t->sc[2]);
    pi_cropmax(&t->pv, jb->pout, r->nmvi, r->prun, t->sc[3]);
    lab = pi_agree(t, nc, &t->wv, t->sc[0], NULL, NULL)
        | pi_agree(t, nc, &t->pv, jb->pout, &t->wv, t->sc[2]) << 1
        | pi_agree(t, nc, &t->wv, jb->out, &t->pv, t->sc[3]) << 2;
    m = hp_ge(hp_sum(&t->r1, jb->i1), t->c1);
    k = hp_ge(hp_sum(&t->r2, jb->i2), t->c2);
    for (o = 0; o < 3; o++) {
        t->nok[o] += lab >> o & 1;
        t->hit1[o] += (m >> o & 1) == (lab >> o & 1);
        t->hit2[o] += (k >> o & 1) == (lab >> o & 1);
    }
    hp_train(&t->r1, jb->i1, lab, 7);
    hp_train(&t->r2, jb->i2, lab, 7);
}

// ---------------------------------------------------------------- scratch / open
static inline int pi_scratch(pi_tagger *t, const pi_params *prm)
{
    LARGE_INTEGER q;
    pi_job *jb;
    uint32_t j, ntmax;

    t->prm = *prm;
    t->S = EVA_S;
    t->maxmvi = prm->maxmvi < 30 ? prm->maxmvi : 30; // crop bits share a u32 with the full image
    t->owner = PI_JOBS;
    t->x = _mm_malloc(sizeof(xf_ctx), 64);
    t->plan = _mm_malloc(sizeof(xf_plan), 64);
    t->sel = _mm_malloc(sizeof(pi_sel), 64);
    xf_init(t->x, prm->lo, prm->hi);
    t->kx = malloc(PI_SMAX * sizeof(float));
    t->ky = malloc(PI_SMAX * sizeof(float));
    t->sx = malloc(PI_SMAX * sizeof(int32_t));
    t->nx = malloc(PI_SMAX * sizeof(int32_t));
    t->sy = malloc(PI_SMAX * sizeof(int32_t));
    t->ny = malloc(PI_SMAX * sizeof(int32_t));
    t->jo = malloc(PI_SMAX * sizeof(uint32_t));
    for (j = 0; j < PI_SMAX; j++) t->jo[j] = (j / 14) * EVA_K0P + j % 14;
    t->band = _mm_malloc(3 * PI_PXRUN * 2, 64);
    memset(t->band, 0, 3 * PI_PXRUN * 2); // pad columns 588..591 of every patch stay zero forever
    t->poolb = (size_t)PI_JOBS * (1 + t->maxmvi) * EVA_IN * 2;
    t->pool = _mm_malloc(t->poolb, 4096);
    if (!t->band || !t->pool) return -1;
    ntmax = t->wv.nt > t->pv.nt ? t->wv.nt : t->pv.nt;
    for (j = 0; j < 4; j++)
        if (!(t->sc[j] = _mm_malloc(ntmax * sizeof(float) + 32, 64))) return -1;
    t->otag = malloc((t->wv.nt + t->pv.nt) * sizeof(pi_tag));
    t->oh = malloc(3 * PI_OH * sizeof(uint32_t));
    if (!t->otag || !t->oh) return -1;
    t->ch = t->oh + PI_OH;
    t->uh = t->ch + PI_OH;
    for (j = 0; j < PI_JOBS; j++) {
        jb = &t->job[j];
        jb->out = malloc((size_t)(t->maxmvi + 1) * t->wv.nt * sizeof(float));
        jb->in = t->pool + (size_t)j * (1 + t->maxmvi) * EVA_IN;
        if (!jb->out) return -1;
    }
    t->jfree = ~0u >> (32 - PI_JOBS);
    if (!t->haspx) return 0;
    t->ppoolb = (size_t)PI_PXS * PX_IN * 2;
    t->ppool = _mm_malloc(t->ppoolb, 4096);
    if (!t->ppool) return -1;
    t->pfree = ~0u >> (32 - PI_PXS);
    for (j = 0; j < PI_JOBS; j++)
        if (!(t->job[j].pout = malloc((size_t)(t->maxmvi + 1) * t->pv.nt * sizeof(float)))) return -1;
    if (hp_init(&t->r1, PI_F1, PI_HROWS) || hp_init(&t->r2, PI_F2, PI_HROWS)) return -1;
    t->c1 = prm->rconf >= 0 ? prm->rconf : t->r1.theta / 2;
    t->c2 = prm->rconf >= 0 ? prm->rconf : t->r2.theta / 2;
    QueryPerformanceCounter(&q);
    t->rng = _mm_set_epi64x((long long)q.QuadPart, 0x243F6A8885A308D3ll);
    t->rkey = _mm_set_epi64x(0x13198A2E03707344ll, (long long)0xA4093822299F31D0ull);
    return 0;
}

// model: eva02.bin from export_eva.py; prm->px: pixai.bin from export_pixai.py (or NULL). done is called once per
// image, as it completes. Both engines come up on their own threads; pi_open returns at once.
static inline int pi_open(pi_tagger *t, const char *model, const char *csv, const pi_params *prm, pi_done_fn done)
{
    char name[1024];
    int dev;

    memset(t, 0, sizeof(*t));
    t->done = done;
    t->haspx = prm->px != NULL;
    dev = prm->device < 0 ? 0 : prm->device;
    if (pi_load_csv(&t->wv, csv, prm)) {
        fprintf(stderr, "cannot read %s\n", csv);
        return -1;
    }
    if (t->haspx) {
        snprintf(name, sizeof(name), "%s.tags", prm->px);
        if (pi_load_tags(&t->pv, name) || pi_shared(&t->wv, &t->pv)) {
            fprintf(stderr, "cannot read %s\n", name);
            return -1;
        }
    }
    if (pi_scratch(t, prm)) return -1;
    if (eva_open(&t->e, model, dev, t->pool, t->poolb)) return -1;
    if (t->e.nt != t->wv.nt) {
        fprintf(stderr, "%s has %u tags, %s has %u\n", model, t->e.nt, csv, t->wv.nt);
        return -1;
    }
    if (t->haspx && px_open(&t->p, prm->px, dev, t->ppool, t->ppoolb)) return -1;
    if (t->haspx && t->p.nt != t->pv.nt) {
        fprintf(stderr, "%s has %u tags, %s has %u\n", prm->px, t->p.nt, name, t->pv.nt);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------- scheduler
// wd batches fill a ring of EVA_NBUF host buffers: col <= sub <= batch. While the engine is still coming up, full
// batches just queue; once it is up, everything queued is submitted and one batch stays in flight while the next
// fills. PixAI inputs are prepared into PI_PXS pinned slots and issued PX_NB at a time from the ready FIFO.
#define PI_BUF(b) ((b) & (EVA_NBUF - 1))
// Next slot of the filling wd batch: input i of job q (already in the pool).
#define PI_SLOT(t, q, i)                                                                \
    ((t)->own[PI_BUF((t)->batch)][(t)->fill] = (pi_owner){(uint16_t)(q), (uint16_t)(i)}, \
     (t)->src[PI_BUF((t)->batch)][(t)->fill] = (t)->job[q].in + (size_t)(i) * EVA_IN)
#define PI_COMMIT(t) (++(t)->fill < EVA_NB ? 0 : pi_launch(t))
#define PI_CHECK(t, q, jb)                                                              \
    do {                                                                                \
        if (!(jb)->pend && !(jb)->prep && !(jb)->wait) (t)->fin[(t)->nfin++] = (q);     \
    } while (0)
#define PI_ALL(n) ((1u << (n)) - 1)

// Free the source pixels once no PixAI input can still be prepared from them: nothing left to prepare, no PixAI
// crop gate pending, and no stage-2 decision that could escalate a D job.
static inline void pi_release(pi_tagger *t, pi_job *jb)
{
    if (!jb->rgba || jb->prep || (jb->wait & 2) || ((jb->wait & 1) && t->haspx && jb->route == PI_D)) return;
    free(jb->rgba);
    jb->rgba = NULL;
}

// Stage 2 sent a D job elsewhere; its wd full pass is done, its wd crops are already in the pool.
static inline void pi_escalate(pi_tagger *t, uint32_t q, uint32_t r)
{
    pi_job *jb;
    uint32_t all;

    jb = &t->job[q];
    all = PI_ALL(jb->res.nmvi);
    jb->route = r;
    if (r == PI_A) {
        jb->prep |= 1;
        jb->res.runm = all;
        jb->pend += jb->res.nmvi;
        if (all) t->defer[t->ndefer++] = q;
    } else if (r == PI_B) {
        jb->prep |= all << 1;
        jb->res.prun = all;
    } else {
        jb->prep |= 1;
        jb->wait |= all ? 2 : 0;
    }
}

static inline int pi_wd_idle(pi_tagger *t)
{
    return t->sub == t->col || eva_query(&t->e, PI_BUF(t->sub - 1));
}

// Prepare one PixAI input into a free slot: from the job whose composite is already in rgbx if it has work, else
// an exploring job (its labels train the router), else the lowest. Composites again when the owner changes.
static inline int pi_pxprep(pi_tagger *t)
{
    pi_job *jb;
    pi_rect *m;
    uint32_t q, best, i, slot, ts, x0, y0, cw, ch;

    if (!t->pfree) return 0;
    for (q = 0, best = PI_JOBS; q < PI_JOBS; q++) {
        jb = &t->job[q];
        if (jb->state != PI_LIVE || !jb->prep) continue;
        if (q == t->owner) {
            best = q;
            break;
        }
        best = best == PI_JOBS || (jb->explore && !t->job[best].explore) ? q : best;
    }
    if (best == PI_JOBS) return 0;
    jb = &t->job[best];
    if (t->owner != best) {
        pi_load(t, jb->rgba, jb->w, jb->h, jb->istride, jb->alpha);
        t->owner = best;
    }
    i = (uint32_t)__builtin_ctz(jb->prep);
    slot = (uint32_t)__builtin_ctz(t->pfree);
    t->pfree &= t->pfree - 1;
    ts = t->prm.ts;
    if (!i) {
        pi_prep_px(t, 0, 0, jb->w, jb->h, t->ppool + (size_t)slot * PX_IN);
    } else {
        m = &jb->res.mvi[i - 1];
        PI_CROPGEO(m, ts, jb->w, jb->h, x0, y0, cw, ch);
        pi_prep_px(t, x0, y0, cw, ch, t->ppool + (size_t)slot * PX_IN);
    }
    jb->prep &= jb->prep - 1;
    jb->pend++;
    t->rdy[t->rh++ % PI_PXS] = (pi_pent){(uint16_t)best, (uint16_t)i, (uint16_t)slot};
    pi_release(t, jb);
    return 1;
}

// Issue up to PX_NB prepared inputs. Held (0) while light work is on the GPU and slots remain to prepare into,
// while the engine is still coming up, or below a full batch - unless forced.
static inline int pi_pxsubmit(pi_tagger *t, uint32_t force)
{
    uint16_t *in[PX_NB];
    uint32_t n, b, s;

    n = t->rh - t->rt;
    if (!n || t->psub - t->pcol == PX_NBUF) return 0;
    if (!force && (n < PX_NB || !px_poll(&t->p) || (t->pfree && !pi_wd_idle(t)))) return 0;
    n = n < PX_NB ? n : PX_NB;
    b = t->psub % PX_NBUF;
    for (s = 0; s < n; s++) {
        t->pown[b][s] = t->rdy[t->rt++ % PI_PXS];
        in[s] = t->ppool + (size_t)t->pown[b][s].slot * PX_IN;
    }
    t->pcnt[b] = n;
    if (px_submit(&t->p, b, in, n)) return -1;
    t->psub++;
    return 1;
}

// Oldest PixAI batch done (or block for it): outputs to their jobs, slots freed; a route-C job whose full pass
// landed runs the PixAI crop gate.
static inline int pi_pxgather(pi_tagger *t, uint32_t block)
{
    const float *out;
    pi_pent *e;
    pi_job *jb;
    uint32_t b, s, nt;

    if (t->pcol == t->psub) return 0;
    b = t->pcol % PX_NBUF;
    if (!block && !px_query(&t->p, b)) return 0;
    if (!(out = px_wait(&t->p, b))) return -1;
    nt = t->pv.nt;
    for (s = 0; s < t->pcnt[b]; s++) {
        e = &t->pown[b][s];
        jb = &t->job[e->job];
        memcpy(jb->pout + (size_t)e->idx * nt, out + (size_t)s * nt, nt * sizeof(float));
        t->pfree |= 1u << e->slot;
        jb->pend--;
        if (!e->idx && (jb->wait & 2)) {
            jb->wait &= ~2u;
            if (pi_crops_matter(t, &t->pv, jb->pout, &jb->res)) {
                jb->res.prun = PI_ALL(jb->res.nmvi);
                jb->prep |= jb->res.prun << 1;
            }
            pi_release(t, jb);
        }
        PI_CHECK(t, e->job, jb);
    }
    t->pcol++;
    return 1;
}

// One unit of heavy-side work, never blocking: collect a finished PixAI batch, else prepare an input, else issue.
static inline int pi_step(pi_tagger *t)
{
    int r;

    if (!t->haspx) return 0;
    if ((r = pi_pxgather(t, 0))) return r;
    if ((r = pi_pxprep(t))) return r;
    return pi_pxsubmit(t, 0);
}

// Heavy work when nothing light is left: prepare everything there is room for, issue (partial batches too), and
// block for the oldest PixAI batch. 1 if anything moved.
static inline int pi_pxforce(pi_tagger *t)
{
    int r, moved;

    for (moved = 0; (r = pi_pxprep(t)) > 0; moved = 1);
    if (r < 0 || (r = pi_pxsubmit(t, 1)) < 0) return -1;
    moved |= r;
    if ((r = pi_pxgather(t, 1)) < 0) return -1;
    return moved | r;
}

// Wait for wd batch b, doing heavy-side work while the GPU is busy with it.
static inline const float *pi_wd_wait(pi_tagger *t, uint32_t b)
{
    int r;

    while (!eva_query(&t->e, PI_BUF(b)))
        if ((r = pi_step(t)) <= 0) {
            t->err |= r < 0;
            break;
        }
    return eva_wait(&t->e, PI_BUF(b));
}

// wd batch b is done: scatter outputs to their jobs. A job whose wd full pass just landed takes its decisions: stage
// 2 of the router (D may escalate), the exact wd crop gate (deferred crops), stage-2 features for an exploring job.
static inline int pi_gather(pi_tagger *t, uint32_t b)
{
    const float *out;
    pi_owner *o;
    pi_job *jb;
    uint32_t x[PI_F2];
    uint32_t s, n, nt, matter, r, router;

    if (!(out = pi_wd_wait(t, b))) return -1;
    nt = t->wv.nt;
    n = t->cnt[PI_BUF(b)];
    router = t->haspx && t->prm.route < 0;
    for (s = 0; s < n; s++) {
        o = &t->own[PI_BUF(b)][s];
        jb = &t->job[o->job];
        memcpy(jb->out + (size_t)o->idx * nt, out + (size_t)s * nt, nt * sizeof(float));
        jb->pend--;
        if (!o->idx && (jb->wait & 1)) {
            jb->wait &= ~1u;
            matter = jb->res.nmvi && pi_crops_matter(t, &t->wv, jb->out, &jb->res);
            if (router) {
                pi_feat2(t, jb, matter, x);
                hp_index(&t->r2, x, jb->i2);
            }
            if (jb->route == PI_D) {
                r = router && !jb->explore ? pi_pick(hp_ge(hp_sum(&t->r2, jb->i2), t->c2)) : PI_D;
                if (r != PI_D) {
                    pi_escalate(t, o->job, r);
                } else if (matter) {
                    jb->res.runm = PI_ALL(jb->res.nmvi);
                    jb->pend += jb->res.nmvi;
                    t->defer[t->ndefer++] = o->job;
                }
            }
            pi_release(t, jb);
        }
        PI_CHECK(t, o->job, jb);
    }
    return 0;
}

// Submit every queued wd batch (blocks only if the engine is not up yet).
static inline int pi_submit(pi_tagger *t)
{
    for (; t->sub != t->batch; t->sub++)
        if (eva_submit(&t->e, PI_BUF(t->sub), t->src[PI_BUF(t->sub)], t->cnt[PI_BUF(t->sub)])) return -1;
    return 0;
}

// Queue the filling wd batch. Engine up: submit, and collect all but the newest in flight. Ring full: wait for the
// engine and free the oldest buffer.
static inline int pi_launch(pi_tagger *t)
{
    t->cnt[PI_BUF(t->batch)] = t->fill;
    t->batch++;
    t->fill = 0;
    if (eva_poll(&t->e) || t->batch - t->col == EVA_NBUF) {
        if (pi_submit(t)) return -1;
        while (t->sub - t->col > 1 || t->batch - t->col == EVA_NBUF)
            if (pi_gather(t, t->col++)) return -1;
    }
    return 0;
}

// Nothing more light is coming for now: launch a partial wd batch and collect everything in flight.
static inline int pi_settle(pi_tagger *t)
{
    if (t->fill && pi_launch(t)) return -1;
    if (pi_submit(t)) return -1;
    while (t->col != t->sub)
        if (pi_gather(t, t->col++)) return -1;
    return 0;
}

// A job has everything: combine per route, train the router on an exploring job, report, free.
static inline void pi_finish(pi_tagger *t, uint32_t q)
{
    pi_job *jb;
    pi_result *r;
    const float *ws, *ps;

    jb = &t->job[q];
    r = &jb->res;
    ws = ps = NULL;
    switch (jb->route) {
    case PI_D:
        pi_combine(t, &t->wv, jb->out, r, r->runm, t->sc[0]);
        ws = t->sc[0];
        r->rating = pi_rating(&t->wv, jb->out);
        break;
    case PI_A:
        pi_cropmax(&t->wv, jb->out, r->nmvi, r->runm, t->sc[0]);
        ps = jb->pout;
        ws = t->sc[0];
        r->rating = pi_rating(&t->pv, jb->pout);
        break;
    case PI_B:
        pi_cropmax(&t->pv, jb->pout, r->nmvi, r->prun, t->sc[1]);
        ws = jb->out;
        ps = t->sc[1];
        r->rating = pi_rating(&t->wv, jb->out);
        break;
    default: // C, X
        pi_combine(t, &t->pv, jb->pout, r, r->prun, t->sc[1]);
        ps = t->sc[1];
        r->rating = pi_rating(&t->pv, jb->pout);
        if (jb->route == PI_X) pi_learn(t, jb); // reads sc[1] (route C), overwrites sc[0], sc[2], sc[3]
    }
    t->nroute[jb->route]++;
    r->route = jb->route;
    r->ncrop = (uint32_t)__builtin_popcount(r->runm) + (uint32_t)__builtin_popcount(r->prun);
    r->n = 0;
    r->tag = t->otag;
    memset(t->oh, 0, PI_OH * sizeof(uint32_t));
    if (ps) pi_emit(t, &t->pv, ps, r);
    if (ws) pi_emit(t, &t->wv, ws, r);
    t->done(t, jb->user, r);
    free(jb->rgba);
    jb->rgba = NULL;
    jb->state = PI_FREE;
    t->jfree |= 1u << q;
    t->live--;
    t->owner = t->owner == q ? PI_JOBS : t->owner;
}

// Queue deferred wd crops, then retire finished jobs.
static inline int pi_pump(pi_tagger *t)
{
    pi_job *jb;
    uint32_t q, i;

    while (t->ndefer) {
        q = t->defer[--t->ndefer];
        jb = &t->job[q];
        for (i = 0; i < jb->res.nmvi; i++) {
            if (!(jb->res.runm >> i & 1)) continue;
            PI_SLOT(t, q, i + 1);
            if (PI_COMMIT(t)) return -1;
        }
    }
    while (t->nfin) pi_finish(t, t->fin[--t->nfin]);
    return t->err ? -1 : 0;
}

// Block until a job slot frees: heavy-side steps first, then flush light work, then force heavy work.
static inline int pi_wait_job(pi_tagger *t)
{
    int r;

    if ((r = pi_step(t)) < 0 || pi_pump(t)) return -1;
    if (t->jfree || r > 0) return 0;
    if (pi_settle(t) || pi_pump(t)) return -1;
    if (t->jfree) return 0;
    if ((r = pi_pxforce(t)) < 0 || pi_pump(t)) return -1;
    if (!r && !t->jfree) {
        fprintf(stderr, "pipeline stalled\n");
        return -1;
    }
    return 0;
}

// Tag one decoded image (RGBA, in_stride bytes per row, malloc'd: the tagger owns and frees it). alpha: 1 if the
// source had an alpha channel. The route is chosen now; wd inputs are prepared and queued now (a D job's crops are
// prepared now and queued only if the full pass says they matter); PixAI inputs are prepared in GPU-wait time.
static inline int pi_push(pi_tagger *t, uint8_t *rgba, uint32_t w, uint32_t h, size_t in_stride, uint32_t alpha,
                          void *user)
{
    pi_job *jb;
    pi_rect *m;
    uint32_t q, i, ts, all, wfull, wcrop, x0, y0, cw, ch;

    while (!t->jfree)
        if (pi_wait_job(t)) return -1;
    q = (uint32_t)__builtin_ctz(t->jfree);
    t->jfree &= t->jfree - 1;
    t->live++;
    jb = &t->job[q];
    ts = t->prm.ts;
    pi_load(t, rgba, w, h, in_stride, alpha);
    t->owner = q;
    jb->rgba = rgba;
    jb->istride = in_stride;
    jb->w = w;
    jb->h = h;
    jb->alpha = alpha;
    jb->user = user;
    jb->state = PI_LIVE;
    jb->explore = 0;
    jb->res.nmvi = pi_find_mvis(t, w, h, &jb->res);
    jb->res.runm = jb->res.prun = 0;
    all = PI_ALL(jb->res.nmvi);
    jb->route = pi_route1(t, jb);
    wfull = jb->route == PI_D || jb->route == PI_B || jb->route == PI_X;
    wcrop = jb->route == PI_A || jb->route == PI_X; // all wd crops now; D's wait for the gate
    jb->prep = jb->route == PI_A || jb->route == PI_C ? 1 : jb->route == PI_B ? all << 1 : jb->route == PI_X ? 1 | all << 1 : 0;
    jb->wait = jb->route == PI_D || jb->route == PI_X ? 1 : jb->route == PI_C && all ? 2 : 0;
    jb->res.runm = wcrop ? all : 0;
    jb->res.prun = jb->route == PI_B || jb->route == PI_X ? all : 0;
    jb->pend = wfull + (wcrop ? jb->res.nmvi : 0);
    if (jb->route == PI_D || wcrop)
        for (i = 0; i < jb->res.nmvi; i++) {
            m = &jb->res.mvi[i];
            PI_CROPGEO(m, ts, w, h, x0, y0, cw, ch);
            pi_prep(t, x0, y0, cw, ch, jb->in + (size_t)(i + 1) * EVA_IN);
        }
    if (wfull) pi_prep(t, 0, 0, w, h, jb->in);
    pi_release(t, jb);
    for (i = 0; wcrop && i < jb->res.nmvi; i++) {
        PI_SLOT(t, q, i + 1);
        if (PI_COMMIT(t)) return -1;
    }
    if (wfull) {
        PI_SLOT(t, q, 0);
        if (PI_COMMIT(t)) return -1;
    }
    return pi_pump(t);
}

// Flush: run everything still queued and report every remaining image.
static inline int pi_drain(pi_tagger *t)
{
    int r;

    while (t->live) {
        if (pi_settle(t) || pi_pump(t)) return -1;
        if (!t->live || t->fill || t->sub != t->col) continue;
        if ((r = pi_pxforce(t)) < 0 || pi_pump(t)) return -1;
        if (!r && t->live && !t->fill) {
            fprintf(stderr, "pipeline stalled\n");
            return -1;
        }
    }
    return 0;
}

static inline void pi_close(pi_tagger *t)
{
    uint32_t j, ne;

    if (t->haspx && t->prm.route < 0 && t->routed) {
        ne = t->nroute[PI_X];
        fprintf(stderr, "router: %u images, routes D %u A %u B %u C %u, explored %u", t->routed, t->nroute[PI_D],
                t->nroute[PI_A], t->nroute[PI_B], t->nroute[PI_C], ne);
        if (ne)
            fprintf(stderr, "; explored agreeing with C: D %.1f%% A %.1f%% B %.1f%%; predicted right before learning:"
                    " stage 1 %.1f/%.1f/%.1f%%, stage 2 %.1f/%.1f/%.1f%%",
                    100.0 * t->nok[0] / ne, 100.0 * t->nok[1] / ne, 100.0 * t->nok[2] / ne,
                    100.0 * t->hit1[0] / ne, 100.0 * t->hit1[1] / ne, 100.0 * t->hit1[2] / ne,
                    100.0 * t->hit2[0] / ne, 100.0 * t->hit2[1] / ne, 100.0 * t->hit2[2] / ne);
        fputc('\n', stderr);
    }
    eva_close(&t->e);
    if (t->haspx) px_close(&t->p);
    for (j = 0; j < PI_JOBS; j++) {
        free(t->job[j].out);
        free(t->job[j].pout);
        free(t->job[j].rgba);
    }
    for (j = 0; j < 4; j++) _mm_free(t->sc[j]);
    hp_free(&t->r1);
    hp_free(&t->r2);
    _mm_free(t->pool);
    _mm_free(t->ppool);
    _mm_free(t->band);
    _mm_free(t->x);
    _mm_free(t->plan);
    _mm_free(t->sel);
    _mm_free(t->rgbx);
    _mm_free(t->tmp);
    _mm_free(t->wx);
    _mm_free(t->wy);
    free(t->kx); free(t->ky); free(t->sx); free(t->nx); free(t->sy); free(t->ny); free(t->jo);
    free(t->otag);
    free(t->oh);
    pi_voc_free(&t->wv);
    pi_voc_free(&t->pv);
}

#endif // PROCIMG_H
