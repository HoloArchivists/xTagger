#ifndef NPU_H
#define NPU_H
// Intel NPU engine (Core Ultra "AI Boost") for wd-eva02, through Level Zero and the driver's graph extension: no
// OpenVINO runtime. The network is written here as an OpenVINO IR (XML + weights) from eva02_fp16.bin (v2, the fp16
// reference weights), compiled by the compiler that ships inside the NPU driver, and the compiled binary is cached
// next to the weights (<weights>.npu) so later runs only load it. ze_loader.dll is loaded at run time: no NPU, no
// import.
//
// Math is the fp16 calibration engine's: patch GEMM, + pos (cls row 0), 24 x [LN1, QKV, RoPE on patch tokens,
// softmax(q k^T ln2) v (q carries scale * log2 e), out-proj, LN2, SwiGLU, LN over 2730, fc2], mean of the 1024 patch
// tokens, LN, head, sigmoid. One image per inference: input = the patch GEMM operand procimg already writes
// (fp16 [1024][592]), output fp32 [nt]. NPU_NBUF inferences in flight, each a recorded command list with its own
// buffers and fence.
//
//   npu_open(n, "eva02_fp16.bin")  ->  npu_free(n) / npu_submit(n, s, input)  ->  npu_done(n, s) / npu_out(n, s)

#include <stdarg.h>
#include <stdbool.h> // ze_graph_ext.h uses bool
#include <ze_api.h>
#include <ze_graph_ext.h>
#include "eva.h"

#define NPU_NBUF  2
#define NPU_HID   2730   // real SwiGLU units (the GPU engines pad to 2736)
#define NPU_MAGIC 0x55504E58u // "XNPU": cache header
#define NPU_GVER  2           // graph version in the cache header: bump whenever npu_eva_ir changes

// Level Zero entry points, resolved from ze_loader.dll at bring-up
#define NPU_ZE(X)                                                                       \
    X(zeInit) X(zeDriverGet) X(zeDeviceGet) X(zeDriverGetExtensionProperties)           \
    X(zeDriverGetExtensionFunctionAddress) X(zeContextCreate) X(zeContextDestroy)       \
    X(zeMemAllocHost) X(zeMemFree) X(zeCommandQueueCreate) X(zeCommandQueueDestroy)     \
    X(zeCommandListCreate) X(zeCommandListDestroy) X(zeCommandListClose)                \
    X(zeCommandQueueExecuteCommandLists) X(zeCommandQueueSynchronize) X(zeFenceCreate)  \
    X(zeFenceDestroy) X(zeFenceQueryStatus) X(zeFenceHostSynchronize) X(zeFenceReset)

typedef struct {
#define NPU_DECL(f) __typeof__(&f) f;
    NPU_ZE(NPU_DECL)
#undef NPU_DECL
} npu_ze;

typedef struct {
    HMODULE dll;
    npu_ze z;
    ze_driver_handle_t drv;
    ze_device_handle_t dev;
    ze_context_handle_t ctx;
    ze_graph_dditable_ext_t *g;
    ze_graph_handle_t graph;
    ze_command_queue_handle_t q;
    ze_command_list_handle_t cl[NPU_NBUF];
    ze_fence_handle_t f[NPU_NBUF];
    void *in[NPU_NBUF], *out[NPU_NBUF];
    HANDLE boot;
    const char *path;
    uint8_t *blob;               // cached native graph: the driver reads it for the graph's lifetime
    uint32_t nt, busy, compiled; // busy: slot bits in flight; compiled: 1 if this run compiled (no cache)
    int bstatus, up;
} npu;

// ---------------------------------------------------------------- IR builder
// Values carry their layer / port and shape; every op appends one <layer> and its input <edge>s. Constants go to the
// weights buffer (32-byte aligned offsets).
typedef struct { uint32_t id, port, rank, d[4]; const char *et; } nv;
typedef struct {
    char *x, *e;
    uint8_t *w;
    size_t xn, xcap, en, ecap, wn, wcap;
    uint32_t id;
    int err;
} nir;

static inline void nir_cat(nir *r, char **b, size_t *n, size_t *cap, const char *fmt, ...)
{
    va_list ap;
    int k;

    for (;;) {
        va_start(ap, fmt);
        k = vsnprintf(*b + *n, *cap - *n, fmt, ap);
        va_end(ap);
        if (k >= 0 && (size_t)k < *cap - *n) break;
        *cap = *cap * 2 + (size_t)(k > 0 ? k : 4096);
        if (!(*b = realloc(*b, *cap))) {
            r->err = 1;
            *n = *cap = 0;
            return;
        }
    }
    *n += (size_t)k;
}

#define NIR_X(r, ...) nir_cat(r, &(r)->x, &(r)->xn, &(r)->xcap, __VA_ARGS__)
#define NIR_E(r, ...) nir_cat(r, &(r)->e, &(r)->en, &(r)->ecap, __VA_ARGS__)
#define NIR_PREC(et) (!strcmp(et, "f16") ? "FP16" : !strcmp(et, "f32") ? "FP32" : "I64")

static inline void nir_port(nir *r, uint32_t port, const nv *v, const char *names)
{
    uint32_t i;

    NIR_X(r, "<port id=\"%u\" precision=\"%s\"%s%s%s>", port, NIR_PREC(v->et), names ? " names=\"" : "",
          names ? names : "", names ? "\"" : "");
    for (i = 0; i < v->rank; i++) NIR_X(r, "<dim>%u</dim>", v->d[i]);
    NIR_X(r, "</port>");
}

static inline void nir_shape(char *s, size_t cap, uint32_t rank, const uint32_t *d)
{
    size_t n;
    uint32_t i;

    for (i = 0, n = 0, s[0] = 0; i < rank && n < cap; i++) n += (size_t)snprintf(s + n, cap - n, i ? ",%u" : "%u", d[i]);
}

static inline nv nir_val(uint32_t id, uint32_t port, const char *et, uint32_t rank, const uint32_t *d)
{
    nv v;

    memset(&v, 0, sizeof(v));
    v.id = id;
    v.port = port;
    v.et = et;
    v.rank = rank;
    memcpy(v.d, d, rank * sizeof(uint32_t));
    return v;
}

static inline nv nir_param(nir *r, const char *name, const char *et, uint32_t rank, const uint32_t *d)
{
    char s[64];
    nv v;

    v = nir_val(r->id++, 0, et, rank, d);
    nir_shape(s, sizeof(s), rank, d);
    NIR_X(r, "<layer id=\"%u\" name=\"%s\" type=\"Parameter\" version=\"opset1\"><data shape=\"%s\" element_type=\"%s\"/>"
          "<output>", v.id, name, s, et);
    nir_port(r, 0, &v, name);
    NIR_X(r, "</output></layer>\n");
    return v;
}

// Constant from raw bytes (src NULL: zero-filled).
static inline nv nir_const(nir *r, const char *et, uint32_t rank, const uint32_t *d, const void *src, size_t bytes)
{
    char s[64];
    size_t off;
    nv v;

    off = (r->wn + 31) & ~(size_t)31;
    if (off + bytes > r->wcap) {
        r->wcap = (off + bytes) * 2;
        if (!(r->w = realloc(r->w, r->wcap))) {
            r->err = 1;
            return nir_val(0, 0, et, rank, d);
        }
    }
    memset(r->w + r->wn, 0, off - r->wn);
    if (src) memcpy(r->w + off, src, bytes);
    else memset(r->w + off, 0, bytes);
    r->wn = off + bytes;
    v = nir_val(r->id++, 0, et, rank, d);
    nir_shape(s, sizeof(s), rank, d);
    NIR_X(r, "<layer id=\"%u\" name=\"c%u\" type=\"Const\" version=\"opset1\"><data element_type=\"%s\" shape=\"%s\" "
          "offset=\"%zu\" size=\"%zu\"/><output>", v.id, v.id, et, s, off, bytes);
    nir_port(r, 0, &v, NULL);
    NIR_X(r, "</output></layer>\n");
    return v;
}

// fp32 source -> f16 constant (F16C, 8 per step)
static inline nv nir_consth(nir *r, uint32_t rank, const uint32_t *d, const float *src, size_t n)
{
    uint16_t *h;
    size_t i;
    nv v;

    h = malloc(n * 2 + 16);
    if (!h) {
        r->err = 1;
        return nir_val(0, 0, "f16", rank, d);
    }
    for (i = 0; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(h + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; i++) h[i] = (uint16_t)_cvtss_sh(src[i], 0);
    v = nir_const(r, "f16", rank, d, h, n * 2);
    free(h);
    return v;
}

static inline nv nir_i64(nir *r, uint32_t n, const int64_t *vals)
{
    return nir_const(r, "i64", n ? 1 : 0, &n, vals, (n ? n : 1) * 8);
}

// One op: inputs on ports 0..nin-1, output on port nin with the given type and shape. attrs: the <data> attributes.
static inline nv nir_op(nir *r, const char *type, const char *ver, const char *attrs, const nv *in, uint32_t nin,
                        const char *et, uint32_t rank, const uint32_t *d)
{
    uint32_t i;
    nv v;

    v = nir_val(r->id++, nin, et, rank, d);
    NIR_X(r, "<layer id=\"%u\" name=\"n%u\" type=\"%s\" version=\"%s\">", v.id, v.id, type, ver);
    if (attrs && attrs[0]) NIR_X(r, "<data %s/>", attrs);
    NIR_X(r, "<input>");
    for (i = 0; i < nin; i++) {
        nir_port(r, i, &in[i], NULL);
        NIR_E(r, "<edge from-layer=\"%u\" from-port=\"%u\" to-layer=\"%u\" to-port=\"%u\"/>\n", in[i].id, in[i].port, v.id, i);
    }
    NIR_X(r, "</input><output>");
    nir_port(r, nin, &v, NULL);
    NIR_X(r, "</output></layer>\n");
    return v;
}

static inline void nir_result(nir *r, nv x)
{
    uint32_t id;

    id = r->id++;
    NIR_X(r, "<layer id=\"%u\" name=\"out\" type=\"Result\" version=\"opset1\"><input>", id);
    nir_port(r, 0, &x, NULL);
    NIR_X(r, "</input></layer>\n");
    NIR_E(r, "<edge from-layer=\"%u\" from-port=\"%u\" to-layer=\"%u\" to-port=\"0\"/>\n", x.id, x.port, id);
}

#define NIR_D(...) ((const uint32_t[]){__VA_ARGS__})
#define NIR_IN(...) ((const nv[]){__VA_ARGS__})
#define NIR_N(...) (sizeof((const nv[]){__VA_ARGS__}) / sizeof(nv))
// binary elementwise op, numpy broadcast, output shaped like a
#define NIR_BIN(r, type, a, b) nir_op(r, type, "opset1", "auto_broadcast=\"numpy\"", NIR_IN(a, b), 2, (a).et, (a).rank, (a).d)
// Y [rows][n] = X [rows][k] . W [n][k]^T
#define NIR_MM(r, X, W) nir_op(r, "MatMul", "opset1", "transpose_a=\"false\" transpose_b=\"true\"", NIR_IN(X, W), 2, \
                               "f16", 2, NIR_D((X).d[0], (W).d[0]))
#define NIR_RESHAPE(r, X, rank, ...)                                                    \
    nir_op(r, "Reshape", "opset1", "special_zero=\"false\"",                            \
           NIR_IN(X, nir_i64(r, rank, (const int64_t[]){__VA_ARGS__})), 2, (X).et, rank, NIR_D(__VA_ARGS__))

// LayerNorm over the last axis (n features of rows): MVN (inside-sqrt eps, as the engines) * gamma + beta
static inline nv nir_ln(nir *r, nv x, const float *w, const float *b, uint32_t n, float eps)
{
    char a[96];
    nv m;

    snprintf(a, sizeof(a), "eps=\"%g\" normalize_variance=\"true\" eps_mode=\"inside_sqrt\"", eps);
    m = nir_op(r, "MVN", "opset6", a, NIR_IN(x, nir_i64(r, 1, (const int64_t[]){-1})), 2, "f16", x.rank, x.d);
    m = NIR_BIN(r, "Multiply", m, nir_consth(r, 1, NIR_D(n), w, n));
    return NIR_BIN(r, "Add", m, nir_consth(r, 1, NIR_D(n), b, n));
}

// ---------------------------------------------------------------- wd-eva02 network from eva02_fp16.bin (v2)
#define NPU_TAKE(off, p, base, bytes)                                                   \
    do {                                                                                \
        (off) = ((off) + 255) & ~(size_t)255;                                           \
        (p) = (base) + (off);                                                           \
        (off) += (bytes);                                                               \
    } while (0)

// RoPE of q or k [T][1024]: x * C + xs * S, C / S [T][1024] f16 (cos / sin; cls row C = 1, S = 0). xs is the
// pair-swapped projection (-x1, x0) per dim pair, produced by its own GEMM against pair-swapped, sign-folded weight
// rows (npu_swap): the NPU's matrix engine does in ~1 ms what a Gather on a [T][512][2] view took ~380 ms for.
static inline nv npu_rope(nir *r, nv x, nv xs, nv C, nv S)
{
    return NIR_BIN(r, "Add", NIR_BIN(r, "Multiply", x, C), NIR_BIN(r, "Multiply", xs, S));
}

// W [n][k] fp16 -> rows (2j, 2j + 1) := (-W[2j + 1], W[2j]): h . sw^T is the pair-swapped, sign-folded projection
static inline void npu_swap(uint16_t *sw, const uint8_t *w, uint32_t n, uint32_t k)
{
    const uint16_t *s;
    uint32_t i, j;

    s = (const uint16_t *)w;
    for (i = 0; i < n; i += 2)
        for (j = 0; j < k; j++) {
            sw[(size_t)i * k + j] = s[(size_t)(i + 1) * k + j] ^ 0x8000;
            sw[(size_t)(i + 1) * k + j] = s[(size_t)i * k + j];
        }
}

// [T][1024] -> [16][T][64]
static inline nv npu_heads(nir *r, nv x, nv perm)
{
    x = NIR_RESHAPE(r, x, 3, EVA_T, EVA_NH, 64);
    return nir_op(r, "Transpose", "opset1", "", NIR_IN(x, perm), 2, "f16", 3, NIR_D(EVA_NH, EVA_T, 64));
}

static inline int npu_eva_ir(nir *r, const uint8_t *blob, uint32_t nt, uint32_t ntp)
{
    const uint8_t *pe, *pos, *cs, *ln1w, *ln1b, *wqkv, *wo, *ln2w, *ln2b, *w12, *mlnw, *mlnb, *w2, *fnw, *fnb, *hw, *hb;
    uint16_t *gw, *xw, *w2s, *sw;
    float *Ct, *St, ln2;
    size_t off;
    uint32_t i, l, c, u, j;
    nv x, h, q, k, v, s, o, C, S, perm, g;

    off = 0;
    NPU_TAKE(off, pe, blob, (size_t)EVA_D * EVA_K0P * 2);
    NPU_TAKE(off, pos, blob, (size_t)EVA_T * EVA_D * 4);
    NPU_TAKE(off, cs, blob, (size_t)EVA_NP * 32 * 2 * 4);
    gw = malloc((size_t)NPU_HID * EVA_D * 2);
    xw = malloc((size_t)NPU_HID * EVA_D * 2);
    w2s = malloc((size_t)EVA_D * NPU_HID * 2);
    sw = malloc((size_t)EVA_D * EVA_D * 2);
    Ct = malloc((size_t)EVA_T * EVA_D * 4);
    St = malloc((size_t)EVA_T * EVA_D * 4);
    if (!gw || !xw || !w2s || !sw || !Ct || !St) return -1;
    // RoPE tables over all 16 heads; row 0 (cls) is the identity
    for (i = 0; i < EVA_T * EVA_D; i++) {
        l = i / EVA_D;
        j = (i % 64) >> 1;
        Ct[i] = l ? ((const float *)cs)[((l - 1) * 32 + j) * 2] : 1.0f;
        St[i] = l ? ((const float *)cs)[((l - 1) * 32 + j) * 2 + 1] : 0.0f;
    }

    x = nir_param(r, "x", "f16", 2, NIR_D(EVA_NP, EVA_K0P));
    x = NIR_MM(r, x, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_K0P), pe, (size_t)EVA_D * EVA_K0P * 2));
    x = nir_op(r, "Concat", "opset1", "axis=\"0\"", NIR_IN(nir_const(r, "f16", 2, NIR_D(1, EVA_D), NULL, EVA_D * 2), x), 2,
               "f16", 2, NIR_D(EVA_T, EVA_D));
    x = NIR_BIN(r, "Add", x, nir_consth(r, 2, NIR_D(EVA_T, EVA_D), (const float *)pos, (size_t)EVA_T * EVA_D));
    C = nir_consth(r, 2, NIR_D(EVA_T, EVA_D), Ct, (size_t)EVA_T * EVA_D);
    S = nir_consth(r, 2, NIR_D(EVA_T, EVA_D), St, (size_t)EVA_T * EVA_D);
    perm = nir_i64(r, 3, (const int64_t[]){1, 0, 2});
    ln2 = 0.69314718f;
    for (l = 0; l < EVA_NL; l++) {
        NPU_TAKE(off, ln1w, blob, EVA_D * 4);
        NPU_TAKE(off, ln1b, blob, EVA_D * 4);
        NPU_TAKE(off, wqkv, blob, (size_t)3 * EVA_D * EVA_D * 2);
        NPU_TAKE(off, wo, blob, (size_t)EVA_D * EVA_D * 2);
        NPU_TAKE(off, ln2w, blob, EVA_D * 4);
        NPU_TAKE(off, ln2b, blob, EVA_D * 4);
        NPU_TAKE(off, w12, blob, (size_t)2 * EVA_HIDP * EVA_D * 2);
        NPU_TAKE(off, mlnw, blob, EVA_HIDP * 4);
        NPU_TAKE(off, mlnb, blob, EVA_HIDP * 4);
        NPU_TAKE(off, w2, blob, (size_t)EVA_D * EVA_HIDP * 2);
        // attention
        h = nir_ln(r, x, (const float *)ln1w, (const float *)ln1b, EVA_D, 1e-6f);
        q = NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_D), wqkv, (size_t)EVA_D * EVA_D * 2));
        k = NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_D), wqkv + (size_t)EVA_D * EVA_D * 2,
                                   (size_t)EVA_D * EVA_D * 2));
        v = NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_D), wqkv + (size_t)2 * EVA_D * EVA_D * 2,
                                   (size_t)EVA_D * EVA_D * 2));
        npu_swap(sw, wqkv, EVA_D, EVA_D);
        s = NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_D), sw, (size_t)EVA_D * EVA_D * 2));
        q = npu_heads(r, npu_rope(r, q, s, C, S), perm);
        npu_swap(sw, wqkv + (size_t)EVA_D * EVA_D * 2, EVA_D, EVA_D);
        s = NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_D), sw, (size_t)EVA_D * EVA_D * 2));
        k = npu_heads(r, npu_rope(r, k, s, C, S), perm);
        v = npu_heads(r, v, perm);
        s = nir_op(r, "MatMul", "opset1", "transpose_a=\"false\" transpose_b=\"true\"", NIR_IN(q, k), 2, "f16", 3,
                   NIR_D(EVA_NH, EVA_T, EVA_T));
        s = NIR_BIN(r, "Multiply", s, nir_consth(r, 1, NIR_D(1), &ln2, 1)); // q carries log2 e: back to natural
        s = nir_op(r, "Softmax", "opset8", "axis=\"-1\"", NIR_IN(s), 1, "f16", 3, s.d);
        o = nir_op(r, "MatMul", "opset1", "transpose_a=\"false\" transpose_b=\"false\"", NIR_IN(s, v), 2, "f16", 3,
                   NIR_D(EVA_NH, EVA_T, 64));
        o = nir_op(r, "Transpose", "opset1", "", NIR_IN(o, perm), 2, "f16", 3, NIR_D(EVA_T, EVA_NH, 64));
        o = NIR_RESHAPE(r, o, 2, EVA_T, EVA_D);
        x = NIR_BIN(r, "Add", x, NIR_MM(r, o, nir_const(r, "f16", 2, NIR_D(EVA_D, EVA_D), wo, (size_t)EVA_D * EVA_D * 2)));
        // MLP: W12 rows are g / x interleaved in 16-unit chunks; split them and drop the 6 pad units
        for (c = 0; c < EVA_HIDP / 16; c++)
            for (u = 0; u < 16 && 16 * c + u < NPU_HID; u++) {
                memcpy(gw + (size_t)(16 * c + u) * EVA_D, w12 + ((size_t)32 * c + u) * EVA_D * 2, EVA_D * 2);
                memcpy(xw + (size_t)(16 * c + u) * EVA_D, w12 + ((size_t)32 * c + 16 + u) * EVA_D * 2, EVA_D * 2);
            }
        for (i = 0; i < EVA_D; i++) memcpy(w2s + (size_t)i * NPU_HID, w2 + (size_t)i * EVA_HIDP * 2, NPU_HID * 2);
        h = nir_ln(r, x, (const float *)ln2w, (const float *)ln2b, EVA_D, 1e-6f);
        g = NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(NPU_HID, EVA_D), gw, (size_t)NPU_HID * EVA_D * 2));
        g = nir_op(r, "Swish", "opset4", "", NIR_IN(g), 1, "f16", 2, g.d);
        g = NIR_BIN(r, "Multiply", g, NIR_MM(r, h, nir_const(r, "f16", 2, NIR_D(NPU_HID, EVA_D), xw,
                                                              (size_t)NPU_HID * EVA_D * 2)));
        g = nir_ln(r, g, (const float *)mlnw, (const float *)mlnb, NPU_HID, 1e-6f);
        x = NIR_BIN(r, "Add", x, NIR_MM(r, g, nir_const(r, "f16", 2, NIR_D(EVA_D, NPU_HID), w2s,
                                                        (size_t)EVA_D * NPU_HID * 2)));
    }
    NPU_TAKE(off, fnw, blob, EVA_D * 4);
    NPU_TAKE(off, fnb, blob, EVA_D * 4);
    NPU_TAKE(off, hw, blob, (size_t)ntp * EVA_D * 2);
    NPU_TAKE(off, hb, blob, (size_t)ntp * 4);
    // mean of the 1024 patch tokens, LN, head, sigmoid -> fp32 [1][nt]
    x = nir_op(r, "Slice", "opset8", "", NIR_IN(x, nir_i64(r, 1, (const int64_t[]){1}),
                                                 nir_i64(r, 1, (const int64_t[]){EVA_T}),
                                                 nir_i64(r, 1, (const int64_t[]){1}), nir_i64(r, 1, (const int64_t[]){0})),
               5, "f16", 2, NIR_D(EVA_NP, EVA_D));
    x = nir_op(r, "ReduceMean", "opset1", "keep_dims=\"true\"", NIR_IN(x, nir_i64(r, 1, (const int64_t[]){0})), 2, "f16",
               2, NIR_D(1, EVA_D));
    x = nir_ln(r, x, (const float *)fnw, (const float *)fnb, EVA_D, 1e-6f);
    x = NIR_MM(r, x, nir_const(r, "f16", 2, NIR_D(nt, EVA_D), hw, (size_t)nt * EVA_D * 2));
    x = NIR_BIN(r, "Add", x, nir_consth(r, 1, NIR_D(nt), (const float *)hb, nt));
    x = nir_op(r, "Sigmoid", "opset1", "", NIR_IN(x), 1, "f16", 2, x.d);
    x = nir_op(r, "Convert", "opset1", "destination_type=\"f32\"", NIR_IN(x), 1, "f32", 2, x.d);
    nir_result(r, x);
    free(gw);
    free(xw);
    free(w2s);
    free(sw);
    free(Ct);
    free(St);
    return r->err ? -1 : 0;
}

// ---------------------------------------------------------------- engine
#define NPU_ZK(call)                                                                    \
    do {                                                                                \
        ze_result_t r_ = (call);                                                        \
        if (r_) {                                                                       \
            fprintf(stderr, "npu: %s: 0x%x\n", #call, (unsigned)r_);                    \
            return -1;                                                                  \
        }                                                                               \
    } while (0)

// IR + weights -> NGRAPH_LITE container [compiler version][u32 2][u64 xml][xml][u64 weights][weights]
static inline uint8_t *npu_container(const ze_graph_compiler_version_info_t *cv, const char *path, size_t *size)
{
    FILE *f;
    uint8_t *file, *buf;
    uint32_t hdr[4], two;
    uint64_t xs, ws;
    nir r;
    long len;

    f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    file = malloc((size_t)len);
    if (!file || fread(file, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(file);
        return NULL;
    }
    fclose(f);
    memcpy(hdr, file, 16);
    if (memcmp(hdr, "EVA2", 4) || hdr[1] != 2) {
        fprintf(stderr, "npu: %s is not an eva02.bin v2 (fp16 weights)\n", path);
        free(file);
        return NULL;
    }
    memset(&r, 0, sizeof(r));
    NIR_X(&r, "<?xml version=\"1.0\"?>\n<net name=\"wd_eva02\" version=\"11\">\n<layers>\n");
    if (npu_eva_ir(&r, file + 256, hdr[2], (hdr[2] + 7) & ~7u)) {
        free(file);
        return NULL;
    }
    free(file);
    NIR_X(&r, "</layers>\n<edges>\n%.*s</edges>\n</net>\n", (int)r.en, r.e);
    xs = r.xn;
    ws = r.wn;
    *size = 4 + 4 + 8 + xs + 8 + ws;
    buf = r.err ? NULL : malloc(*size);
    if (buf) {
        two = 2;
        memcpy(buf, cv, 4);
        memcpy(buf + 4, &two, 4);
        memcpy(buf + 8, &xs, 8);
        memcpy(buf + 16, r.x, xs);
        memcpy(buf + 16 + xs, &ws, 8);
        memcpy(buf + 24 + xs, r.w, ws);
    }
    free(r.x);
    free(r.e);
    free(r.w);
    return buf;
}

// Compiled graph from <path>.npu when it matches this compiler, else compile the IR and write the cache.
static inline int npu_graph(npu *n, const ze_device_graph_properties_t *gp)
{
    ze_graph_desc_t d;
    FILE *f;
    char name[1024];
    uint8_t *buf;
    uint32_t hdr[4];
    size_t size;
    uint64_t bs;
    int r;

    snprintf(name, sizeof(name), "%s.npu", n->path);
    memset(&d, 0, sizeof(d));
    d.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
    buf = NULL;
    if ((f = fopen(name, "rb"))) {
        if (fread(hdr, 4, 4, f) == 4 && hdr[0] == NPU_MAGIC && !memcmp(&hdr[1], &gp->compilerVersion, 4) && hdr[2] == n->nt
            && hdr[3] == NPU_GVER
            && fread(&bs, 8, 1, f) == 1 && (buf = malloc(bs)) && fread(buf, 1, bs, f) == bs) {
            d.format = ZE_GRAPH_FORMAT_NATIVE;
            d.inputSize = bs;
            d.pInput = buf;
        }
        fclose(f);
    }
    if (!d.pInput) {
        free(buf);
        fprintf(stderr, "npu: compiling wd for the NPU (once; cached in %s)\n", name);
        if (!(buf = npu_container(&gp->compilerVersion, n->path, &size))) return -1;
        d.format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
        d.inputSize = size;
        d.pInput = buf;
        d.pBuildFlags = "--inputs_precisions=\"0:FP16\" --inputs_layouts=\"0:NC\" --outputs_precisions=\"0:FP32\" "
                        "--outputs_layouts=\"0:NC\" --config PERFORMANCE_HINT=\"THROUGHPUT\"";
        n->compiled = 1;
    }
    r = n->g->pfnCreate(n->ctx, n->dev, &d, &n->graph);
    if (d.format == ZE_GRAPH_FORMAT_NATIVE) n->blob = buf; // freed in npu_close, after the graph
    else free(buf);
    if (r) {
        fprintf(stderr, "npu: graph create failed (0x%x)\n", (unsigned)r);
        return -1;
    }
    if (!n->compiled || n->g->pfnGetNativeBinary(n->graph, &size, NULL) || !(buf = malloc(size))) return 0;
    if (!n->g->pfnGetNativeBinary(n->graph, &size, buf) && (f = fopen(name, "wb"))) {
        hdr[0] = NPU_MAGIC;
        memcpy(&hdr[1], &gp->compilerVersion, 4);
        hdr[2] = n->nt;
        hdr[3] = NPU_GVER;
        bs = size;
        fwrite(hdr, 4, 4, f);
        fwrite(&bs, 8, 1, f);
        fwrite(buf, 1, size, f);
        fclose(f);
    }
    free(buf);
    return 0;
}

// Bring-up on the boot thread: loader, driver, graph extension, graph (cached or compiled), graph initialisation,
// slots with recorded command lists.
static inline int npu_up(npu *n)
{
    ze_driver_extension_properties_t ext[64];
    ze_device_graph_properties_t gp;
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0};
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                  ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_desc_t ld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, 0, 0};
    ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL, 0};
    ze_fence_desc_t fd = {ZE_STRUCTURE_TYPE_FENCE_DESC, NULL, 0};
    ze_command_list_handle_t init;
    uint32_t cnt, i, best;

    if (!(n->dll = LoadLibraryA("ze_loader.dll"))) {
        fprintf(stderr, "npu: no ze_loader.dll (Intel NPU driver not installed)\n");
        return -1;
    }
#define NPU_LOAD(f) if (!(n->z.f = (__typeof__(&f))(void (*)(void))GetProcAddress(n->dll, #f))) return -1;
    NPU_ZE(NPU_LOAD)
#undef NPU_LOAD
    NPU_ZK(n->z.zeInit(ZE_INIT_FLAG_VPU_ONLY));
    cnt = 1;
    NPU_ZK(n->z.zeDriverGet(&cnt, &n->drv));
    cnt = 1;
    NPU_ZK(n->z.zeDeviceGet(n->drv, &cnt, &n->dev));
    cnt = 64;
    NPU_ZK(n->z.zeDriverGetExtensionProperties(n->drv, &cnt, ext));
    for (i = 0, best = cnt; i < cnt; i++)
        if (!strncmp(ext[i].name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME))
            && (best == cnt || ext[i].version > ext[best].version)) best = i;
    if (best == cnt) {
        fprintf(stderr, "npu: driver has no %s\n", ZE_GRAPH_EXT_NAME);
        return -1;
    }
    NPU_ZK(n->z.zeDriverGetExtensionFunctionAddress(n->drv, ext[best].name, (void **)&n->g));
    memset(&gp, 0, sizeof(gp));
    gp.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    NPU_ZK(n->g->pfnDeviceGetGraphProperties(n->dev, &gp));
    NPU_ZK(n->z.zeContextCreate(n->drv, &cd, &n->ctx));
    if (npu_graph(n, &gp)) return -1;
    NPU_ZK(n->z.zeCommandQueueCreate(n->ctx, n->dev, &qd, &n->q));
    // the graph is initialised on the device before any execute is recorded against it
    NPU_ZK(n->z.zeCommandListCreate(n->ctx, n->dev, &ld, &init));
    NPU_ZK(n->g->pfnAppendGraphInitialize(init, n->graph, NULL, 0, NULL));
    NPU_ZK(n->z.zeCommandListClose(init));
    NPU_ZK(n->z.zeCommandQueueExecuteCommandLists(n->q, 1, &init, NULL));
    NPU_ZK(n->z.zeCommandQueueSynchronize(n->q, UINT64_MAX));
    n->z.zeCommandListDestroy(init);
    for (i = 0; i < NPU_NBUF; i++) {
        NPU_ZK(n->z.zeMemAllocHost(n->ctx, &hd, EVA_IN * 2, 64, &n->in[i]));
        NPU_ZK(n->z.zeMemAllocHost(n->ctx, &hd, (size_t)n->nt * 4, 64, &n->out[i]));
        NPU_ZK(n->g->pfnSetArgumentValue(n->graph, 0, n->in[i]));
        NPU_ZK(n->g->pfnSetArgumentValue(n->graph, 1, n->out[i]));
        NPU_ZK(n->z.zeCommandListCreate(n->ctx, n->dev, &ld, &n->cl[i]));
        NPU_ZK(n->g->pfnAppendGraphExecute(n->cl[i], n->graph, NULL, NULL, 0, NULL));
        NPU_ZK(n->z.zeCommandListClose(n->cl[i]));
        NPU_ZK(n->z.zeFenceCreate(n->q, &fd, &n->f[i]));
    }
    return 0;
}

static DWORD WINAPI npu_boot(void *arg)
{
    npu *n;

    n = arg;
    n->bstatus = npu_up(n);
    return 0;
}

// Starts the bring-up thread; the NPU takes work once npu_ready says so (a first-run compile never holds up the GPU).
static inline int npu_open(npu *n, const char *weights, uint32_t nt)
{
    memset(n, 0, sizeof(*n));
    n->path = weights;
    n->nt = nt;
    n->bstatus = -1;
    n->boot = CreateThread(NULL, 0, npu_boot, n, 0, NULL);
    return n->boot ? 0 : -1;
}

// Non-blocking: 1 once the NPU is up (joins the boot thread once); a failed bring-up stays 0 for good.
static inline int npu_ready(npu *n)
{
    if (n->up || !n->boot) return n->up;
    if (WaitForSingleObject(n->boot, 0) != WAIT_OBJECT_0) return 0;
    CloseHandle(n->boot);
    n->boot = NULL;
    n->up = !n->bstatus;
    if (!n->up) fprintf(stderr, "npu: bring-up failed, running without it\n");
    return n->up;
}

// Free slot index, or -1.
static inline int npu_free(npu *n)
{
    uint32_t f;

    f = ~n->busy & ((1u << NPU_NBUF) - 1);
    return npu_ready(n) && f ? __builtin_ctz(f) : -1;
}

// Copy one model input (EVA_IN halves) into slot s and launch it.
static inline int npu_submit(npu *n, uint32_t s, const uint16_t *input)
{
    memcpy(n->in[s], input, EVA_IN * 2);
    NPU_ZK(n->z.zeCommandQueueExecuteCommandLists(n->q, 1, &n->cl[s], n->f[s]));
    n->busy |= 1u << s;
    return 0;
}

// 1 when slot s finished (block: wait for it). Its scores stay in npu_out until npu_release.
static inline int npu_done(npu *n, uint32_t s, uint32_t block)
{
    if (!(n->busy >> s & 1)) return 0;
    if (block) return n->z.zeFenceHostSynchronize(n->f[s], UINT64_MAX) == ZE_RESULT_SUCCESS;
    return n->z.zeFenceQueryStatus(n->f[s]) == ZE_RESULT_SUCCESS;
}

static inline const float *npu_out(npu *n, uint32_t s) { return (const float *)n->out[s]; }

static inline void npu_release(npu *n, uint32_t s)
{
    n->z.zeFenceReset(n->f[s]);
    n->busy &= ~(1u << s);
}

static inline void npu_close(npu *n)
{
    uint32_t i;

    if (n->boot) {
        WaitForSingleObject(n->boot, INFINITE);
        CloseHandle(n->boot);
        n->boot = NULL;
    }
    if (!n->dll) return;
    if (n->q) n->z.zeCommandQueueSynchronize(n->q, UINT64_MAX);
    for (i = 0; i < NPU_NBUF; i++) {
        if (n->f[i]) n->z.zeFenceDestroy(n->f[i]);
        if (n->cl[i]) n->z.zeCommandListDestroy(n->cl[i]);
        if (n->in[i]) n->z.zeMemFree(n->ctx, n->in[i]);
        if (n->out[i]) n->z.zeMemFree(n->ctx, n->out[i]);
    }
    if (n->graph) n->g->pfnDestroy(n->graph);
    free(n->blob);
    if (n->q) n->z.zeCommandQueueDestroy(n->q);
    if (n->ctx) n->z.zeContextDestroy(n->ctx);
    FreeLibrary(n->dll);
}

#endif // NPU_H
