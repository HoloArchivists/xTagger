#ifndef HP_H
#define HP_H
// 8-bit hashed perceptrons, 32 outputs per ymm. Every feature owns a table of 1 << rb rows; a row is 32 int8 weights,
// one per output, so a feature lookup is one aligned ymm load and the 32 dot products are a running sum of rows.
// Row index = CRC32C(feature seed, value) (SSE4.2), so features are arbitrary uint32 values: quantize, don't scale.
// Training is the perceptron rule with Jimenez's threshold: an output with a label is updated when it is wrong or its
// |sum| < theta, by +-1 into every selected row at once (saturating byte add: the weights are saturating counters).
//
//   hp_init(&p, nf, rb)                       nf features, 2^rb rows each
//   hp_index(&p, x, idx)                      x[nf] feature values -> idx[nf] row numbers (keep them for training)
//   y = hp_sum(&p, idx)                       32 sums (int16, even outputs in y.e, odd in y.o)
//   hp_ge(y, k)                               bit o set when sum o >= k
//   hp_train(&p, idx, t, m)                   learn targets t (bit o: output o should be positive) on outputs m

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    __m256i *w;      // [nf << rb] rows
    uint32_t nf, rb;
    int32_t theta;
} hp;

typedef struct { __m256i e, o; } hp_y; // int16 sums: e[i] = output 2i, o[i] = output 2i + 1

// 4 * log2(x) for x >= 1, quarter-octave buckets from the leading one and the next two bits (0 for x = 0)
#define HP_LOG4(x) ((x) ? 4u * (31u - (uint32_t)__builtin_clz(x)) + (((x) << __builtin_clz(x)) >> 29 & 3u) : 0u)

static inline int hp_init(hp *p, uint32_t nf, uint32_t rb)
{
    p->nf = nf;
    p->rb = rb;
    p->theta = (int32_t)(1.93f * (float)nf + 14.0f);
    p->w = _mm_malloc((size_t)nf << rb << 5, 64);
    if (!p->w) return -1;
    memset(p->w, 0, (size_t)nf << rb << 5);
    return 0;
}

static inline void hp_free(hp *p)
{
    _mm_free(p->w);
    p->w = NULL;
}

static inline void hp_index(const hp *p, const uint32_t *x, uint32_t *idx)
{
    uint32_t f, m;

    m = (1u << p->rb) - 1;
    for (f = 0; f < p->nf; f++) idx[f] = f << p->rb | (_mm_crc32_u32(0x9E3779B9u * (f + 1), x[f]) & m);
}

// Even bytes sign-extended in place (shift up, arithmetic shift down), odd bytes by one arithmetic shift: no shuffles.
static inline hp_y hp_sum(const hp *p, const uint32_t *idx)
{
    hp_y y;
    __m256i r;
    uint32_t f;

    y.e = y.o = _mm256_setzero_si256();
    for (f = 0; f < p->nf; f++) {
        r = _mm256_load_si256(p->w + idx[f]);
        y.e = _mm256_add_epi16(y.e, _mm256_srai_epi16(_mm256_slli_epi16(r, 8), 8));
        y.o = _mm256_add_epi16(y.o, _mm256_srai_epi16(r, 8));
    }
    return y;
}

// 16 int16 compare masks (2 movemask bits per lane) -> 16 bits, then even / odd interleaved back into output order
#define HP_BITS(ce, co) (_pdep_u32(_pext_u32((uint32_t)_mm256_movemask_epi8(ce), 0x55555555u), 0x55555555u) \
                       | _pdep_u32(_pext_u32((uint32_t)_mm256_movemask_epi8(co), 0x55555555u), 0xAAAAAAAAu))

static inline uint32_t hp_ge(hp_y y, int32_t k)
{
    __m256i v;

    v = _mm256_set1_epi16((int16_t)(k - 1));
    return HP_BITS(_mm256_cmpgt_epi16(y.e, v), _mm256_cmpgt_epi16(y.o, v));
}

// One sum as a scalar (diagnostics / stats).
static inline int32_t hp_get(hp_y y, uint32_t o)
{
    int16_t s[16];

    _mm256_storeu_si256((__m256i *)s, o & 1 ? y.o : y.e);
    return s[o >> 1];
}

// 32 mask bits -> 32 bytes of 0x00 / 0xFF
static inline __m256i hp_expand(uint32_t m)
{
    __m256i v, sel;

    v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)m),
                            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                             2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    sel = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, sel), sel);
}

static inline void hp_train(hp *p, const uint32_t *idx, uint32_t t, uint32_t m)
{
    hp_y y;
    __m256i d;
    uint32_t pos, strong, f;

    y = hp_sum(p, idx);
    pos = hp_ge(y, 1);
    strong = hp_ge(y, p->theta) | ~hp_ge(y, 1 - p->theta);
    m &= (pos ^ t) | ~strong;
    if (!m) return;
    // +1 where the target is positive, -1 where negative, 0 off the update mask
    d = _mm256_and_si256(hp_expand(m), _mm256_or_si256(_mm256_xor_si256(hp_expand(t), _mm256_set1_epi8(-1)),
                                                       _mm256_set1_epi8(1)));
    for (f = 0; f < p->nf; f++) _mm256_store_si256(p->w + idx[f], _mm256_adds_epi8(_mm256_load_si256(p->w + idx[f]), d));
}

#endif // HP_H
