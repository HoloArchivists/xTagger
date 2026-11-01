#ifndef STEAL_H
#define STEAL_H
// Tag stealing: wd's head learns online from PixAI's outputs (procimg.h, -s). wd's head is one linear layer over the
// pooled, normed embedding x (1024 fp16, read back with every batch): z_j = w_j . x + b_j, score = sigmoid(z_j). The
// backbone stays frozen; only head rows move, so a training event costs a few dot products and AXPYs on the CPU and
// a 2 KB upload per changed row.
//
// Targets are threshold-relative: PixAI's margin over its own threshold in logit space, m = logit(p) - logit(thr_px),
// becomes wd's target z* = logit(thr_wd) + m, so a PixAI decision becomes the same wd decision with the same margin
// and neither model's thresholds need tuning. Only confident PixAI decisions teach (|m| >= ST_G), only in the
// categories stolen (-sk), headcount tags never (PixAI contradicts itself on those; wd is right).
//
// Rows move by normalised LMS (the embedding is layer-normed, so a fixed step converges), perceptron-style: a row
// moves only while wd's decision is wrong or inside ST_MARGIN of the threshold on the right side.
//
// PixAI tags wd's head does not have get a new row the first time PixAI is confident (|m| >= ST_GI) about a whole
// image wd also embedded (imprinting): w = ST_BETA (x - mu) / |x - mu|^2, b set so z = logit(thr) + ST_BETA *
// (cos - ST_KAPPA), where cos is the projection on the imprint image (1 there, ~0 on unrelated images) and mu the
// running mean embedding. The row is usable from that image on; later events refine it. Each new row carries
// PixAI's name; output dedupes by name as before (no normalisation of either vocabulary).
//
// Events (procimg pi_steal). The xorvariance MVIs are what make stealing work on crowded frames: an explored job runs
// both models on the same crops, so wd crop i and PixAI crop i are one instance seen twice, and a character PixAI
// finds in crop i teaches (or imprints) from crop i's embedding, not from the whole scene's. Explored: full vs full
// plus every crop vs its PixAI twin; px-crops: wd full vs PixAI's crop maxima, positives only (a tag in a crop is in
// the image); px-full: PixAI full against wd's crops (multiple-instance: a positive teaches the crop wd already
// likes most, a negative every crop that fires).
//
// -sf <file>: rows that moved, the new rows (by PixAI name) and mu persist across runs. Sidecar tags (tagg.h) share
// the extra rows and the file.

#define ST_G      1.0f   // logit margin over PixAI's threshold for a decision to teach
#define ST_GI     2.0f   // ... to create a new row
#define ST_M      4.0f   // target clamp: |z* - logit(thr_wd)| <= ST_M (no chasing saturated sigmoids)
#define ST_MARGIN 1.0f   // a row keeps moving until its z clears the threshold by this much
#define ST_ETA    0.5f   // NLMS step (share of the error removed per event)
#define ST_BETA   8.0f   // imprint: logit swing between unrelated images and the imprint image
#define ST_KAPPA  0.5f   // imprint: projection at which a new row crosses its threshold
#define ST_MU0    8      // embeddings in mu before imprinting is allowed
#define ST_MAGIC  0x31535458u // "XTS1"
// Instance scope: series, artist and meta tags describe the whole image (PixAI calls a crop "original" when the crop
// alone carries no series cues), so they teach only from whole-image instances; characters and general tags teach
// from crops too.
#define ST_FULL   1u
#define ST_CROP   2u
#define ST_SCOPE(cat) ((cat) == 0 || (cat) == 4 ? ST_FULL | ST_CROP : ST_FULL)

typedef struct {
    float *w, *b;            // fp32 master rows [nt][EVA_D], biases; row j valid once have[j]
    float *lt, *plt;         // logit of each wd row's / PixAI tag's threshold
    float *phi, *plo;        // PixAI scores at plt +- ST_G: the teach test without a logf per tag
    float *whi, *wlo;        // wd scores at lt +- ST_MARGIN (unused rows: 2, never passed)
    uint8_t *have, *dirty, *ok; // ok: where PixAI tag k may teach, ST_FULL | ST_CROP (0: not stolen / headcount)
    uint32_t *dl, nd;        // rows to upload, in order
    uint16_t *h16;           // the file's head (fp16), fetched on first use
    float *hb16;
    int32_t *pw, *wp;        // PixAI tag -> wd row and back, -1 none (yet)
    uint32_t *seen;          // NLMS updates per row
    float mu[EVA_D];         // running mean of wd full-image embeddings
    float x[EVA_D];          // event scratch: one embedding in fp32
    uint16_t stage[EVA_D];   // upload staging
    uint32_t nmu, nt0, nt, ns, cap, nimp, nupd, nev;
    const char *file;
    float tchar, tgen;
} st_learn;

static inline float st_logit(float p)
{
    p = p < 1e-4f ? 1e-4f : p > 1.0f - 1e-4f ? 1.0f - 1e-4f : p;
    return logf(p / (1.0f - p));
}

#define ST_HSUM(v) (_mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(_mm256_castps256_ps128(v), \
    _mm256_extractf128_ps(v, 1)), _mm_setzero_ps()), _mm_setzero_ps())))

static inline float st_dot(const float *a, const float *b)
{
    __m256 s0, s1;
    uint32_t i;

    s0 = s1 = _mm256_setzero_ps();
    for (i = 0; i < EVA_D; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    return ST_HSUM(s0);
}

static inline void st_axpy(float *w, const float *x, float a)
{
    __m256 va;
    uint32_t i;

    va = _mm256_set1_ps(a);
    for (i = 0; i < EVA_D; i += 8) _mm256_storeu_ps(w + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(w + i)));
}

static inline void st_h2f(float *d, const uint16_t *s)
{
    uint32_t i;

    for (i = 0; i < EVA_D; i += 8) _mm256_storeu_ps(d + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + i))));
}

#define ST_SIG(z) (1.0f / (1.0f + expf(-(z))))
#define ST_ROW(L, j) ((L)->w + (size_t)(j) * EVA_D)
#define ST_MARK(L, j) ((L)->dirty[j] ? 0 : ((L)->dirty[j] = 1, (L)->dl[(L)->nd++] = (j)))
// Headcount tags (1girl, 6+boys, solo, multiple_girls, ...): wd's call stands.
#define ST_COUNT(s)                                                                     \
    (((s)[0] >= '0' && (s)[0] <= '9' && (strstr(s, "girl") || strstr(s, "boy") || strstr(s, "other"))) \
     || !strncmp(s, "solo", 4) || !strncmp(s, "multiple_", 9) || !strcmp(s, "head_out_of_frame"))

// Master row j valid: the file's row (fetched once from the device), or a new row's zero.
static inline int st_row(st_learn *L, eva *e, uint32_t j)
{
    if (L->have[j]) return 0;
    if (j < L->nt0 && !L->h16) {
        L->h16 = malloc((size_t)L->nt0 * EVA_D * 2);
        L->hb16 = malloc((size_t)L->nt0 * 4);
        if (!L->h16 || !L->hb16 || eva_head_get(e, L->h16, L->hb16)) return -1;
    }
    if (j < L->nt0) {
        st_h2f(ST_ROW(L, j), L->h16 + (size_t)j * EVA_D);
        L->b[j] = L->hb16[j];
    } else {
        memset(ST_ROW(L, j), 0, EVA_D * 4);
        L->b[j] = -16.0f;
    }
    L->have[j] = 1;
    return 0;
}

// One NLMS step of row j on the event's embedding (xx = |x|^2) toward logit zt.
static inline void st_nlms(st_learn *L, uint32_t j, float xx, float zt)
{
    float d;

    d = ST_ETA * (zt - (st_dot(ST_ROW(L, j), L->x) + L->b[j])) / (xx + 1.0f);
    st_axpy(ST_ROW(L, j), L->x, d);
    L->b[j] += d;
    L->seen[j]++;
    L->nupd++;
    ST_MARK(L, j);
}

// Row c of wd's vocabulary becomes PixAI tag k (by name; thresholds from wd's category rules).
static inline void st_adopt(st_learn *L, pi_voc *wv, pi_voc *pv, uint32_t c, uint32_t k)
{
    wv->name[c] = pv->name[k];
    wv->cat[c] = pv->cat[k];
    wv->hash[c] = pv->hash[k];
    wv->thr[c] = pv->cat[k] == 4 ? L->tchar : L->tgen;
    wv->chr[c] = pv->cat[k] == 4 ? 1.0f : 0.0f;
    wv->mvimask[c] = 1.0f;
    wv->shared[c] = pv->shared[k] = 1;
    L->lt[c] = st_logit(wv->thr[c]);
    L->whi[c] = ST_SIG(L->lt[c] + ST_MARGIN);
    L->wlo[c] = ST_SIG(L->lt[c] - ST_MARGIN);
    L->pw[k] = (int32_t)c;
    L->wp[c] = (int32_t)k;
}

// New row for PixAI tag k from the embedding in L->x (see the header).
static inline void st_imprint(st_learn *L, pi_voc *wv, pi_voc *pv, uint32_t k)
{
    float *w, d2;
    uint32_t c, i;

    c = L->nt0 + L->ns++;
    st_adopt(L, wv, pv, c, k);
    w = ST_ROW(L, c);
    for (i = 0; i < EVA_D; i++) w[i] = L->x[i] - L->mu[i];
    d2 = st_dot(w, w) + 1e-6f;
    for (i = 0; i < EVA_D; i++) w[i] *= ST_BETA / d2;
    L->b[c] = L->lt[c] - ST_BETA * ST_KAPPA - st_dot(w, L->mu);
    L->have[c] = 1;
    L->nimp++;
    ST_MARK(L, c);
}

// mu += (x - mu) / n
static inline void st_mean(st_learn *L, const uint16_t *x16)
{
    __m256 a, mu;
    uint32_t i;

    L->nmu++;
    a = _mm256_set1_ps(1.0f / (float)L->nmu);
    for (i = 0; i < EVA_D; i += 8) {
        mu = _mm256_loadu_ps(L->mu + i);
        mu = _mm256_fmadd_ps(a, _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x16 + i))), mu), mu);
        _mm256_storeu_ps(L->mu + i, mu);
    }
}

// Indices i < n with a[i] CMP b[i], 8 per compare (both arrays readable to n rounded up to 8; the bound arrays are
// padded with 2.0, which no score passes).
#define ST_SCAN(n, a, b, CMP, i, k, m, BODY)                                            \
    for ((i) = 0; (i) < (n); (i) += 8)                                                  \
        for ((m) = (uint32_t)_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps((a) + (i)), \
                                                _mm256_loadu_ps((b) + (i)), CMP)); (m); (m) &= (m) - 1) { \
            (k) = (i) + (uint32_t)__builtin_ctz(m);                                     \
            BODY;                                                                       \
        }
#define ST_CLAMP(m) ((m) > ST_M ? ST_M : (m) < -ST_M ? -ST_M : (m))

// PixAI is confident tag k is present (score p) on the instance in L->x (scope: ST_FULL or ST_CROP): a new row, or
// row j pulled up unless wd already clears its threshold by ST_MARGIN.
static inline int st_pos(st_learn *L, eva *e, pi_voc *wv, pi_voc *pv, uint32_t k, float p, const float *ws, float xx,
                         uint32_t scope)
{
    float m;
    int32_t j;

    if (!(L->ok[k] & scope)) return 0;
    m = st_logit(p) - L->plt[k];
    j = L->pw[k];
    if (j < 0) {
        if (m >= ST_GI && L->ns < L->cap && L->nmu >= ST_MU0) st_imprint(L, wv, pv, k);
        return 0;
    }
    if (ws[j] >= L->whi[j]) return 0;
    if (st_row(L, e, (uint32_t)j)) return -1;
    st_nlms(L, (uint32_t)j, xx, L->lt[j] + ST_CLAMP(m));
    return 0;
}

// wd row j fires or nearly does on the instance in L->x: pushed down if PixAI is confident its tag is absent (ps).
static inline int st_neg(st_learn *L, eva *e, uint32_t j, const float *ps, float xx, uint32_t scope)
{
    int32_t k;

    if ((k = L->wp[j]) < 0 || !(L->ok[k] & scope) || ps[k] > L->plo[k]) return 0;
    if (st_row(L, e, j)) return -1;
    st_nlms(L, j, xx, L->lt[j] + ST_CLAMP(st_logit(ps[k]) - L->plt[k]));
    return 0;
}

// One instance seen by both models: wd's embedding x16 and scores ws against PixAI's scores ps on the same pixels
// (scope: ST_FULL whole image, ST_CROP a crop). mode bit 0: PixAI's positives teach; bit 1: PixAI's absences correct
// what wd fires. Candidates come from two AVX2 scans (PixAI's confident positives, wd's firing rows), never a walk
// over either vocabulary.
static inline int st_teach(st_learn *L, eva *e, pi_voc *wv, pi_voc *pv, const uint16_t *x16, const float *ws,
                           const float *ps, uint32_t mode, uint32_t scope)
{
    float xx;
    uint32_t i, k, m;

    st_h2f(L->x, x16);
    xx = st_dot(L->x, L->x);
    if (mode & 1)
        ST_SCAN(pv->nt, ps, L->phi, _CMP_GE_OQ, i, k, m, if (st_pos(L, e, wv, pv, k, ps[k], ws, xx, scope)) return -1);
    if (mode & 2) ST_SCAN(L->nt, ws, L->wlo, _CMP_GT_OQ, i, k, m, if (st_neg(L, e, k, ps, xx, scope)) return -1);
    return 0;
}

// PixAI's whole-image positives against wd's crops only (px-full): the crop wd already scores highest for the tag is
// taught (multiple-instance); no imprinting, the instance is not known.
static inline int st_milpos(st_learn *L, eva *e, uint32_t k, const uint16_t *x16, const float *ws, uint32_t nt,
                            uint32_t mask, float p)
{
    float best;
    int32_t j;
    uint32_t i, bi;

    if (!(L->ok[k] & ST_CROP) || (j = L->pw[k]) < 0) return 0;
    for (i = 0, best = -1.0f, bi = 0; i < 32; i++) {
        bi = (mask >> i & 1) && ws[(size_t)i * nt + j] > best ? i : bi;
        best = (mask >> i & 1) && ws[(size_t)i * nt + j] > best ? ws[(size_t)i * nt + j] : best;
    }
    if (best < 0.0f || best >= L->whi[j]) return 0;
    if (st_row(L, e, (uint32_t)j)) return -1;
    st_h2f(L->x, x16 + (size_t)bi * EVA_D);
    st_nlms(L, (uint32_t)j, st_dot(L->x, L->x), L->lt[j] + ST_CLAMP(st_logit(p) - L->plt[k]));
    return 0;
}

// Upload every row that moved (stream-ordered: the next batch sees them).
static inline int st_flush(st_learn *L, eva *e)
{
    uint32_t i, j, c;

    if (!L->nd) return 0;
    if (eva_ready(e)) return -1;
    for (i = 0; i < L->nd; i++) {
        j = L->dl[i];
        for (c = 0; c < EVA_D; c += 8)
            _mm_storeu_si128((__m128i *)(L->stage + c), _mm256_cvtps_ph(_mm256_loadu_ps(ST_ROW(L, j) + c), 0));
        if (eva_head_put(e, j, L->stage, L->b[j])) return -1;
        L->dirty[j] = 0;
    }
    L->nd = 0;
    return 0;
}

// -sf file: u32 magic, nt0, rows, nmu; float mu[EVA_D]; per row: u32 row, seen, u16 name length, name (new rows
// only), float bias, float w[EVA_D]. User tags (tagg.h: extra rows without a PixAI twin) follow in their own section.
#define ST_KEEP(L, j) ((j) < (L)->nt0 ? (L)->seen[j] != 0 : (j) < (L)->nt0 + (L)->ns && (L)->wp[j] >= 0)
static inline void st_save(st_learn *L, const pi_voc *wv, FILE *f)
{
    uint32_t j, n, hdr[4];
    uint16_t nl;

    for (j = 0, n = 0; j < L->nt; j++) n += ST_KEEP(L, j);
    hdr[0] = ST_MAGIC;
    hdr[1] = L->nt0;
    hdr[2] = n;
    hdr[3] = L->nmu;
    fwrite(hdr, 4, 4, f);
    fwrite(L->mu, 4, EVA_D, f);
    for (j = 0; j < L->nt; j++) {
        if (!ST_KEEP(L, j)) continue;
        nl = j >= L->nt0 ? (uint16_t)strlen(wv->name[j]) : 0;
        fwrite(&j, 4, 1, f);
        fwrite(&L->seen[j], 4, 1, f);
        fwrite(&nl, 2, 1, f);
        fwrite(wv->name[j], 1, nl, f);
        fwrite(&L->b[j], 4, 1, f);
        fwrite(ST_ROW(L, j), 4, EVA_D, f);
    }
}

// PixAI tag named s (len n), -1 if PixAI has no such tag. Load time only.
static inline int32_t st_find(const pi_voc *pv, const char *s, uint32_t n)
{
    uint32_t k;

    for (k = 0; k < pv->nt; k++)
        if (!strncmp(pv->name[k], s, n) && !pv->name[k][n]) return (int32_t)k;
    return -1;
}

// 0 when the file's rows all loaded (f is then at the next section).
static inline int st_load(st_learn *L, pi_voc *wv, pi_voc *pv, FILE *f)
{
    char name[1024];
    uint32_t hdr[4], i, j, seen, c;
    uint16_t nl;
    int32_t k;

    if (!f) return -1;
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != ST_MAGIC || hdr[1] != L->nt0 || fread(L->mu, 4, EVA_D, f) != EVA_D) {
        fprintf(stderr, "steal: %s is not a stolen head for this model, starting fresh\n", L->file);
        memset(L->mu, 0, sizeof(L->mu));
        return -1;
    }
    L->nmu = hdr[3];
    for (i = 0; i < hdr[2]; i++) {
        if (fread(&j, 4, 1, f) != 1 || fread(&seen, 4, 1, f) != 1 || fread(&nl, 2, 1, f) != 1 || nl >= sizeof(name)
            || fread(name, 1, nl, f) != nl)
            break;
        c = j;
        if (j >= L->nt0) { // new row: placed by PixAI name, wherever the next free row is
            k = st_find(pv, name, nl);
            if (k < 0 || L->pw[k] >= 0 || L->ns >= L->cap) {
                fseek(f, 4 * (1 + EVA_D), SEEK_CUR);
                continue;
            }
            c = L->nt0 + L->ns++;
            st_adopt(L, wv, pv, c, (uint32_t)k);
        }
        if (fread(&L->b[c], 4, 1, f) != 1 || fread(ST_ROW(L, c), 4, EVA_D, f) != EVA_D) return -1;
        L->seen[c] = seen;
        L->have[c] = 1;
        ST_MARK(L, c);
    }
    return i == hdr[2] ? 0 : -1;
}

// wv is loaded with room for cap more rows, pv with pi_shared run (or empty: no -px, sidecars only). Extends wv to
// nt0 + cap rows (unused ones never fire: threshold 2), maps PixAI names onto wd rows. The caller runs st_load.
static inline int st_init(st_learn *L, pi_voc *wv, pi_voc *pv, uint32_t cap, uint32_t cats, const char *file,
                          float tchar, float tgen)
{
    uint32_t *tab, m, i, k, n8;

    memset(L, 0, sizeof(*L));
    L->nt0 = wv->nt;
    L->cap = cap;
    L->nt = wv->nt + cap;
    L->file = file;
    L->tchar = tchar;
    L->tgen = tgen;
    L->w = _mm_malloc((size_t)L->nt * EVA_D * 4, 64);
    L->b = malloc((size_t)L->nt * 4);
    L->lt = malloc((size_t)L->nt * 4);
    L->have = calloc(L->nt, 1);
    L->dirty = calloc(L->nt, 1);
    L->dl = malloc((size_t)L->nt * 4);
    L->seen = calloc(L->nt, 4);
    n8 = (L->nt > pv->nt ? L->nt : pv->nt) + 8; // scan bounds readable to a multiple of 8
    L->plt = malloc((size_t)pv->nt * 4 + 32);
    L->phi = malloc((size_t)n8 * 4);
    L->plo = malloc((size_t)pv->nt * 4 + 32);
    L->whi = malloc((size_t)L->nt * 4);
    L->wlo = malloc((size_t)n8 * 4);
    L->ok = malloc(pv->nt + 32);
    L->pw = malloc((size_t)pv->nt * 4 + 32);
    L->wp = malloc((size_t)L->nt * 4);
    if (!L->w || !L->b || !L->lt || !L->have || !L->dirty || !L->dl || !L->seen || !L->plt || !L->phi || !L->plo
        || !L->whi || !L->wlo || !L->ok || !L->pw || !L->wp)
        return -1;
    for (i = 0; i < n8; i++) L->phi[i] = L->wlo[i] = 2.0f;
    for (i = 0; i < L->nt; i++) {
        L->whi[i] = 2.0f;
        L->wp[i] = -1;
    }
    for (i = L->nt0; i < L->nt; i++) {
        wv->name[i] = "";
        wv->cat[i] = 0;
        wv->hash[i] = 0;
        wv->shared[i] = 0;
        wv->thr[i] = 2.0f;
        wv->mvimask[i] = 1.0f;
        wv->chr[i] = 0.0f;
    }
    for (i = 0; i < L->nt0; i++) {
        L->lt[i] = st_logit(wv->thr[i]);
        L->whi[i] = wv->cat[i] == 9 ? 2.0f : ST_SIG(L->lt[i] + ST_MARGIN);
        L->wlo[i] = wv->cat[i] == 9 ? 2.0f : ST_SIG(L->lt[i] - ST_MARGIN);
    }
    for (k = 0; k < pv->nt; k++) {
        L->plt[k] = st_logit(pv->thr[k]);
        L->ok[k] = pv->cat[k] != 9 && (cats >> pv->cat[k] & 1) && !ST_COUNT(pv->name[k]) ? ST_SCOPE(pv->cat[k]) : 0;
        L->phi[k] = L->ok[k] ? ST_SIG(L->plt[k] + ST_G) : 2.0f; // tags that never teach never pass the scan
        L->plo[k] = ST_SIG(L->plt[k] - ST_G);
        L->pw[k] = -1;
    }
    // PixAI name -> wd row: one hash table of wd's names, probed with PixAI's
    m = 1u << (32 - __builtin_clz(2 * L->nt0));
    if (!(tab = calloc(m, 4))) return -1;
    for (i = 0; i < L->nt0; i++) {
        for (k = wv->hash[i] & (m - 1); tab[k]; k = (k + 1) & (m - 1));
        tab[k] = i + 1;
    }
    for (i = 0; i < pv->nt; i++)
        for (k = pv->hash[i] & (m - 1); tab[k]; k = (k + 1) & (m - 1))
            if (wv->hash[tab[k] - 1] == pv->hash[i] && !strcmp(wv->name[tab[k] - 1], pv->name[i])) {
                L->pw[i] = (int32_t)(tab[k] - 1);
                L->wp[tab[k] - 1] = (int32_t)i;
            }
    free(tab);
    wv->nt = L->nt;
    return 0;
}

static inline void st_free(st_learn *L)
{
    _mm_free(L->w);
    free(L->b); free(L->lt); free(L->have); free(L->dirty); free(L->dl); free(L->seen);
    free(L->plt); free(L->phi); free(L->plo); free(L->whi); free(L->wlo); free(L->ok); free(L->pw); free(L->wp);
    free(L->h16); free(L->hb16);
}

#endif // STEAL_H
